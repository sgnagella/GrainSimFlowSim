#define EXTRA_COMPUTABLES (torque)

#include "Contacts.cuh"
// #include "GPU_contacts.cuh"
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

using namespace uammd;
__global__ void gpu_print_debug(){
  printf("Debug kernel executed successfully.\n");
  return;
}

__global__ void gpu_cleanupContacts(
  ContactManager *contact_mgr, // Contact history manager
  ContactManager::HashEntry *hash_table, // Hash table to update
  real time){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= contact_mgr->max_contacts) return;
  // printf("ContactManager pointer: %p\n", contact_mgr);
  // ContactManager::HashEntry *hash_table = contact_mgr->hash_table;
  const int MAX_PROBES = ContactManager::MAX_PROBES;
  const int hash_size = contact_mgr->hash_size;
  // ContactHistory *contacts = contact_mgr->contacts; 
  ContactHistory *contact = &contact_mgr->contacts[idx];
  const int i = contact->particle_i;
  const int j = contact->particle_j;
  // if( i<0 || j<0 ) return; // Invalid contact
  // else if( !contact->is_active ) return; // Never activated contact
  // printf("Active: %d Contact index: %d between particles %d and %d with last active time %f\n", contact->is_active, idx, i, j, contact->contact_age);
  if (contact->is_active && i >= 0 && j >= 0) {
    // printf("Contact[%d]: i=%d, j=%d, age=%f\n", idx, contact->particle_i, contact->particle_j, contact->contact_age); 
    // printf("Evaluating contact at index %d between %d and %d\n", idx, i, j);
    // Prune criterion: time of last active contact 
    // printf("Checking contact between %d and %d at index %d (last active at time %f, current time %f)\n", i, j, idx, contact->contact_age, time);
    if (contact->contact_age < time){
      // printf("Found inactive contact between %d and %d at index %d (last active at time %f, current time %f)\n", i, j, idx, contact->contact_age, time);
      uint64_t key = contact_mgr->pack_key(i, j);
      uint64_t slot = contact_mgr->hash1(key);
      uint64_t stride = contact_mgr->hash2(key);
      unsigned long long desired = contact_mgr->pack_entry(key, idx);

      // Find the hash entry using linear probing
      for (int probe = 0; probe < MAX_PROBES; ++probe){
        uint64_t slot_calc = slot + uint64_t(probe) * stride;
        uint64_t current_slot = slot_calc & (hash_size - 1);
        // Snapshot current packed value
        ContactManager::HashEntry *entry = &hash_table[current_slot];
        unsigned long long cur = entry->packed;
        uint64_t cur_key = contact_mgr->unpack_key(cur);
        // Found existing contact with this key
        // if (cur_key == key and old != ContactManager::EMPTY_PACKED){
        if(cur_key == key){
          unsigned long long old = atomicCAS(&entry->packed, 
                                             desired, 
                                             ContactManager::EMPTY_PACKED);
          // bool is_active = atomicCAS(&contact->is_active, true, false);
          // entry->packed = ContactManager::EMPTY_PACKED;
          // A new active contact will occupy this empty slot
          // printf("Cleaning up contact between %d and %d at index %d\n", i, j, idx); 
          // atomicSub(contact_mgr->contact_count, 1);
          if (old == desired){
            // Successfully removed
            // printf("Cleaned up contact between %d and %d at index %d\n", i, j, idx);
            contact->is_active = false;
            atomicAdd(contact_mgr->cleanup_count, 1);
            return;
          }
          // If we hit an empty slot, contact isn't in hash table
          // if (cur == ContactManager::EMPTY_PACKED) {
          //   // Contact not found in hash - still mark inactive
          //   contact->is_active = false;
          //   atomicAdd(contact_mgr->cleanup_count, 1);
          //   return;
          // }
          // else{
          //   // Failed to remove - another thread modified it
          //   continue;
          // }
        }
      }
    }
  }
  return;
}

__global__ void gpu_rebuildHashTable(
  ContactManager *contact_mgr, // Contact history manager
  ContactManager::HashEntry *hash_table // Hash table to rebuild
  ){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= contact_mgr->max_contacts) return;

  // using HashEntry = ContactManager::HashEntry;
  // HashEntry *hash_table = contact_mgr->hash_table;
  const int MAX_PROBES = ContactManager::MAX_PROBES;
  const int hash_size = contact_mgr->hash_size;
  ContactHistory *contacts = contact_mgr->contacts; 
  ContactHistory *contact = &contacts[idx];
  const int i = contact->particle_i;
  const int j = contact->particle_j;
  if( i<0 || j<0 ) return; // Invalid contact
  else if( !contact->is_active ) return; // Never activated contact

  uint64_t key = contact_mgr->pack_key(i, j); 
  uint64_t slot = contact_mgr->hash1(key); 
  uint64_t stride = contact_mgr->hash2(key);

  // Linear probe to find or insert 
  for(int probe = 0; probe < MAX_PROBES; probe++){
    uint64_t slot_calc = slot + uint64_t(probe) * stride;
    uint64_t current_slot = slot_calc & (hash_size -1);
    ContactManager::HashEntry* entry = &hash_table[current_slot];
    unsigned long long cur = entry->packed;

    if (cur == ContactManager::EMPTY_PACKED){
      // Try to claim the empty slot 
      unsigned long long desired = contact_mgr->pack_entry(key, idx);

      // Try to install (key,index) atomically
      unsigned long long old = atomicCAS(&entry->packed, 
                                         ContactManager::EMPTY_PACKED, 
                                         desired);

      if (old == ContactManager::EMPTY_PACKED) {
        // Success!
        return;
      } else {
        // Failed - check if it's our key
        uint64_t old_key = contact_mgr->unpack_key(old);
        if (old_key == key) {
          // Another thread inserted the same contact
          return;
        }
        // Otherwise continue probing
      }
    }
  }
  printf("In GPU Rebuild: ContactManager hash table full or too many collisions!\n");
  return;
}

// A new way of using a neighbour list
template <class NeighbourContainer>
__global__ void processNeighboursContacts(
    NeighbourContainer ni, // Provides iterator with neighbours of a particle
    real time, // Current simulation time
    int Nwrite,
    int numberParticles,
    Box box,
    real *radius, // Radii in group indexing
    real3 *vel, // Velocities in group indexing
    real4 *ang_vel, // Angular velocities in group indexing
    real4 *force, // Forces in group indexing
    real4 *torque, // Torques in group indexing
    real *energy, // Energies in group indexing
    real *virial,  // Virial in group indexing
    real3 *stress_x, // Stress components in group indexing
    real3 *stress_y,
    real3 *stress_z,
    real dt, // Time step
    real kn, // Normal spring constant for overlap force
    real kt, // Tangential spring constant for overlap force
    real mu, // Coefficient of friction
    real gamma_n, // Damping coefficient for normal direction
    real gamma_t, // Damping coefficient for tangential direction
    ContactManager *contact_mgr // Contact history manager
  ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numberParticles)
    return;
  // Set ni to provide iterators for particle i
  ni.set(i);
  const real3 pi =
      make_real3(cub::ThreadLoad<cub::LOAD_LDG>(ni.getSortedPositions() + i));
  real3 f = real3();
  real3 t = real3();
  // real3 sx = real3();
  // real3 sy = real3();
  // real3 sz = real3();
  real e = 0;
  real v = 0; 
  // for(auto neigh: ni){ //This is equivalent to the while loop, although a tad
  // slower
  auto it = ni.begin(); // Iterator to the first neighbour of particle i
  // Note that ni.end() is not a pointer to the last neighbour, it just
  // represents "no more neighbours" and
  //  should not be dereferenced

  const int gli = ni.getGroupIndexes()[i];
  const real3 vi = vel[gli];
  const real3 avi = make_real3(ang_vel[gli]);
  const real ri = radius[gli];
  int nneigh;
  // bool writeStep = ( ((time+1) % Nwrite) == 0 );
  while (it) { // it will cast to false when there are no more neighbours
    auto neigh = *it++; // The iterator can only be advanced and dereferenced
    nneigh = 0;
    int j = neigh.getGroupIndex();
    const real3 pj = make_real3(neigh.getPos());
    const real3 vj = vel[j];
    if (j == gli)
      continue; // Skip self-interaction
    // const int glj = neigh.getGroupIndexes()[j];
    const real3 vij = vi - vj; // TODO: Account for PBCs in homogeneous shear flow using Lees-Edwards
    const real3 rij = box.apply_pbc(pj - pi);
    const real r2 = dot(rij, rij);
    const real3 avij = ri * avi + radius[j] * make_real3(ang_vel[j]);
    const real radii_sum = ri + radius[j];
    // printf("Processing contact between %d and %d: rij = %f, %f, %f, r2 = %f, radii_sum = %f\n", gli, j, rij.x, rij.y, rij.z, r2, radii_sum);
    const real radii_sum2 = radii_sum * radii_sum;
    if (r2 > 0 and r2 < (real(6.25))) {
      
      // TODO: reset the contact history if contact breaks using contact age
      // TODO: Account for different particle sizes here 
      // r2 < (Ri + Rj)^2
      // For now assume all particles have diameter 2 -> (1+1)^2 = 4

      if (r2 < radii_sum2) {
        nneigh++;
        // printf("Time: %f In block %d, thread %d: Particle %d interacting with %d\n", time, blockIdx.x, threadIdx.x, gli, j);
        // Particles are in contact - get or create contact history
        // printf("Contact age before update: %d\n", contact->contact_age);
        // Check for existing contact history and 
        if (gli < j){
          ContactHistory *contact = contact_mgr->getContact_v1(gli, j);
          // printf("Retrieved contact for (%d, %d): age=%f at time %f \n", gli, j, contact->contact_age, time);
          // ContactHistory *contact = contact_mgr->getContact(gli, j);
          if(contact->contact_time > 0){
            // printf("Existing contact between %d and %d found with age %d\n", contact->particle_i, contact->particle_j, *(contact->contact_age));

            // If the contact was last active before the previous timestep, reset the history
            if(contact->contact_age < (time-dt)){
              // printf("Contact between %d and %d was last active at time %f at present time %f, resetting history\n", contact->particle_i, contact->particle_j, contact->contact_age, time);
              // printf("Resetting contact history between %d and %d\n", contact->particle_i, contact->particle_j);
              contact->xi = make_real3(0,0,0);
              contact->contact_time = real(0.0);
            }
            // else{
            //   // Continuing contact - increment time and age

            //   // printf("Continuing contact between %d and %d\n", contact->particle_i, contact->particle_j);
            //   contact->contact_time += 0.5*dt; // +1/2 for each particle in the pair during counting
            //   // printf("Contact time between %d and %d is now %f\n", contact->particle_i, contact->particle_j, contact->contact_time);
            //   contact->xi += (gli > j ? real(1) : real(-1) ) * 0.5 * (vij - normal_component(vij, rij)) * dt; 
            //   printf("Updated xi between %d and %d is %f, %f, %f\n", contact->particle_i, contact->particle_j, contact->xi.x, contact->xi.y, contact->xi.z);
            // }
          }
          // Otherwise, this is a new contact or continuing contact - update age, time, and tangential displacement
          // Record the current time as the last active time 
          contact->contact_age = time;
          // printf("Continuing contact between %d and %d\n", contact->particle_i, contact->particle_j);
          contact->contact_time += dt; // +1/2 for each particle in the pair during counting
          // printf("Contact time between %d and %d is now %f \n", contact->particle_i, contact->particle_j, contact->contact_time);
          contact->xi += tangential_component(vij, rij) * dt;
          // contact->xi += (gli > j ? real(1) : real(-1) ) * ( (vij - normal_component(vij, rij)) + cross_product_2D( avij, rij*rsqrt(dot(rij,rij)) ) ) * dt; 
          // contact->xi += (gli > j ? real(1) : real(-1) ) * 0.5 * ( (vij - normal_component(vij, rij)) ) * dt; 
          // printf("Updated xi between %d and %d is %10f, %10f, %10f\n", contact->particle_i, contact->particle_j, contact->xi.x, contact->xi.y, contact->xi.z);
    
          // printf("Contact between %d and %d active at time %f\n", contact->particle_i, contact->particle_j, time);
          // printf("contact between %d and %d has xi %f, %f, %f\n", contact->particle_i, contact->particle_j, contact->xi.x, contact->xi.y, contact->xi.z);
          // printf("Contact between %d and %d has relative velocity %f, %f, %f\n", contact->particle_i, contact->particle_j, vij.x, vij.y, vij.z);
          const real3 fmod = (force or virial) ? total_contact_force_ij(kn, kt, gamma_n, gamma_t, mu, dt, rij, vij, contact, radii_sum) : real3();
          const real3 tmod = (torque) ? ri * static_torque_ij_2D(fmod, rij) : real3();

          // const real3 foverlap = overlap_force_ij(kn, rij, radii_sum);

          // real3 fel = real3();
          // real3 fdiss = real3();
          // compute_contact_forces(kn, kt, gamma_n, gamma_t, mu, rij, vij, contact, radii_sum, fel, fdiss);
          // real3 sx = (stress_x) ? compute_stress_ij(make_real3(rij.x), fel): real3();
          // sx += (stress_x) ? compute_stress_ij(make_real3(rij.x), fdiss, true): real3();

          // real3 sy = (stress_y) ? compute_stress_ij(make_real3(rij.y), fel): real3();
          // sy += (stress_y) ? compute_stress_ij(make_real3(rij.y), fdiss, true): real3();

          // real3 sz = (stress_z) ? compute_stress_ij(make_real3(rij.z), fel): real3();
          // sz += (stress_z) ? compute_stress_ij(make_real3(rij.z), fdiss, true): real3();  

          const real3 sx = (stress_x) ? 0.5 * compute_stress_ij(rij, fmod, 0): real3();
          const real3 sy = (stress_y) ? 0.5 * compute_stress_ij(rij, fmod, 1): real3();
          const real3 sz = (stress_z) ? 0.5 * compute_stress_ij(rij, fmod, 2): real3();

          // const real3 sy = (stress_y) ? compute_stress_ij(make_real3(rij.y), fmod): real3();
          // const real3 sz = (stress_z) ? compute_stress_ij(make_real3(rij.z), fmod): real3();

          // const real3 sx = (stress_x) ? compute_stress_i(make_real3(rij.x), foverlap): real3();
          // const real3 sy = (stress_y) ? compute_stress_i(make_real3(rij.y), foverlap): real3();
          // const real3 sz = (stress_z) ? compute_stress_i(make_real3(rij.z), foverlap): real3();

          // printf("Components of sx for particles %d and %d: %f, %f, %f\n", gli, j, sx.x, sx.y, sx.z);
          // printf("Components of sy for particles %d and %d: %f, %f, %f\n", gli, j, sy.x, sy.y, sy.z);
          
          // printf("Force between %d and %d is %f, %f, %f\n\n", contact->particle_i, contact->particle_j, fmod.x, fmod.y, fmod.z);
          if (force){
            // f += fmod;
            // t += tmod;
            atomicAdd(&force[gli].x, fmod.x);
            atomicAdd(&force[gli].y, fmod.y);
            atomicAdd(&force[gli].z, fmod.z);

            atomicAdd(&force[j].x, -fmod.x);
            atomicAdd(&force[j].y, -fmod.y);
            atomicAdd(&force[j].z, -fmod.z);

            
          }
          if (torque){
            // printf("Torque between %d and %d is %f, %f, %f\n\n", contact->particle_i, contact->particle_j, tmod.x, tmod.y, tmod.z);
            t += tmod;
          } 
          if (stress_x){
            // sx += compute_stress_i(make_real3(rij.x), fmod);
            // sx = compute_stress_i(make_real3(rij.x), fmod);
            // printf("Components of sx for particles %d and %d: %f, %f, %f\n", gli, j, sx.x, sx.y, sx.z);
            atomicAdd(&stress_x[gli].x, sx.x); // split equally between the two particles in the pair
            atomicAdd(&stress_x[gli].y, sx.y);
            atomicAdd(&stress_x[gli].z, sx.z);

            atomicAdd(&stress_x[j].x, sx.x);
            atomicAdd(&stress_x[j].y, sx.y);
            atomicAdd(&stress_x[j].z, sx.z);
          }
          if (stress_y){
            // sy += compute_stress_i(make_real3(rij.y), fmod);
            // sy = compute_stress_i(make_real3(rij.y), fmod);
            // printf("Components of sy for particles %d and %d: %f, %f, %f\n", gli, j, sy.x, sy.y, sy.z);
            atomicAdd(&stress_y[gli].x, sy.x);
            atomicAdd(&stress_y[gli].y, sy.y);
            atomicAdd(&stress_y[gli].z, sy.z);

            atomicAdd(&stress_y[j].x, sy.x);
            atomicAdd(&stress_y[j].y, sy.y);
            atomicAdd(&stress_y[j].z, sy.z);
          }
          if (stress_z){
            // sz += compute_stress_i(make_real3(rij.z), fmod);
            // sz = compute_stress_i(make_real3(rij.z), fmod);
            // printf("Components of sz for particles %d and %d: %f, %f, %f\n", gli, j, sz.x, sz.y, sz.z);
            atomicAdd(&stress_z[gli].x, sz.x);
            atomicAdd(&stress_z[gli].y, sz.y);
            atomicAdd(&stress_z[gli].z, sz.z);

            atomicAdd(&stress_z[j].x, sz.x);
            atomicAdd(&stress_z[j].y, sz.y);
            atomicAdd(&stress_z[j].z, sz.z);

          }
          // if (energy)
          //   e += lj_energy(r2);
          // if (virial)
          //   v += dot(fmod, rij);

        } // gli < j check
      } // overlap check 
    } // r2 vicinity check
  } // while neighbours  

  if (force){
    // force[gli] += make_real4(f);
    // torque[gli] += make_real4(t);
    // printf("Total force on particle %d: %f, %f, %f\n", gli, f.x, f.y, f.z);
    // printf("Total torque on particle %d: %f, %f, %f\n", gli, t.x, t.y, t.z);
  }
  if (stress_x){
    // stress_x[gli] += (nneigh > 0) ? sx / nneigh : real3();
    // stress_x[gli] /=  (nneigh > 0) ? nneigh : real(1.0);
  }
  if (stress_y){
    // stress_y[gli] += (nneigh > 0) ? sy / nneigh : real3();
    // stress_y[gli] /= (nneigh > 0) ? nneigh : real(1.0);
  }
  if (stress_z){
    // stress_z[gli] += (nneigh > 0) ? sz / nneigh : real3();
    // stress_z[gli] /= (nneigh > 0) ? nneigh : real(1.0);
  }
  if (torque){
    torque[gli] += make_real4(t);
  }
  if (energy){
    energy[gli] += e;
  }
  if (virial){
    virial[gli] += v;
  }
}

// CustomContactInteractor::~CustomContactInteractor(){
//     if (d_contact_mgr) {
//             cudaFree(d_contact_mgr);
//             d_contact_mgr = nullptr;
//         }
// }

// void CustomContactInteractor::updateSimulationTime(real newTime) { time = newTime; }

// void CustomContactInteractor::h_cleanupContacts(cudaStream_t st){
//     // std::cout << "Contact cleanup check at time " << time << std::endl;
//     // Periodically clean up inactive contacts 
//     ++steps; 
//     if ((steps % 10000) == 0) {
//         std::cout << "DEBUG step=" << steps
//                 << " time/dt=" << (time/dt)
//                 << " time=" << time << "\n";
//     }
//     if ( (steps % cleanup_interval) == 0){
//         cudaStreamSynchronize(st);
//         auto policy = thrust::cuda::par.on(st);

//         thrust::device_ptr<ContactHistory> d_begin(contact_mgr->contacts);
//         thrust::device_ptr<ContactHistory> d_end = d_begin + contact_mgr->max_contacts; 
//         int h_count = thrust::count_if(
//         policy,
//         d_begin,
//         d_end, 
//         [] __device__ (const ContactHistory& contact) { return contact.is_active; }
//         );

//         // int h_count = 0; 
//         // cudaMemcpy(&h_count, contact_mgr->contact_count, sizeof(int), cudaMemcpyDeviceToHost);
//         std::cout << "Starting contact cleanup at time " << time << " with " << h_count << " active contacts." << std::endl;
//         // fprintf(stdout, "Cleaning up inactive contacts at time %f (step %d)\n", time, steps);
//         int threads = 128;
//         int blocks = (contact_mgr->max_contacts + threads - 1) / threads;
//         gpu_cleanupContacts<<<blocks, threads, 0, st>>>(d_contact_mgr, contact_mgr->hash_table, time-2*dt);
//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
//         }
//         err = cudaDeviceSynchronize();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
//             // Optional: exit or raise error so program stops near the failing kernel
//             exit(1);
//         }
//         // Print number of cleaned contacts
//         // cudaMemcpy(&h_count, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
//         // std::cout << "Cleaned up " << h_count << " inactive contacts at time " << time << std::endl;

//         int cleaned;
//         cudaMemcpy(&cleaned, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
//         std::cout << "Marked " << cleaned << " contacts as inactive" << std::endl;

//         int total_active = count_if(
//         policy,
//         d_begin,
//         d_end, 
//         [] __device__ (const ContactHistory& contact) { return contact.is_active; }
//         );
//         std::cout << "Total active contacts before compaction: " << total_active << std::endl;

//         // Reset cleanup count
//         // int zero = 0;
//         // cudaMemcpy(contact_mgr->cleanup_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

//         // Copy contact data back to host for compaction
//         // cudaMemcpy(contact_mgr->contacts, d_contact_mgr->contacts, 
//         //            contact_mgr->max_contacts * sizeof(ContactHistory), 
//         //            cudaMemcpyDeviceToHost);
//         // cudaMemcpy(contact_mgr->contact_count, d_contact_mgr->contact_count, 
//         //            sizeof(int), 
//         //            cudaMemcpyDeviceToHost);

//         std::cout << "Compacting contacts array after cleanup..." << std::endl;
//         // thrust::device_ptr<ContactHistory> d_begin(contact_mgr->contacts);
//         // thrust::device_ptr<ContactHistory> d_end = d_begin + contact_mgr->max_contacts;  
//         // Compact the contacts array to remove inactive contacts
//         // auto new_end = thrust::remove_if(
//         //   thrust::device,
//         //   contact_mgr->contacts,
//         //   contact_mgr->contacts + cleaned,
//         //   [] __device__ (const ContactHistory& contact) { return !contact.is_active; }
//         // );

//         auto new_end = thrust::partition(
//         // thrust::device,
//         policy,
//         d_begin,
//         d_end,
//         [] __device__ (const ContactHistory& contact) { return contact.is_active; }
//         );
//         std::cout << "Compaction complete." << std::endl;
//         // Obtain new contact count by summing over active contacts
//         int new_count = thrust::distance(d_begin, new_end);
//         // int new_count = thrust::count_if(
//         //   d_begin,
//         //   new_end, 
//         //   [] __device__ (const ContactHistory& contact) { return contact.is_active; }
//         // );
//         std::cout << "After compaction: " << new_count << " active contacts (removed " 
//                 << (h_count - new_count) << ")" << std::endl;

//         // int new_count = thrust::count_if(
//         //     thrust::device, 
//         //     contact_mgr->contacts, 
//         //     contact_mgr->contacts + ((contact_mgr->max_contacts)), 
//         //     [] __device__ (const ContactHistory& contact) { return contact.is_active; }
//         //   );
//         // cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
//         // std::cout << "Updated contact count after compaction: " << new_count << std::endl;

//         // Step 4: Update contact counts
//         cudaMemset(contact_mgr->contact_count, 0, sizeof(int));
//         cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), 
//         cudaMemcpyHostToDevice);

//         // Reset cleanup counter
//         cudaMemset(contact_mgr->cleanup_count, 0, sizeof(int));

//         // Copy updated contact data back to device
//         // cudaMemcpy(d_contact_mgr->contacts, contact_mgr->contacts, 
//         //            contact_mgr->max_contacts * sizeof(ContactHistory), 
//         //            cudaMemcpyHostToDevice);
//         // cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
        
//         // Reset the hash table 
//         contact_mgr->resetHashTable();
//         blocks = (new_count + threads - 1) / threads;
//         std::cout << "Rebuilding hash table for " << new_count << " contacts..." << std::endl;
        
//         gpu_rebuildHashTable<<<blocks, threads, 0, st>>>(d_contact_mgr, contact_mgr->hash_table);
        
//         err = cudaDeviceSynchronize();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
//             // Optional: exit or raise error so program stops near the failing kernel
//             exit(1);
//         }

//         err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "Rebuild kernel failed: %s\n", cudaGetErrorString(err));
//             exit(1);
//         }
//         // cudaStreamSynchronize(st);

//         std::cout << "Cleanup complete.\n" << std::endl;
//         // std::cout << "Rebuilding hash table after cleanup..." << std::endl;
//         // h_rebuildHashTable(st);

//     }
//     return;
// }

// void CustomContactInteractor::sum(Computables comp, cudaStream_t st) {
//     // std::cout << "=========" << " CustomContactInteractor at time " << time << " " <<"=========" << std::endl;
//     Box box(boxSize);
//     nl->update(box, rcut, st);
//     comp.torque = false; // Ensure torque is computed
//     comp.stress = true; // Ensure stress is computed

//     // DON'T mark contacts inactive - we want to preserve history!
//     // Instead, the kernel will mark contacts as active when found
    
//     // NeighbourContainer can provide forward iterators with the neighbours of
//     // each particle The drawback of it being a forward iterator is that it can
//     // only be advanced, once you have asked for the next neighbour there is no
//     // going back without starting from the first. With it=ni.begin() you can
//     // only do it++, etc, there is no operator[] nor it--
//     auto ni = nl->getNeighbourContainer();
//     auto vel = pd->getHalfVel(access::gpu, access::read).raw();
//     auto ang_vel = pd->getHalfAngVel(access::gpu, access::read).raw();
//     auto radius = pd->getRadius(access::gpu, access::read).raw();
//     auto force =
//         comp.force
//             ? pd->getForce(access::location::gpu, access::mode::readwrite).raw()
//             : nullptr;

//     auto torque = 
//         comp.torque
//             ? pd->getTorque(access::location::gpu, access::mode::readwrite).raw()
//             : nullptr;

//     auto stress_x = 
//         comp.stress
//             ? pd->getStressX(access::location::gpu, access::mode::readwrite).raw()
//             : nullptr;

//     auto stress_y =
//         comp.stress
//             ? pd->getStressY(access::location::gpu, access::mode::readwrite).raw()
//             : nullptr;

//     auto stress_z =
//         comp.stress
//             ? pd->getStressZ(access::location::gpu, access::mode::readwrite).raw()
//             : nullptr;

//     auto energy = comp.energy ? pd->getEnergy(access::location::gpu,
//                                               access::mode::readwrite)
//                                     .raw()
//                               : nullptr;
//     auto virial = comp.virial ? pd->getVirial(access::location::gpu,
//                                               access::mode::readwrite)
//                                     .raw()
//                               : nullptr;
//     processNeighboursContacts<decltype(ni)><<<numberParticles / 128 + 1, 128, 0, st>>>(
//         ni, time, Nwrite, numberParticles, box, radius, vel, ang_vel, force, torque, energy, virial, stress_x, stress_y, stress_z,
//         dt, kn, kt, mu, gamma_n, gamma_t, d_contact_mgr);
    
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
//     }
//     err = cudaDeviceSynchronize();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
//         // Optional: exit or raise error so program stops near the failing kernel
//         exit(1);
//     }

//     h_cleanupContacts(st);

// };

