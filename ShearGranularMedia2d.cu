// UAMMD Brownian Dynamics simulation of 3D ideal particle diffusion 
/* Raul P. Pelaez 2021
   Moving particles with Integrators.
   In this tutorial we will learn about another basic UAMMD module, Integrator.
   Integrator is a small interface that allows to encode the concept of evolving
   particles due to some dynamics. We will see how to create and use a Brownian
   Dynamics (BD) Integrator which will allow us to simulate ideal (non
   interacting) particles. After this tutorial, you will have almost every tool
   you need to construct simulations, the only thing left is how to add
   interactions. Which we will cover in the next tutorial.

 */

#define EXTRA_COMPUTABLES (torque)
// #define EXTRA_COMPUTABLES (stress)
#include <uammd.cuh>
#include "Integrator/VelocityVerlet.cuh" //Each Integrator has a particular include
// #include "Forces/PairForces.cuh"
// #include "ComputeForces.cuh"
// #include "PairForces.cuh"
#include <Interactor/ExternalForces.cuh>
#include <Interactor/Interactor.cuh>
#include <Interactor/NeighbourList/CellList.cuh>
#include <ParticleData/ParticleGroup.cuh>
#include <iterator>
#include <fstream>
#include <random>
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
// #include <thrust/transform_reduce.h>
#include <utils/vector.cuh>
#include <utils/InitialConditions.cuh>

#include "ExternalForcesStructs.cuh"
#include "Contacts.cuh"
// #include "GPU_contacts.cuh"
#include "SimUtils.cuh"
using namespace uammd;

struct ExternalForceField: public ParameterUpdatable {
    real3 fext; // external force magnitude
    real time;

    ExternalForceField(real3 fext=make_real3(1.0, 0.0, 0.0)):
        fext(fext), time(0) { }

    __device__ ForceEnergyVirial sum(Interactor::Computables comp){
        // real3 f = {fext, 0.0, 0.0};
        real energy = comp.energy ? 0: 0; 
        real virial = comp.virial ? 0: 0; 

        return {fext, energy, virial};
    }

    auto getArrays(ParticleData *pd) {
        return std::make_tuple();
    }

    virtual void updateSimulationTime(real newTime) override { time = newTime; }

};

struct MovingHarmonicField: public ParameterUpdatable {
    real k; // spring constant
    // real3 center; // center of the harmonic potential as a function of time
    real time;
    bool isPeriodic = true; // If true, apply minimum image convention
    real3 boxSize;
    real3 minusInvBoxSize;
    real moving; // If non-zero, the centers of the wells moves in time along x

    // TODO: parallelize this function
    inline void __host__ getInitCenter(std::shared_ptr<ParticleData> pd) const{
        // Get the particle initial position to set the centers of the wells
        if( time==0){
          int numParticles = pd->getNumParticles();
          auto pos = pd->getPos(access::cpu, access::read);
          auto initCenter = pd->getInitCenter(access::cpu, access::readwrite);
          for( int ii = 0; ii < numParticles; ii++ ){
            initCenter[ii] = {pos[ii].x, pos[ii].y, pos[ii].z};
          }
          return;
        }
    }

    // MovingHarmonicField(Parameters par):
    MovingHarmonicField(UAMMD sim, real moving_=1.0):
        k(sim.par.k), time(0), boxSize(sim.par.boxSize), minusInvBoxSize(-1.0/boxSize), moving(moving_) { 
          getInitCenter(sim.pd);
        }

    inline __host__ __device__ real3 applyPBC(real3 r){
        real3 offset = floorf(r * minusInvBoxSize + real(0.5)); // MIC Algorithm
        r.x += offset.x * boxSize.x;
        r.y += offset.y * boxSize.y;
        r.z += offset.z * boxSize.z;
        return r;
    }

    __device__ ForceEnergyVirial sum(Interactor::Computables comp, 
                                    const real4 &pos, const real3 &initCenter){
        
        // TODO: Allow for moving center in any specified trajectory by user
        // center = {2*time, 0, 0}; 
        real3 center = {initCenter.x + moving*time, initCenter.y, initCenter.z};
        // real3 center = {initCenter.x, initCenter.y + real(0.1)*moving*time, initCenter.z + real(0.25)*moving*time};
        // real3 center = {initCenter.x, initCenter.y, initCenter.z + 0.25*moving*time};
        real3 disp;
        //TODO: Generalize for non-cubic boxes
        //TODO: Generalize for non-periodic directions
        if( isPeriodic ){
            real3 offset = applyPBC(center);
            center.x = offset.x;
            center.y = offset.y;
            center.z = offset.z;

            // Apply MIC on displacement calculation between center well and particle 
            disp.x = center.x - pos.x;
            disp.y = center.y - pos.y;
            disp.z = center.z - pos.z;

            offset = applyPBC(disp);
            disp.x = offset.x;
            disp.y = offset.y;
            disp.z = offset.z;
        }
        else{
            disp.x = center.x - pos.x;
            disp.y = center.y - pos.y;
            disp.z = center.z - pos.z;
        }

        real3 f = {k*disp.x,
                   k*disp.y,
                   k*disp.z};
        real energy = comp.energy ? 0: 0; 
        real virial = comp.virial ? 0: 0; 

        return {f, energy, virial};
    }

    auto getArrays(ParticleData *pd) {
        auto pos = pd->getPos(access::gpu, access::read);
        auto initCenter = pd->getInitCenter(access::gpu, access::read);
        return std::make_tuple(pos.begin(), initCenter.begin());
    }

    virtual void updateSimulationTime(real newTime) override { time = newTime; }

};

// This function will print particle positions and velocities to a file called
// particles.dat
void writeSimulation(UAMMD sim) {
  // Lets store a file stream statically
  static std::ofstream out("particles3.dat");
  static std::ofstream out_stress("stress.dat");
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  auto vel = sim.pd->getVel(access::cpu, access::read);
  auto stress_x = sim.pd->getStressX(access::cpu, access::read);
  auto stress_y = sim.pd->getStressY(access::cpu, access::read);
  auto image = sim.pd->getImage(access::cpu, access::read);
  // A permutation iterator takes an iterator and an index iterator and the
  // indirection when accessed
  auto pos_by_id = thrust::make_permutation_iterator(
      pos.begin(),
      id2index); // pos_by_id[i] is now equivalent to pos[id2index[i]]
  auto vel_by_id = thrust::make_permutation_iterator(vel.begin(), id2index);
  auto image_by_id = thrust::make_permutation_iterator(image.begin(), id2index);
  for (int i = 0; i < sim.par.numberParticles; i++){
    out << pos_by_id[i] << " " << vel_by_id[i] << " " << image_by_id[i] << std::endl;
    out_stress << stress_x[i] << " " << stress_y[i] << std::endl;
  }

}

// Creates and returns a UAMMD struct
UAMMD initializeUAMMD(int argc, char *argv[]) {
  UAMMD sim;
  sim.par = Parameters();
  // Initialize ParticleData
  sim.pd = std::make_shared<ParticleData>(sim.par.numberParticles);
  return sim;
}


// Initialize positions from a file
void initializePositionsFromFile(UAMMD sim, std::string filename) {
  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  std::vector<real4> positions;
  real x, y, z;
  while (infile >> x >> y >> z) {
    positions.push_back(make_real4(x, y, z, 0.0));
  }
  infile.close();
  if (positions.size() != sim.par.numberParticles) {
    std::cerr << "Error: number of positions in file does not match numberParticles"
              << std::endl;
    return;
  }
  auto pos = sim.pd->getPos(access::cpu, access::write);
  std::copy(positions.begin(), positions.end(), pos.begin());

  // Print the first line of the file to verify
  std::cout << "First line of pos " << positions[0] << std::endl;
}

void initializeRadii(UAMMD sim){
  auto radii = sim.pd->getRadius(access::cpu, access::write);
  // set to default value
  // Read from file 
  std::ifstream infile("input/radii.dat");
  if (!infile)
  {
    std::cerr << "Error opening file: radii.dat" << std::endl;
    return;
  }
  for (int ii = 0; ii < sim.par.numberParticles; ii++)
  {
    infile >> radii[ii];
  }
  infile.close();

  // for(int ii=0; ii < sim.par.numberParticles; ii++){
  //   radii[ii] = sim.par.radius;
  // }
}

void initializeVelocities(UAMMD sim){
  auto vel = sim.pd->getVel(access::cpu, access::write); 
  auto half_vel = sim.pd->getHalfVel(access::cpu, access::write);
  auto ang_vel = sim.pd->getAngVel(access::cpu, access::write);
  auto half_ang_vel = sim.pd->getHalfAngVel(access::cpu, access::write);
  // set to zero
  for(int ii=0; ii < sim.par.numberParticles; ii++){
    vel[ii] = make_real3(0.0); 
    half_vel[ii] = make_real3(0.0);
    ang_vel[ii] = make_real4(0.0);
    half_ang_vel[ii] = make_real4(0.0);
  }
}

// void initializeOldForces(UAMMD sim){
//   auto force = sim.pd->getOldForce(access::cpu, access::write); 
//   // set to zero
//   for(int ii=0; ii < sim.par.numberParticles; ii++){
//     force[ii] = make_real4(0.0, 0.0, 0.0, 0.0); 
//   }
// }

void initializeMasses(UAMMD sim){
  auto mass = sim.pd->getMass(access::cpu, access::write);
  for(int ii=0; ii < sim.par.numberParticles; ii++){
    mass[ii] = sim.par.mass; 
  }
}
// This function constructs and returns a Newtonian Dynamics integrator.
std::shared_ptr<Integrator> createNewtonianDynamicsIntegrator(UAMMD sim) {
  using ND = ND::NewtonEuler;
  ND::Parameters par;
  par.is2D = sim.par.is2D;
  par.dt = sim.par.dt;
  par.temperature = sim.par.temperature;
  par.hydrodynamicRadius = sim.par.hydrodynamicRadius;
  par.viscosity = sim.par.viscosity;
  par.box = Box(sim.par.boxSize); 
  par.box.setPeriodicity(true, true, true); // Periodic boundaries
  // par.K[0] = sim.par.Kx;
  return std::make_shared<ND>(sim.pd, par);
}

// Prints the positions of particle with names from 0 to 9 using what we saw in
// the previous tutorial
void printFirst10Particles(UAMMD sim) {
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  // A permutation iterator takes an iterator and an index iterator and the
  // indirection when accessed
  auto pos_by_id = thrust::make_permutation_iterator(
      pos.begin(),
      id2index); // pos_by_id[i] is now equivalent to pos[id2index[i]]
  std::cout << "Particles with names from 0 to 9:" << std::endl;
  std::cout << "Name \t position" << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << i << "\t" << pos_by_id[i] << std::endl;
}

// This function fills a vector with the particle positions ordered by id (name)
// and returns it
auto vector_from_pd_positions(UAMMD sim) {
  auto id2index = sim.pd->getIdOrderedIndices(access::cpu);
  auto pos = sim.pd->getPos(access::cpu, access::read);
  auto pos_by_id = thrust::make_permutation_iterator(pos.begin(), id2index);
  return std::vector<real4>(pos_by_id, pos_by_id + pos.size());
}

auto createExternalForceInteractor(UAMMD sim, std::shared_ptr<ParticleGroup> pg, real3 fext=make_real3(1.0, 0.0, 0.0)) {
  auto forceField = std::make_shared<ExternalForceField>(fext);
  auto ext = std::make_shared<ExternalForces<ExternalForceField>>(pg, forceField);
  return ext;
}

auto createExternalPotentialInteractor(UAMMD sim, std::shared_ptr<ParticleGroup> pg, real moving=1.0) {
  auto well = std::make_shared<MovingHarmonicField>(sim, moving);
  auto ext = std::make_shared<ExternalForces<MovingHarmonicField>>(pg, well);
  return ext;
}

__device__ void compute_contact_forces(
    real kn, real kt, 
    real gamma_n, 
    real gamma_t,
    real mu, real3 rij, real3 vij, 
    ContactHistory* contact, real radius_sum, real3 &fn, real3 &fdiss){
    // Modifies fn and fdiss by reference
      if(( contact != nullptr) && ( contact->is_active)){
        fn = overlap_force_ij(kn, rij, radius_sum);
        fdiss = -damping_force_ij(gamma_n, vij, rij, true);
        fdiss += -damping_force_ij(gamma_t, vij, rij, false);
        fdiss += tangential_frictional_force_ij(kt, contact->xi, mu, fn);
      }
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

          const real3 sx = (stress_x) ? 0.5 * compute_stress_ij(rij, fmod, 0): real3();
          const real3 sy = (stress_y) ? 0.5 * compute_stress_ij(rij, fmod, 1): real3();
          const real3 sz = (stress_z) ? 0.5 * compute_stress_ij(rij, fmod, 2): real3();

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
  if (torque)
    torque[gli] += make_real4(t);
  if (energy)
    energy[gli] += e;
  if (virial)
    virial[gli] += v;
}

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


// Define a custom interactor to compute the normal and tangential
// frictional forces at contact 
// Needs to be aware of the simulation time to update the tangential
class CustomContactInteractor : public ParameterUpdatable, public Interactor {
  using NeighbourList = CellList;
  std::shared_ptr<NeighbourList> nl;
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<ContactManager> contact_mgr;  // Contact history manager
  ContactManager *d_contact_mgr = nullptr; // Device pointer to contact manager
  real dt;
  real time = 0;
  uint64_t steps = 0;
  int Nwrite;
  real rcut; 
  real3 boxSize;
  int numberParticles; 
  real kn; // Normal spring constant for overlap force
  real kt; // Tangential spring constant for overlap force
  real mu; // Coefficient of friction
  real gamma_n; // Damping coefficient for normal direction
  real gamma_t; // Damping coefficient for tangential direction
  uint64_t cleanup_interval; // Interval for cleaning inactive contacts

public: 
  CustomContactInteractor( UAMMD sim ) : 
    Interactor(sim.pd, "Custom"),
    pd(sim.pd),
    rcut(sim.par.rcut), 
    boxSize(sim.par.boxSize), 
    numberParticles(sim.par.numberParticles), 
    dt(sim.par.dt), 
    time(0),
    Nwrite(sim.par.Nwrite),
    kn(sim.par.kn), 
    kt(sim.par.kt), 
    mu(sim.par.mu), 
    gamma_n(sim.par.gamma_n),
    gamma_t(sim.par.gamma_t)
     {
      cleanup_interval = uint64_t(1.0 / dt); // Clean up every 1.0 time units
      // cleanup_interval = 1250;
      nl = std::make_shared<CellList>(sim.pd);
      // Initialize contact manager with estimated max contacts
      // int max_contacts = numberParticles * 1000;  // Estimate 10 contacts per particle
      contact_mgr = std::make_shared<ContactManager>(numberParticles);
      // create device copy of contact manager
      cudaError_t err = cudaMalloc((void**)&d_contact_mgr, sizeof(ContactManager));
      if (err != cudaSuccess) {
          throw std::runtime_error("Failed to allocate device memory for ContactManager");
      }
      err = cudaMemcpy(d_contact_mgr, contact_mgr.get(), sizeof(ContactManager), cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          cudaFree(d_contact_mgr);
          throw std::runtime_error("Failed to copy ContactManager to device");
      }

  }

  ~CustomContactInteractor() override {
    // free device-side struct; the host contact_mgr destructor will free its internal data
    if (d_contact_mgr) {
        cudaFree(d_contact_mgr);
        d_contact_mgr = nullptr;
    }

  }

  virtual void updateSimulationTime(real newTime) override { time = newTime; }

  void h_rebuildHashTable(cudaStream_t st) {
    // Rebuild the hash table on the device
    contact_mgr->resetHashTable(); 

    cudaError_t err;
    // err = cudaMemcpy(d_contact_mgr, contact_mgr.get(), sizeof(ContactManager), cudaMemcpyHostToDevice);
    // if (err != cudaSuccess) {
    //     throw std::runtime_error("Failed to copy ContactManager to device during hash table rebuild");
    // }

    gpu_rebuildHashTable<<<contact_mgr->max_contacts / 128 + 1, 128, 0, st>>>(
      d_contact_mgr,
      contact_mgr->hash_table
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }

  void h_cleanupContacts(cudaStream_t st) {
    // std::cout << "Contact cleanup check at time " << time << std::endl;
    // Periodically clean up inactive contacts 
    ++steps; 
    if ((steps % 10000) == 0) {
      std::cout << "DEBUG step=" << steps
                << " time/dt=" << (time/dt)
                << " time=" << time << "\n";
    }
    if ( (steps % cleanup_interval) == 0){
      cudaStreamSynchronize(st);
      auto policy = thrust::cuda::par.on(st);

      thrust::device_ptr<ContactHistory> d_begin(contact_mgr->contacts);
      thrust::device_ptr<ContactHistory> d_end = d_begin + contact_mgr->max_contacts; 
      int h_count = thrust::count_if(
        policy,
        d_begin,
        d_end, 
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
      );

      // int h_count = 0; 
      // cudaMemcpy(&h_count, contact_mgr->contact_count, sizeof(int), cudaMemcpyDeviceToHost);
      std::cout << "Starting contact cleanup at time " << time << " with " << h_count << " active contacts." << std::endl;
      // fprintf(stdout, "Cleaning up inactive contacts at time %f (step %d)\n", time, steps);
      int threads = 128;
      int blocks = (contact_mgr->max_contacts + threads - 1) / threads;
      gpu_cleanupContacts<<<blocks, threads, 0, st>>>(d_contact_mgr, contact_mgr->hash_table, time-2*dt);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
      }
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
          fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
          // Optional: exit or raise error so program stops near the failing kernel
          exit(1);
      }
      // Print number of cleaned contacts
      // cudaMemcpy(&h_count, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
      // std::cout << "Cleaned up " << h_count << " inactive contacts at time " << time << std::endl;

      int cleaned;
      cudaMemcpy(&cleaned, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
      std::cout << "Marked " << cleaned << " contacts as inactive" << std::endl;

      int total_active = count_if(
        policy,
        d_begin,
        d_end, 
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
      );
      std::cout << "Total active contacts before compaction: " << total_active << std::endl;

      std::cout << "Compacting contacts array after cleanup..." << std::endl;

      auto new_end = thrust::partition(
        policy,
        d_begin,
        d_end,
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
      );
      std::cout << "Compaction complete." << std::endl;
      // Obtain new contact count by summing over active contacts
      int new_count = thrust::distance(d_begin, new_end);
      std::cout << "After compaction: " << new_count << " active contacts (removed " 
                << (h_count - new_count) << ")" << std::endl;

      // Step 4: Update contact counts
      cudaMemset(contact_mgr->contact_count, 0, sizeof(int));
      cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), 
      cudaMemcpyHostToDevice);

      // Reset cleanup counter
      cudaMemset(contact_mgr->cleanup_count, 0, sizeof(int));
      
      // Reset the hash table 
      contact_mgr->resetHashTable();
      blocks = (new_count + threads - 1) / threads;
      std::cout << "Rebuilding hash table for " << new_count << " contacts..." << std::endl;
      
      gpu_rebuildHashTable<<<blocks, threads, 0, st>>>(d_contact_mgr, contact_mgr->hash_table);
      
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
          fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
          // Optional: exit or raise error so program stops near the failing kernel
          exit(1);
      }

      err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "Rebuild kernel failed: %s\n", cudaGetErrorString(err));
          exit(1);
      }
      // cudaStreamSynchronize(st);
    
      std::cout << "Cleanup complete.\n" << std::endl;
      // std::cout << "Rebuilding hash table after cleanup..." << std::endl;
      // h_rebuildHashTable(st);

    }
    return;
  }

  void sum(Computables comp, cudaStream_t st) override {
    // std::cout << "=========" << " CustomContactInteractor at time " << time << " " <<"=========" << std::endl;
    Box box(boxSize);
    nl->update(box, rcut, st);
    comp.torque = false; // Ensure torque is computed
    comp.stress = true; // Ensure stress is computed

    // DON'T mark contacts inactive - we want to preserve history!
    // Instead, the kernel will mark contacts as active when found
    
    // NeighbourContainer can provide forward iterators with the neighbours of
    // each particle The drawback of it being a forward iterator is that it can
    // only be advanced, once you have asked for the next neighbour there is no
    // going back without starting from the first. With it=ni.begin() you can
    // only do it++, etc, there is no operator[] nor it--
    auto ni = nl->getNeighbourContainer();
    auto vel = pd->getHalfVel(access::gpu, access::read).raw();
    auto ang_vel = pd->getHalfAngVel(access::gpu, access::read).raw();
    auto radius = pd->getRadius(access::gpu, access::read).raw();
    auto force =
        comp.force
            ? pd->getForce(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto torque = 
        comp.torque
            ? pd->getTorque(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto stress_x = 
        comp.stress
            ? pd->getStressX(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto stress_y =
        comp.stress
            ? pd->getStressY(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto stress_z =
        comp.stress
            ? pd->getStressZ(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto energy = comp.energy ? pd->getEnergy(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    auto virial = comp.virial ? pd->getVirial(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    processNeighboursContacts<decltype(ni)><<<numberParticles / 128 + 1, 128, 0, st>>>(
        ni, time, Nwrite, numberParticles, box, radius, vel, ang_vel, force, torque, energy, virial, stress_x, stress_y, stress_z,
        dt, kn, kt, mu, gamma_n, gamma_t, d_contact_mgr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
        // Optional: exit or raise error so program stops near the failing kernel
        exit(1);
    }

    h_cleanupContacts(st);

  };
};

int main(int argc, char *argv[]) {
  auto sim = initializeUAMMD(argc, argv);
  initializePositionsFromFile(sim, "input/initial_positions.dat");
  // initializePositionsFromFile(sim, "plate_positions.dat");
  // initializePositionsFromFile(sim, "test_particle_position.dat");
  initializeVelocities(sim); 
  // initializeOldForces(sim);
  initializeMasses(sim);
  initializeRadii(sim);

  // Verify that the OldForces and Velocities are initialized to be zero
  // auto vel = sim.pd->getVel(access::cpu, access::read);
  // auto old_force = sim.pd->getOldForce(access::cpu, access::read);
  // for(int ii=0; ii < 1 ; ii++){
  //   std::cout << "Particle " << ii << " vel: " << vel[ii] << std::endl;
  //   std::cout << "Particle " << ii << " old_force: " << old_force[ii] << std::endl;
  // }

  auto nd = createNewtonianDynamicsIntegrator(sim);
  printFirst10Particles(sim);

  bool flag = false;

  if(flag){
    sim.pd->getSystem()->finish();
    return;
  }

  // Hold the particles in the trap using particle groups
  auto idrange = std::vector<int>(sim.par.movingParticles); std::iota(idrange.begin(), idrange.end(), 0);
  auto pg = std::make_shared<ParticleGroup>(idrange.begin(), idrange.end(), sim.pd, "MovingPlate");

  auto idrange2 = std::vector<int>(sim.par.stationaryParticles); std::iota(idrange2.begin(), idrange2.end(), sim.par.movingParticles);
  auto pg2 = std::make_shared<ParticleGroup>(idrange2.begin(), idrange2.end(), sim.pd, "StationaryPlate");

  // Apply a constant external force on the remaining particles (interior of the plates)
  auto idrange3 = std::vector<int>(sim.par.interiorParticles); std::iota(idrange3.begin(), idrange3.end(), sim.par.movingParticles + sim.par.stationaryParticles);
  auto pg3 = std::make_shared<ParticleGroup>(idrange3.begin(), idrange3.end(), sim.pd, "InteriorParticles");

  // sim.integrator = bd;
  // Enable the external potential 
  if (sim.par.k != 0){
    auto ext = createExternalPotentialInteractor(sim, pg, 0.0);
    ext->sum({.force=true, .energy=false, .virial=false});
    nd->addInteractor(ext);
    std::cout << "StationaryPlate enabled" << std::endl;

    auto ext2 = createExternalPotentialInteractor(sim, pg2, 0.0);
    ext2->sum({.force=true, .energy=false, .virial=false});
    nd->addInteractor(ext2);
    std::cout << "StationaryPlate enabled" << std::endl;
  }

  auto constant_force = createExternalForceInteractor(sim, pg3, sim.par.fext);
  constant_force->sum({.force=true, .energy=false, .virial=false});
  nd->addInteractor(constant_force);
  std::cout << "Constant external force enabled." << std::endl;

  auto inter = std::make_shared<CustomContactInteractor>(sim);
  nd->addInteractor(inter);
  std::cout << "Contacts interactor enabled." << std::endl;

  for( int step = 0; step < sim.par.Nsteps; step++){
  // for( int step = 0; step < 100; step++){
      nd->forwardTime();
      // std::cout << "Step " << step << " done." << std::endl;
      if ((step+1) % sim.par.Nwrite == 0 || step == 0){ // Write every Nwrite steps
        std::cout << "Frame " << (step+1)/sim.par.Nwrite << std::endl;  
        // printFirst10Particles(sim);
        writeSimulation(sim);
      }
      // Clean up old contacts every Ncleanup steps
      // inter->h_cleanupContacts();
  }

  // Destroy the UAMMD environment and exit
  sim.pd->getSystem()->finish();
  return 0;
}
