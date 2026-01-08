#ifndef CONTACTS_CUH
#define CONTACTS_CUH

#include <uammd.cuh>
#include <utils/vector.cuh>
#include <Interactor/Interactor.cuh>
#include <Interactor/NeighbourList/CellList.cuh>
#include "ExternalForcesStructs.cuh"
#include "SimUtils.cuh"

using namespace uammd;

struct ContactHistory {
  int particle_i, particle_j;     // Particle pair (always i < j)
  real3 xi;                       // Accumulated tangential distance
  real contact_time;              // How long they've been in contact
  real contact_age;                // Last timestep they were in contact
  bool is_active;                 // Whether contact exists this timestep
  
  // Constructor for new contact
  __host__ __device__ ContactHistory(int i, int j) : 
    particle_i(min(i,j)), particle_j(max(i,j)),
    xi(make_real3(0,0,0)),
    contact_time(0.0), contact_age(0.0), is_active(true) {}
    
  // Default constructor
  __host__ __device__ ContactHistory() : 
    particle_i(-1), particle_j(-1),
    xi(make_real3(0,0,0)),
    contact_time(0.0), contact_age(0.0), is_active(false) {}
};

struct ContactManager {
  ContactHistory* contacts;         // Array of contact histories
  int* contact_count;               // Number of active contacts
  int Ncleanup = 1000;              // Cleanup interval
  int max_contacts;                 // Maximum contacts we can store
  int* cleanup_count;               // Statistics for cleanup
  float cutoff_age;                   // Age after which contact is removed

  // Hash table for O(1) lookup
  static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
  static constexpr unsigned long long EMPTY_PACKED = 0xFFFFFFFFFFFFFFFFull;
  static constexpr int MAX_PROBES = 512; 
  using u64 = unsigned long long;
  struct HashEntry {
  // Upper 32 bits: key (packed (i,j))
  // Lower 32 bits: index into contacts[]
    u64 packed;
  };

  HashEntry* hash_table; 
  int hash_size; 

  // Constructor
  ContactManager(int num_particles) : cutoff_age(25.0) {
    // Estimate the max size of the hash table as ~6x number of particles
    // int expected_contacts = 6 * num_particles;
    max_contacts =  10 * num_particles; // Allow some extra space

    cudaMalloc(&contacts, max_contacts * sizeof(ContactHistory));
    cudaMalloc(&contact_count, sizeof(int));
    cudaMemset(contact_count, 0, sizeof(int));
    cudaMalloc(&cleanup_count, sizeof(int));
    cudaMemset(cleanup_count, 0, sizeof(int));

    // Initialize hash table (2x number of contacts for 50% load factor)
    hash_size = 1;
    while(hash_size < max_contacts * 2){
      hash_size *= 2;
    }
    assert((hash_size & (hash_size - 1)) == 0 && "hash_size must be power of two");

    cudaMalloc(&hash_table, hash_size * sizeof(HashEntry));
    // HashEntry empty_entry = {EMPTY_KEY, -1};

    // Initialize with empty markers
    // cudaMemset(hash_table, 0xFF, hash_size * sizeof(HashEntry));
    resetHashTable();
    // for(int i=0; i<hash_size; i++){
    //   cudaMemcpy(&hash_table[i], &empty_entry, sizeof(HashEntry), 
    //               cudaMemcpyHostToDevice);
    // }
    printf("ContactManager initialized: max_contacts=%d, hash_size=%d\n", 
               max_contacts, hash_size);
    
  }
  // Destructor
  ~ContactManager() {
    cudaFree(contacts);
    cudaFree(contact_count);
    cudaFree(hash_table);
  }

  void resetHashTable(){
    cudaMemset(hash_table, 0xFF, hash_size * sizeof(HashEntry));
  }

  __device__ __forceinline__ u64 pack_entry(uint32_t key, int index) {
    // high 32 bits = key
    u64 hi = static_cast<u64>(key) << 32;

    // low 32 bits = index (assumes index >= 0)
    u64 lo = static_cast<unsigned int>(index);

    return hi | lo;
  }

  __device__ __forceinline__ uint32_t unpack_key(unsigned long long packed) {
    return uint32_t(packed >> 32);
  }

  __device__ __forceinline__ int unpack_index(unsigned long long packed) {
    return int(packed & 0xFFFFFFFFu);
  }

  // Pack two particle IDs into one key
  __device__ uint64_t pack_key(int i, int j) {
    // Ensure consistent ordering
    uint32_t min_id = min(i, j);
    uint32_t max_id = max(i, j);
    // Assumes particle IDs < 65536 (16 bits each)
    return (uint32_t(min_id) << 16) | max_id;
  }

  // Hash function
  __device__ uint64_t hash1(uint64_t key){
    // MurmurHash3 finalizer
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (hash_size - 1);
  }

  // Secondary hash for stride
  __device__ uint64_t hash2(uint64_t key){
    // Different hash function to avoid correlation
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    
    // CRITICAL: Must return odd number for power-of-2 table sizes
    // This ensures we can probe all slots
    uint64_t stride = key & (hash_size - 1);
    return stride | 1;  // Set lowest bit to ensure odd
  }

  __device__ ContactHistory* returnContact(int index) {
    if (index < 0 || index >= *contact_count) {
      return nullptr;
    }
    return &contacts[index];
  }

  // Main function: Find existing contact or create a new one
  __device__ ContactHistory* getContact_v1(int i, int j) {
    // printf("getContact_v1 called for (%d, %d)\n", i, j);
    uint64_t key  = pack_key(i, j);
    uint64_t slot = hash1(key);
    uint64_t stride = hash2(key);

    // Reserve a contact index up-front
    int my_idx = atomicAdd(contact_count, 1);
    // printf("Contact count after reserving: %d\n", *contact_count);
    if( *contact_count <= 0 ){
      printf("Error: contact count negative after reserving index!\n");
      return nullptr;
    }
    if (my_idx >= max_contacts) {
      atomicSub(contact_count, 1);  // Undo
      printf("ContactManager full! max_contacts=%d\n", max_contacts);
      return nullptr;
    }
    // printf("Current contact count: %d\n", *contact_count);
    // Linear probe to find or insert
    for (int probe = 0; probe < MAX_PROBES; ++probe) {
      uint64_t slot_calc = slot + uint64_t(probe) * stride;
      uint64_t current_slot = slot_calc & (hash_size - 1);
      HashEntry* entry = &hash_table[current_slot];

      // Snapshot current packed value
      unsigned long long cur = entry->packed;
      uint64_t cur_key = unpack_key(cur);

      // Case 1: found existing contact with this key
      if (cur_key == key) {
        // We don't need the new index we reserved
        atomicSub(contact_count, 1);
        int idx = unpack_index(cur);
        contacts[idx].is_active = true;
        // printf("Retrieved contact for (%d, %d): idx=%d, age=%f\n", contacts[idx].particle_i, contacts[idx].particle_j, idx, contacts[idx].contact_age);
        return &contacts[idx];
        // return returnContact(idx);
      }

      // Case 2: empty slot, try to claim it
      if (cur == EMPTY_PACKED) {
        // int my_idx = unpack_index(cur);
        unsigned long long desired = pack_entry(key, my_idx);
        // unsigned long long desired = pack_entry(key, my_idx);
        
        // Try to install (key,index) atomically
        unsigned long long old = atomicCAS(&entry->packed, EMPTY_PACKED, desired);

        if (old == EMPTY_PACKED) {
          // We successfully inserted this contact
          contacts[my_idx] = ContactHistory(i, j);
          contacts[my_idx].is_active = true;
          // printf("Created new contact: i=%d, j=%d, idx=%d, age=%f\n", i, j, my_idx, contacts[my_idx].contact_age);

          // update hash entry
          // entry->packed = desired;
          return &contacts[my_idx];
          // return returnContact(my_idx);
        }

        // if (old == EMPTY_PACKED) {
        //   // We successfully inserted this contact
        //   contacts[my_idx] = ContactHistory(i, j);
        //   contacts[my_idx].is_active = true;
        //   return &contacts[my_idx];
        // }

        // Another thread beat us to this slot
        uint64_t old_key = unpack_key(old);
        if (old_key == key) {
          // Another thread inserted the same contact
          atomicSub(contact_count, 1);  // give back our unused index
          int idx = unpack_index(old);
          contacts[idx].is_active = true;
          // printf("Retrieved contact for (%d, %d): idx=%d, age=%f\n", contacts[idx].particle_i, contacts[idx].particle_j, idx, contacts[idx].contact_age);
          return &contacts[idx];
          // return returnContact(idx);
        }

        // Otherwise, different key grabbed this slot; continue probing
        continue;
      }

      
      // Case 3: occupied by a different key, keep probing
    }

    // Failed to insert or find after MAX_PROBES
    atomicSub(contact_count, 1);  // release reserved index
    printf("ContactManager hash table full or too many collisions!\n");
    return nullptr;
  }

  // Find existing contact or create new one
  __device__ ContactHistory* getContact(int i, int j) {
    int min_id = min(i, j);
    int max_id = max(i, j);
    
    // Search for existing contact
    
    for (int idx = 0; idx < *contact_count; idx++) {
      if (contacts[idx].particle_i == min_id && 
          contacts[idx].particle_j == max_id) {
        contacts[idx].is_active = true;
        // printf("Found existing contact between %d and %d at index %d\n", min_id, max_id, idx);
        return &contacts[idx];
      }
    }
    
    // Create new contact if space available
    int new_idx = atomicAdd(contact_count, 1);
    if (new_idx < max_contacts) {
      contacts[new_idx] = ContactHistory(min_id, max_id);
      return &contacts[new_idx];
    }
    
    return nullptr; // No space for new contact
  }
};

__inline__ __device__ real3 total_contact_force_ij(
    real kn, real kt, 
    real gamma_n, 
    real gamma_t,
    real mu, real dt, 
    real3 rij, real3 vij, 
    ContactHistory* contact, real radius_sum){
      if(( contact != nullptr) && ( contact->is_active)){
        real3 fn = overlap_force_ij(kn, rij, radius_sum); 
        // printf("Normal overlap force: %f, %f, %f\n", fn.x, fn.y, fn.z);
        // fn += normal_frictional_force_ij(gamma_n, vij, rij);
        fn -= damping_force_ij(gamma_n, vij, rij, true);
        // printf("Normal frictional force: %f, %f, %f\n", fn.x, fn.y, fn.z);
        // Update tangential displacement xi
        real3 ft = tangential_frictional_force_ij(kt, contact->xi, mu, fn);
        ft -= damping_force_ij(gamma_t, vij, rij, false);
        // printf("tangential frictional force: %f, %f, %f\n\n", ft.x, ft.y, ft.z);
        // real3 ft = make_real3(0,0,0);
        return fn + ft; 
      }
      return make_real3(0,0,0);
}

// __global__ void gpu_cleanupContacts(
//   ContactManager *contact_mgr, // Contact history manager
//   ContactManager::HashEntry *hash_table, // Hash table to update
//   real time);

// __global__ void gpu_rebuildHashTable(
//   ContactManager *contact_mgr, // Contact history manager
//   ContactManager::HashEntry *hash_table // Hash table to rebuild
//   );

// template <class NeighbourContainer>
// __global__ void processNeighboursContacts(
//     NeighbourContainer ni, // Provides iterator with neighbours of a particle
//     real time, // Current simulation time
//     int Nwrite,
//     int numberParticles,
//     Box box,
//     real *radius, // Radii in group indexing
//     real3 *vel, // Velocities in group indexing
//     real4 *ang_vel, // Angular velocities in group indexing
//     real4 *force, // Forces in group indexing
//     real4 *torque, // Torques in group indexing
//     real *energy, // Energies in group indexing
//     real *virial,  // Virial in group indexing
//     real3 *stress_x, // Stress components in group indexing
//     real3 *stress_y,
//     real3 *stress_z,
//     real dt, // Time step
//     real kn, // Normal spring constant for overlap force
//     real kt, // Tangential spring constant for overlap force
//     real mu, // Coefficient of friction
//     real gamma_n, // Damping coefficient for normal direction
//     real gamma_t, // Damping coefficient for tangential direction
//     ContactManager *contact_mgr // Contact history manager
//   );

// class CustomContactInteractor: public ParameterUpdatable, public Interactor{
//   using NeighbourList = CellList;
//   std::shared_ptr<NeighbourList> nl;
//   std::shared_ptr<ParticleData> pd;
//   std::shared_ptr<ContactManager> contact_mgr;  // Contact history manager
//   ContactManager *d_contact_mgr = nullptr; // Device pointer to contact manager
//   real dt;
//   real time = 0;
//   uint64_t steps = 0;
//   int Nwrite;
//   real rcut; 
//   real3 boxSize;
//   int numberParticles; 
//   real kn; // Normal spring constant for overlap force
//   real kt; // Tangential spring constant for overlap force
//   real mu; // Coefficient of friction
//   real gamma_n; // Damping coefficient for normal direction
//   real gamma_t; // Damping coefficient for tangential direction
//   uint64_t cleanup_interval; // Interval for cleaning inactive contacts
// public: 
//     CustomContactInteractor( UAMMD sim ) : 
//         Interactor(sim.pd, "Custom"),
//         pd(sim.pd),
//         rcut(sim.par.rcut), 
//         boxSize(sim.par.boxSize), 
//         numberParticles(sim.par.numberParticles), 
//         dt(sim.par.dt), 
//         time(0),
//         Nwrite(sim.par.Nwrite),
//         kn(sim.par.kn), 
//         kt(sim.par.kt), 
//         mu(sim.par.mu), 
//         gamma_n(sim.par.gamma_n),
//         gamma_t(sim.par.gamma_t)
//         {
//             cleanup_interval = uint64_t(1.0 / dt); // Clean up every 1.0 time units
//             // cleanup_interval = 1250;
//             nl = std::make_shared<CellList>(sim.pd);
//             // Initialize contact manager with estimated max contacts
//             // int max_contacts = numberParticles * 1000;  // Estimate 10 contacts per particle
//             contact_mgr = std::make_shared<ContactManager>(numberParticles);
//             // create device copy of contact manager
//             cudaError_t err = cudaMalloc((void**)&d_contact_mgr, sizeof(ContactManager));
//             if (err != cudaSuccess) {
//                 throw std::runtime_error("Failed to allocate device memory for ContactManager");
//             }
//             err = cudaMemcpy(d_contact_mgr, contact_mgr.get(), sizeof(ContactManager), cudaMemcpyHostToDevice);
//             if (err != cudaSuccess) {
//                 cudaFree(d_contact_mgr);
//                 throw std::runtime_error("Failed to copy ContactManager to device");
//             }
//         }

//     // Inline destructor
//     ~CustomContactInteractor() override; 
//     virtual void updateSimulationTime(real newTime) override;
//     // ~CustomContactInteractor() override {
//     //     if (d_contact_mgr) {
//     //             cudaFree(d_contact_mgr);
//     //             d_contact_mgr = nullptr;
//     //         }
//     // }

//     // void updateSimulationTime(real newTime) override { time = newTime; }
//     void h_cleanupContacts(cudaStream_t st);
//     void sum(Computables comp, cudaStream_t stream) override;

// };
#endif