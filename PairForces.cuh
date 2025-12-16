#ifndef PAIR_FORCES_CUH
#define PAIR_FORCES_CUH
#include <uammd.cuh>
#include <global/defines.h>
#include <utils/vector.cuh>

// Bring the uammd types (real, real3, etc.) into this header's scope so
// headers that use bare `real3` don't need to qualify with `uammd::`.
using namespace uammd;

// Utility kernels to compute pair forces
// __device__ real3 normal(real3 rij){
//   return rij / sqrtf(dot(rij, rij));
// }

__inline__ __device__ real3 normal_component(real3 v) {
  real3 nij = normalize(v);
  real3 vn = make_real3(dot(v, nij));
  vn.x *= nij.x;
  vn.y *= nij.y;
  vn.z *= nij.z; 
  return vn;
}

__inline__ __device__ real3 tangential_component(real3 v) {
  return v - normal_component(v);
}

// All functions here assume the overlap state (r < (a + b)) is already checked
__device__ real3 overlap_force_ij(real kn, real3 rij){
    real r = length(rij); 
    real delta = 2.0 - r;
    return -kn * delta * (rij / r); 
}

__device__ real3 normal_frictional_force_ij(real gamma_n, real3 vij){
    return -gamma_n * normal_component(vij);
}

__device__ real3 static_force_ij(real kt, real3 xi){
    return -kt * xi; 
}

__device__ real3 tangential_frictional_force_ij(real kt, real3 xi, real mu, real3 fn){
    real ft_magnitude = length(static_force_ij(kt, xi));
    real fn_magnitude = length(fn); 
    if( ft_magnitude > mu * fn_magnitude ){
        return make_real3(mu * fn_magnitude/length(xi)) * xi; 
    }
    return static_force_ij(kt, xi); 
}

struct ContactHistory {
  int particle_i, particle_j;     // Particle pair (always i < j)
  real3 xi;                       // Accumulated tangential distance
  real contact_time;              // How long they've been in contact
  int contact_age;                // Last timestep they were in contact
  bool is_active;                 // Whether contact exists this timestep
  
  // Constructor for new contact
  __host__ __device__ ContactHistory(int i, int j) : 
    particle_i(min(i,j)), particle_j(max(i,j)),
    xi(make_real3(0,0,0)),
    contact_time(0.0), contact_age(0), is_active(true) {}
    
  // Default constructor
  __host__ __device__ ContactHistory() : 
    particle_i(-1), particle_j(-1),
    xi(make_real3(0,0,0)),
    contact_time(0.0), contact_age(0), is_active(false) {}
};

__device__ real3 total_contact_force_ij(
    real kn, real kt, 
    real gamma_n, 
    real mu, real dt, 
    real3 rij, real3 vij, 
    ContactHistory* contact){
      if(( contact != nullptr) && ( contact->is_active)){
        real3 fn = overlap_force_ij(kn, rij); 
        fn += normal_frictional_force_ij(gamma_n, vij);
        // Update tangential displacement xi
        real3 ft = tangential_frictional_force_ij(kt, contact->xi, mu, fn);
        return fn + ft; 
      }
      return make_real3(0,0,0);
}

struct ContactManager {
  ContactHistory* contacts;         // Array of contact histories
  int* contact_count;               // Number of active contacts
  int max_contacts;                 // Maximum contacts we can store
  int cutoff_age;                   // Age after which contact is removed

  ContactManager(int max_size) : max_contacts(max_size), cutoff_age(10) {
    cudaMalloc(&contacts, max_contacts * sizeof(ContactHistory));
    cudaMalloc(&contact_count, sizeof(int));
    cudaMemset(contact_count, 0, sizeof(int));
  }
  
  ~ContactManager() {
    cudaFree(contacts);
    cudaFree(contact_count);
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
  
  // Mark all contacts as inactive before processing timestep
  // NOTE: CUDA __global__ kernels cannot be member functions. The kernel is
  // defined as a free function below and a host helper can launch it.
  
  // Remove inactive contacts (call from host)
  // void cleanupContacts() {
  //   // Launch kernel to compact active contacts
  //   // (Implementation would use thrust::remove_if or custom compaction)
  // }

  // TODO: Remove inactive contacts 
  // Remove inactive contacts (call from host)
  // void cleanupContacts() 
};

// CUDA kernel (free function) to mark contacts inactive in parallel.
__global__ void markAllInactiveKernel(ContactHistory* contacts, int* contact_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < *contact_count) {
    contacts[idx].is_active = false;
  }
}



#endif

