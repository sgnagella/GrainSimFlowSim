#ifndef GPU_CONTACTS_CUH
#define GPU_CONTACTS_CUH
#include <uammd.cuh>
#include <utils/vector.cuh>
#include <Interactor/Interactor.cuh>
#include <Interactor/NeighbourList/CellList.cuh>
#include "ExternalForcesStructs.cuh"
using namespace uammd;

__device__ real3 total_contact_force_ij(
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

__global__ void gpu_cleanupContacts(
  ContactManager *contact_mgr, // Contact history manager
  ContactManager::HashEntry *hash_table, // Hash table to update
  real time);

__global__ void gpu_rebuildHashTable(
  ContactManager *contact_mgr, // Contact history manager
  ContactManager::HashEntry *hash_table // Hash table to rebuild
  );

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
  );



#endif