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
// #include "external.cuh" // Our external potential
// #include <cuco/static_map.cuh>
// #include <cuco/dynamic_map.cuh>
// #include <cuda/functional>
// #include <cuda/std/tuple>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/count.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/transform_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
// #include <thrust/transform_reduce.h>
#include <utils/vector.cuh>
#include <utils/InitialConditions.cuh>

#include "hasher/hasher.cuh"
// #include "SimUtils.cuh"
using namespace uammd;

// Lets group here a few parameters that our example is going to use. For the
// time being, lets simply hardcode some values
// Later, we will see how to read these parameters from a file.
struct Parameters {
  int numberParticles = 286;
  int movingParticles = 15; 
  int stationaryParticles = 15; 
  int interiorParticles = numberParticles - movingParticles - stationaryParticles;
  real3 boxSize = make_real3(38.01401138305664, 42.782547, 1.0); // Size of the box in each direction

  real mass = 0.001; // Mass of the particles (Stokes number is small ~ O(10^-2))
  real viscosity = 1.0 / (6 * M_PI);
  real hydrodynamicRadius = 1.0;
  real temperature = 0.0;
  real dt = 0.0008; // Time integration step
  // real k = 5.0; // Spring constant for the harmonic well
  // real k = 10.0;
  real3 center = {0, 0, 0}; // Center of the harmonic well
  real gravity = 0.0; // Gravity strength
  real kwall = 1.0; // Strength of the soft wall potential
  // real zwall = -boxSize * 0.5; // Position of the wall
  real rcut = 2.5 * 1.0; // Cutoff for the LJ interaction
  // real kn = 5.0; // Normal spring constant for overlap force
  // real kn = 3.5; 
  // real kt = 1.0; // Tangential spring constant for overlap force
  real kt = (2./7.);
  // real kt = 0.0;
  // real kt = 15.0;
  // real kt = 10.0;
  real kn = 1.0; 
  real k = 100.0; // Spring constant for the harmonic well
  // real kn = 3.5 * kt;
  // real k = 3.5 * kt;
  real mu = 0.33; // Coefficient of friction
  real gamma_n = 0.25; // Damping coefficient for normal direction
  // real gamma_n = 0.0;
  real gamma_t = 0.25; // Deamping coefficient for tangential direction
  // real gamma_t = 0.0;
  real3 fext = make_real3(0.01, 0.0, 0.0); // External force on interior particles
  // real gamma_n = 0.005;
  bool is2D = true;
  // real3 Kx = make_real3(0.0, 0.0, 0.0); // shear flow in x-direction whose gradient lies along y
  // real shearRate = 0.1; // shear rate for Lees-Edwards BC  
  // real strainrate = 2.0; // speed of the bottom "plate"

  // int Nsteps = 395978;
  // int Nwrite = 1250; // Write every (1/dt) steps (1 time unit)
  int Nwrite = 1250; // Steps required for plate particle to travel its size 
  // int Nwrite = 1;
  int Nsteps = 1000 * Nwrite;
  // int Nsteps = 1 * Nwrite;

};

// I like to place these basic UAMMD objects in a struct so it is easy to pass
// them around
struct UAMMD {
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<Integrator> integrator;
  Parameters par;
};

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

// struct ExternalTorqueField: public ParameterUpdatable {
//     real time;
//     bool isPeriodic = true; // If true, apply minimum image convention

//     // TODO: parallelize this function
//     // inline void __host__ getExistingTorque(std::shared_ptr<ParticleData> pd) const{
//     //     // Get the current particle torques after all other torques have been calcuated
//     //     int numParticles = pd->getNumParticles();
//     //     auto torque = pd->getTorque(access::cpu, access::readwrite);
//     //     for( int ii = 0; ii < numParticles; ii++ ){
//     //       torque[ii] = make_real4(0.0, 0.0, 0.0, 0.0);
//     //     }
//     //     return;
//     // }

//     // MovingHarmonicField(Parameters par):
//     ExternalTorqueField(UAMMD sim):
//         time(0) { 
//           // getExistingTorque(sim.pd);
//         }
    
//     __device__ ForceEnergyVirial sum(Interactor::Computables comp, real3 *torque){
//         // Torques on the boundary particles to cancel out any rotation
//         torque = make_real3(0.0, 0.0, 0.0);
//         real3 f = comp.force ? 0: 0;
//         real energy = comp.energy ? 0: 0; 
//         real virial = comp.virial ? 0: 0; 

//         return {f, energy, virial};
//     }

//     auto getArrays(ParticleData *pd) {
//         auto torque = pd->getTorque(access::gpu, access::read);
//         return std::make_tuple(torque.begin());
//     }

//     virtual void updateSimulationTime(real newTime) override { time = newTime; }

// };

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
  std::cout << "Name\tposition" << std::endl;
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

// auto createExternalTorqueInteractor(UAMMD sim, std::shared_ptr<ParticleGroup> pg) {
//   auto well = std::make_shared<ExternalTorqueField>(sim);
//   auto ext = std::make_shared<ExternalForces<ExternalTorqueField>>(pg, well);
//   return ext;
// }

__inline__ __device__ real3 normal_component(real3 v, real3 rij) {
  real3 nij = rij * rsqrt(dot(rij, rij));
  // real3 vn = dot(v, nij) * nij;
  return dot(v, nij) * nij;
}

__inline__ __device__ real3 tangential_component(real3 v, real3 rij) {
  return v - normal_component(v, rij);
}

__inline__ __device__ real3 cross_product_2D(real3 a, real3 b) {
    return make_real3(0.0, 0.0, a.x * b.y - a.y * b.x);
}

// All functions here assume the overlap state (r < (a + b)) is already checked
__device__ real3 overlap_force_ij(real kn, real3 rij, real radius_sum){
    real r = length(rij); 
    real delta = r - radius_sum; // Negative for overlapping particles
    return kn * delta * (rij / r); 
}

__device__ real3 damping_force_ij(real gamma, real3 vij, real3 rij, bool isNormal){
    if( gamma == 0.0 ) return make_real3(0.0, 0.0, 0.0);
    if( isNormal ){
        return gamma * normal_component(vij, rij);
    }
    return gamma * tangential_component(vij, rij);
}

__device__ real3 normal_frictional_force_ij(real gamma_n, real3 vij, real3 rij){
    if( gamma_n == 0.0 ) return make_real3(0.0, 0.0, 0.0);
    return -gamma_n * normal_component(vij, rij);
}

// __device__ real3 tangential_frictional_force_ij(real gamma_t, real3 vij)

__device__ real3 static_force_ij(real kt, real3 xi){
    return -kt * xi; 
}

__device__ real3 static_torque_ij_2D(real3 fij, real3 rij){
    // Returns scalar torque in the z direction for 2D case
    real r = length(rij); 
    return make_real3(0.0, 0.0, cross_product_2D(rij/r, fij).z);
}

__device__ real3 tangential_frictional_force_ij(real kt, real3 &xi, real mu, real3 fn){
    if (kt == 0.0) return make_real3(0.0, 0.0, 0.0);
    real ft_magnitude = kt * length(xi);
    // real ft_magnitude = length(static_force_ij(kt, xi));
    real fn_magnitude = length(fn); 
    if( ft_magnitude > mu * fn_magnitude ){
      // Set xi max
      xi = ( mu * fn_magnitude/ ft_magnitude ) * xi; // normalizes xi (tangent)
    }
    return static_force_ij(kt, xi); 
}

__device__ real3 compute_stress_i(real3 rij, real3 fij){
    // Stress contribution from a pairwise interaction
    // printf("rij: %f, %f, %f | fij: %f, %f, %f\n", rij.x, rij.y, rij.z, fij.x, fij.y, fij.z);
    return make_real3( rij.x * fij.x, rij.y * fij.y, rij.z * fij.z );
}

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

// __device__ real3 compute_stress_ij(
//   real3 rij, real3 force, bool dissipative=false){
//   // if(dissipative){
//   //   return 0.5 * ( compute_stress_i(rij, force) + compute_stress_i(force, rij) );
//   // }
//   // return compute_stress_i(rij, force);
//   // }

//   return 0.5 * ( compute_stress_i(rij, force) + compute_stress_i(force, rij) );
// }

__device__ real3 compute_stress_ij(
  real3 rij, real3 force, int idx){
  if(idx > 2){ printf("Error: stress index out of bounds\n"); return make_real3(0.0, 0.0, 0.0); }
  if(idx==0){
    return 0.5 * ( compute_stress_i(make_real3(rij.x), force) + 
                   compute_stress_i(make_real3(force.x), rij) );
  }
  if(idx==1){
    return 0.5 * ( compute_stress_i(make_real3(rij.y), force) + 
                   compute_stress_i(make_real3(force.y), rij) );
  }
  if(idx==2){
    return 0.5 * ( compute_stress_i(make_real3(rij.z), force) + 
                   compute_stress_i(make_real3(force.z), rij) );
  }
}

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
  static constexpr int MAX_PROBES = 128; 
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

  // __device__ HashEntry* returnHashEntry(int slot) {
  //   if (slot < 0 || slot >= hash_size) {
  //     return nullptr;
  //   }
  //   return &hash_table[slot];
  // }

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


  // ~ContactManager() {
  //   cudaFree(contacts);
  //   cudaFree(contact_count);
  // }
  
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

  // struct is_inactive{
  //   int current_time;
  //   __host__ __device__
  //   bool operator()(const ContactHistory& contact){
  //     return !contact.is_active || ((current_time - contact.contact_age) > cutoff_age);
  //   }
  // };
  
  // void __device__ cleanupContacts(float time){
  //   if(time%Ncleanup != 0) return; // Only cleanup every Ncleanup steps
  //   thrust::device_ptr<ContactHistory> d_begin(contacts);
  //   thrust::device_ptr<ContactHistory> d_end = d_begin + (*contact_count);

  //   // Remove inactive contacts
  //   auto new_end = thrust::remove(d_begin, d_end, is_inactive{time});
    
  //   // Update contact count
  //   int new_count = thrust::distance(d_begin, new_end);
  //   cudaMemcpy(contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
  // }

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


// struct ContactManager {
//   ContactHistory* contacts;         // Array of contact histories
//   int* contact_count;               // Number of active contacts
//   int Ncleanup = 1000;              // Cleanup interval
//   int max_contacts;                 // Maximum contacts we can store
//   float cutoff_age;                   // Age after which contact is removed

//   // Hash table for O(1) lookup
//   static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
//   static constexpr int MAX_PROBES = 64; 

//   unsigned long long* hash_table;
//   int hash_size; 

//   // Constructor
//   ContactManager(int num_particles) : cutoff_age(25.0) {
//     // Estimate the max size of the hash table as ~6x number of particles
//     int expected_contacts = 6 * num_particles;
//     max_contacts = expected_contacts * 2; // Allow some extra space

//     cudaMalloc(&contacts, max_contacts * sizeof(ContactHistory));
//     cudaMalloc(&contact_count, sizeof(int));
//     cudaMemset(contact_count, 0, sizeof(int));

//     // Initialize hash table (2x number of contacts for 50% load factor)
//     hash_size = 1;
//     while(hash_size < max_contacts){
//       hash_size *= 2;
//     }

//     cudaMalloc(&hash_table, hash_size * sizeof(unsigned long long));
//     // HashEntry empty_entry = {EMPTY_KEY, -1};

//     // Initialize with empty markers
//     for(int i=0; i<hash_size; i++){
//       cudaMemcpy(&hash_table[i], &EMPTY_KEY, sizeof(unsigned long long), 
//                   cudaMemcpyHostToDevice);
//     }
//     printf("ContactManager initialized: max_contacts=%d, hash_size=%d\n", 
//                max_contacts, hash_size);
//   }
//   // Destructor
//   ~ContactManager() {
//     cudaFree(contacts);
//     cudaFree(contact_count);
//     cudaFree(hash_table);
//   }

//   // Pack two particle IDs into one key
//   __device__ uint32_t pack_key(int i, int j) {
//     // Ensure consistent ordering
//     uint16_t min_id = min(i, j);
//     uint16_t max_id = max(i, j);
//     // Assumes particle IDs < 65536 (16 bits each)
//     return (uint32_t(min_id) << 16) | max_id;
//   }

//   // Hash function
//   __device__ uint32_t hash(uint32_t key){
//     // MurmurHash3 finalizer
//     key ^= key >> 16;
//     key *= 0x85ebca6b;
//     key ^= key >> 13;
//     key *= 0xc2b2ae35;
//     key ^= key >> 16;
//     return key & (hash_size - 1);
//   }

//   // Main function: Find existing contact or create a new one
//   __device__ ContactHistory* getContact_v1(int i, int j){
//     uint32_t key = pack_key(i,j); 
//     uint32_t slot = hash(key); 

//     // CHANGE: Allocate contact index BEFORE trying to claim slot
//     int my_idx = atomicAdd(contact_count, 1);
//     if (my_idx >= max_contacts) {
//         atomicSub(contact_count, 1);  // Undo
//         return nullptr;
//     }
//     // Prepare empty value (do this outside the loop)
//     unsigned long long empty_value = 
//         (unsigned long long)(EMPTY_KEY) << 32 | 0xFFFFFFFFULL;

//     // Linear probe to find or insert 
//     for(int probe = 0; probe < MAX_PROBES; probe++){
//       uint32_t current_slot = (slot + probe) & (hash_size -1);

//       unsigned long long current = hash_table[current_slot];
//       uint32_t stored_key = (uint32_t)(current >> 32);
//       int32_t stored_idx = (int32_t)(current & 0xFFFFFFFFULL);

//       if (stored_key == key) {
//         // Found it - index is guaranteed valid
//         contacts[stored_idx].is_active = true;
//         return &contacts[stored_idx];
//       }

//       if (stored_key == EMPTY_KEY){
//         // Allocate contact first
//         int new_idx = atomicAdd(contact_count, 1);
//         if(new_idx >= max_contacts){
//           atomicSub(contact_count, 1); // Undo
//           return nullptr;
//         }

//         // Create new contact
//         contacts[new_idx] = ContactHistory(i, j);

//         // Prepare new entry value
//         unsigned long long new_entry = 
//             ((unsigned long long)key << 32) | (unsigned long long)new_idx;

//         // Correct atomicCAS syntax for 64-bit
//         unsigned long long old = atomicCAS(
//             (unsigned long long*)&hash_table[current_slot],
//             empty_value,
//             new_entry
//         );

//         if (old == empty_value) {
//           // Success!
//           return &contacts[new_idx];
      
//         } else {
//           // Failed - undo allocation
//           atomicSub(contact_count, 1);

//           // Check if it's our key
//           uint32_t old_key = (uint32_t)(old >> 32);
//           if (old_key == key) {
//             int32_t idx = (int32_t)(old & 0xFFFFFFFFULL);
//             contacts[idx].is_active = true;
//             return &contacts[idx];
//           }
//           // Otherwise continue probing
//         }
//       }
//     }
//     return nullptr;
//   }

//   // struct is_inactive{
//   //   int current_time;
//   //   __host__ __device__
//   //   bool operator()(const ContactHistory& contact){
//   //     return !contact.is_active || ((current_time - contact.contact_age) > cutoff_age);
//   //   }
//   // };
  
//   // void __device__ cleanupContacts(float time){
//   //   if(time%Ncleanup != 0) return; // Only cleanup every Ncleanup steps
//   //   thrust::device_ptr<ContactHistory> d_begin(contacts);
//   //   thrust::device_ptr<ContactHistory> d_end = d_begin + (*contact_count);

//   //   // Remove inactive contacts
//   //   auto new_end = thrust::remove(d_begin, d_end, is_inactive{time});
    
//   //   // Update contact count
//   //   int new_count = thrust::distance(d_begin, new_end);
//   //   cudaMemcpy(contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
//   // }

//   // Mark all contacts as inactive before processing timestep
//   // NOTE: CUDA __global__ kernels cannot be member functions. The kernel is
//   // defined as a free function below and a host helper can launch it.
  
//   // Remove inactive contacts (call from host)
//   // void cleanupContacts() {
//   //   // Launch kernel to compact active contacts
//   //   // (Implementation would use thrust::remove_if or custom compaction)
//   // }

//   // TODO: Remove inactive contacts 
//   // Remove inactive contacts (call from host)
//   // void cleanupContacts() 
// };


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
  uint64_t cleanup_interval = 1250; // Interval for cleaning inactive contacts

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

      // Reset cleanup count
      // int zero = 0;
      // cudaMemcpy(contact_mgr->cleanup_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

      // Copy contact data back to host for compaction
      // cudaMemcpy(contact_mgr->contacts, d_contact_mgr->contacts, 
      //            contact_mgr->max_contacts * sizeof(ContactHistory), 
      //            cudaMemcpyDeviceToHost);
      // cudaMemcpy(contact_mgr->contact_count, d_contact_mgr->contact_count, 
      //            sizeof(int), 
      //            cudaMemcpyDeviceToHost);

      std::cout << "Compacting contacts array after cleanup..." << std::endl;
      // thrust::device_ptr<ContactHistory> d_begin(contact_mgr->contacts);
      // thrust::device_ptr<ContactHistory> d_end = d_begin + contact_mgr->max_contacts;  
      // Compact the contacts array to remove inactive contacts
      // auto new_end = thrust::remove_if(
      //   thrust::device,
      //   contact_mgr->contacts,
      //   contact_mgr->contacts + cleaned,
      //   [] __device__ (const ContactHistory& contact) { return !contact.is_active; }
      // );

      auto new_end = thrust::partition(
        // thrust::device,
        policy,
        d_begin,
        d_end,
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
      );
      std::cout << "Compaction complete." << std::endl;
      // Obtain new contact count by summing over active contacts
      int new_count = thrust::distance(d_begin, new_end);
      // int new_count = thrust::count_if(
      //   d_begin,
      //   new_end, 
      //   [] __device__ (const ContactHistory& contact) { return contact.is_active; }
      // );
      std::cout << "After compaction: " << new_count << " active contacts (removed " 
                << (h_count - new_count) << ")" << std::endl;

      // int new_count = thrust::count_if(
      //     thrust::device, 
      //     contact_mgr->contacts, 
      //     contact_mgr->contacts + ((contact_mgr->max_contacts)), 
      //     [] __device__ (const ContactHistory& contact) { return contact.is_active; }
      //   );
      // cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
      // std::cout << "Updated contact count after compaction: " << new_count << std::endl;

      // Step 4: Update contact counts
      cudaMemset(contact_mgr->contact_count, 0, sizeof(int));
      cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), 
      cudaMemcpyHostToDevice);

      // Reset cleanup counter
      cudaMemset(contact_mgr->cleanup_count, 0, sizeof(int));

      // Copy updated contact data back to device
      // cudaMemcpy(d_contact_mgr->contacts, contact_mgr->contacts, 
      //            contact_mgr->max_contacts * sizeof(ContactHistory), 
      //            cudaMemcpyHostToDevice);
      // cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
      
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
    
    // int steps = (time/dt)+1;
    // if (steps % 100 == 0) {
    //   printf("Forcing sync at step %lld...\n", steps);
    //   fflush(stdout);
      
    //   auto start = std::chrono::high_resolution_clock::now();
    //   cudaStreamSynchronize(st);
    //   auto end = std::chrono::high_resolution_clock::now();
      
    //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //   if (duration.count() > 1000) {
    //     printf("WARNING: Sync took %lld ms - kernel is too slow!\n", duration.count());
        
    //     // Force cleanup even if not scheduled
    //     printf("Forcing emergency cleanup due to slow kernel\n");
    //     h_cleanupContacts(st);
    //   }
    // }
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

  //   // std::cout << "Contact cleanup check at time " << time << std::endl;
  //   // Periodically clean up inactive contacts 
  //   steps = (time/dt)+1;
  //   if ( (steps % cleanup_interval) == 0){
  //     cudaError_t err;      
  //     // fprintf(stdout, "Cleaning up inactive contacts at time %f (step %d)\n", time, steps);
  //     gpu_cleanupContacts<<<contact_mgr->max_contacts / 128 + 1, 128, 0, st>>>(d_contact_mgr, time);
  //     // gpu_print_debug<<<1,1,0,st>>>();
  //     err = cudaGetLastError();
  //     if (err != cudaSuccess) {
  //       fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
  //       exit(1);
  //     }
  //     err = cudaDeviceSynchronize();
  //     if (err != cudaSuccess) {
  //       fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
  //       exit(1);
  //     }

  //     // Print number of cleaned contacts
  //     int h_cleanup_count = 0;
  //     cudaMemcpy(&h_cleanup_count, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
  //     std::cout << "Cleaned up " << h_cleanup_count << " inactive contacts at time " << time << std::endl;
  //     // Reset cleanup count
  //     int zero = 0;
  //     cudaMemcpy(contact_mgr->cleanup_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
  //   }

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

  //   // Hasher initialization 
  // std::size_t num_entries = 8 * sim.par.numberParticles;
  // using KeyType = int; 
  // using ValueType = int;
  // auto constexpr empty_key_sentinel = -1;
  // auto constexpr empty_value_sentinel = -1;

  // // Allocate a map with ~50% load factor.
  // // auto map =
  // // cuco::static_map{cuco::extent<std::size_t>{num_entries},
  // //                   cuco::empty_key{empty_key},
  // //                   cuco::empty_value{empty_value},
  // //                   heterogeneous_key_equal{},
  // //                   cuco::linear_probing<1, heterogeneous_hasher>{heterogeneous_hasher{}}};

  // cuco::dynamic_map<KeyType,ValueType> map{num_entries, 
  //                              cuco::empty_key{empty_key_sentinel},
  //                              cuco::empty_value{empty_value_sentinel}};

  auto inter = std::make_shared<CustomContactInteractor>(sim);
  nd->addInteractor(inter);
  std::cout << "Contacts interactor enabled." << std::endl;
  
  // if (sim.par.k != 0){
  //   // Add external torques on boundary particles to cancel out induced rotations 
  //   // from interactions
  //   auto torque_ext = createExternalTorqueInteractor(sim, pg);
  //   auto torque_ext2 = createExternalTorqueInteractor(sim, pg2);
  //   torque_ext->sum({.force=false, .torque=true, .energy=false, .virial=false});
  //   torque_ext2->sum({.force=false, .torque=true, .energy=false, .virial=false});
  //   nd->addInteractor(torque_ext2);
  //   nd->addInteractor(torque_ext);
  // }

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
