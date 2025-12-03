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
#include <thrust/transform_reduce.h>
#include <utils/vector.cuh>
#include <utils/InitialConditions.cuh>

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
  real kt = 0.5;
  // real kt = 15.0;
  // real kt = 10.0;
  real kn = 1.0; 
  real k = 100.0; // Spring constant for the harmonic well
  // real kn = 3.5 * kt;
  // real k = 3.5 * kt;
  real mu = 0.25; // Coefficient of friction
  real gamma_n = 0.25; // Damping coefficient for normal direction
  real gamma_t = 0.25; // Damping coefficient for tangential direction
  real3 fext = make_real3(0.01, 0.0, 0.0); // External force on interior particles
  // real gamma_n = 0.005;
  bool is2D = true;
  // real3 Kx = make_real3(0.0, 0.0, 0.0); // shear flow in x-direction whose gradient lies along y
  // real shearRate = 0.1; // shear rate for Lees-Edwards BC  
  // real strainrate = 2.0; // speed of the bottom "plate"

  int Nsteps = 395978;
  // int Nsteps = 100000;
  // int Nsteps = 2;
  // int Nwrite = 1250; // Write every (1/dt) steps (1 time unit)
  int Nwrite = 1584; // Steps required for plate particle to travel its size 
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
        return -gamma * normal_component(vij, rij);
    }
    return -gamma * tangential_component(vij, rij);
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
        xi = ( mu * fn_magnitude/ ft_magnitude ) * xi;
    }
    return static_force_ij(kt, xi); 
}

__device__ real3 compute_stress_i(real3 rij, real3 fij){
    // Stress contribution from a pairwise interaction
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
        fn += damping_force_ij(gamma_n, vij, rij, true);
        // printf("Normal frictional force: %f, %f, %f\n", fn.x, fn.y, fn.z);
        // Update tangential displacement xi
        real3 ft = tangential_frictional_force_ij(kt, contact->xi, mu, fn);
        ft += damping_force_ij(gamma_t, vij, rij, false);
        // printf("tangential frictional force: %f, %f, %f\n\n", ft.x, ft.y, ft.z);
        // real3 ft = make_real3(0,0,0);
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
  real3 sx = real3();
  real3 sy = real3();
  real3 sz = real3();
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
  // bool writeStep = ( ((time+1) % Nwrite) == 0 );
  int nneigh = 0;
  while (it) { // it will cast to false when there are no more neighbours
    nneigh++;
    auto neigh = *it++; // The iterator can only be advanced and dereferenced
    int j = neigh.getGroupIndex();
    // const int glj = neigh.getGroupIndexes()[j];
    const real3 vj = vel[j];
    const real3 vij = vi - vj; // TODO: Account for PBCs in homogeneous shear flow using Lees-Edwards
    const real3 pj = make_real3(neigh.getPos());
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
        // printf("Particle %d interacting with %d\n", gli, j);
        // Particles are in contact - get or create contact history
        ContactHistory *contact = contact_mgr->getContact(gli, j);
        // printf("Contact age before update: %d\n", contact->contact_age);
        // Check for existing contact history and update

        if(contact->contact_time > 0){
          // printf("Existing contact between %d and %d found with age %d\n", contact->particle_i, contact->particle_j, contact->contact_age);

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
        contact->contact_time += 0.5*dt; // +1/2 for each particle in the pair during counting
        // printf("Contact time between %d and %d is now %f\n", contact->particle_i, contact->particle_j, contact->contact_time);
        contact->xi += (gli > j ? real(1) : real(-1) ) * 0.5 * ( (vij - normal_component(vij, rij)) + cross_product_2D( avij, rij*rsqrt(dot(rij,rij)) ) ) * dt; 
        // contact->xi += (gli > j ? real(1) : real(-1) ) * 0.5 * ( (vij - normal_component(vij, rij)) ) * dt; 
        // printf("Updated xi between %d and %d is %10f, %10f, %10f\n", contact->particle_i, contact->particle_j, contact->xi.x, contact->xi.y, contact->xi.z);
  
        // printf("Contact between %d and %d active at time %f\n", contact->particle_i, contact->particle_j, time);
        // printf("contact between %d and %d has xi %f, %f, %f\n", contact->particle_i, contact->particle_j, contact->xi.x, contact->xi.y, contact->xi.z);
        // printf("Contact between %d and %d has relative velocity %f, %f, %f\n", contact->particle_i, contact->particle_j, vij.x, vij.y, vij.z);
        const real3 fmod = (force or virial) ? total_contact_force_ij(kn, kt, gamma_n, gamma_t, mu, dt, rij, vij, contact, radii_sum) : real3();
        const real3 tmod = (torque) ? ri * static_torque_ij_2D(fmod, rij) : real3();

        // printf("Force between %d and %d is %f, %f, %f\n\n", contact->particle_i, contact->particle_j, fmod.x, fmod.y, fmod.z);
        if (force){
          f += fmod;
          // t += tmod;
        }
        if (torque){
          // printf("Torque between %d and %d is %f, %f, %f\n\n", contact->particle_i, contact->particle_j, tmod.x, tmod.y, tmod.z);
          t += tmod;
        } 
        if (stress_x){
          sx += compute_stress_i(make_real3(rij.x), fmod);
        }
        if (stress_y){
          sy += compute_stress_i(make_real3(rij.y), fmod);
        }
        if (stress_z){
          sz += compute_stress_i(make_real3(rij.z), fmod);
        }
        // if (energy)
        //   e += lj_energy(r2);
        // if (virial)
        //   v += dot(fmod, rij);
      }

    }
  }

  if (force){
    force[gli] += make_real4(f);
    // torque[gli] += make_real4(t);
    // printf("Total force on particle %d: %f, %f, %f\n", gli, f.x, f.y, f.z);
    // printf("Total torque on particle %d: %f, %f, %f\n", gli, t.x, t.y, t.z);
  }
  if (stress_x){
    stress_x[gli] += (nneigh > 0) ? sx / nneigh : real3();
  }
  if (stress_y){
    stress_y[gli] += (nneigh > 0) ? sy / nneigh : real3();
  }
  if (stress_z){
    stress_z[gli] += (nneigh > 0) ? sz / nneigh : real3();
  }
  if (torque)
    torque[gli] += make_real4(t);
  if (energy)
    energy[gli] += e;
  if (virial)
    virial[gli] += v;
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
  int Nwrite;
  real rcut; 
  real3 boxSize;
  int numberParticles; 
  real kn; // Normal spring constant for overlap force
  real kt; // Tangential spring constant for overlap force
  real mu; // Coefficient of friction
  real gamma_n; // Damping coefficient for normal direction
  real gamma_t; // Damping coefficient for tangential direction

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
      int max_contacts = numberParticles * 1000;  // Estimate 10 contacts per particle
      contact_mgr = std::make_shared<ContactManager>(max_contacts);
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
  void sum(Computables comp, cudaStream_t st) override {
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

  // int Nsteps = 312500;
  // int Nsteps = 395978;
  // int Nsteps = 100000;
  // int Nsteps = 2;
  // int Nwrite = 1250; // Write every (1/dt) steps (1 time unit)
  // int Nwrite = 1584; // Steps required for plate particle to travel its size 
  for( int step = 0; step < sim.par.Nsteps; step++){
      nd->forwardTime();
      std::cout << "Step " << step << " done." << std::endl;
      if ((step+1) % sim.par.Nwrite == 0 || step == 0){ // Write every Nwrite steps
        std::cout << "Frame " << (step+1)/sim.par.Nwrite << std::endl;  
        printFirst10Particles(sim);
        writeSimulation(sim);

      }
  }

  // Destroy the UAMMD environment and exit
  sim.pd->getSystem()->finish();
  return 0;
}
