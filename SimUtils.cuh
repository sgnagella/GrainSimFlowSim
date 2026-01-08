#ifndef SIMUTILS_CUH
#define SIMUTILS_CUH

#include <uammd.cuh>
#include <global/defines.h>

using namespace uammd;

struct Parameters {
  int numberParticles = 286;
  int movingParticles = 15; 
  int stationaryParticles = 15; 
  int interiorParticles = numberParticles - movingParticles - stationaryParticles;
  real3 boxSize = make_real3(38.01401138305664, 42.782547, 1.0); // Size of the box in each direction

  real viscosity = 1.0 / (6 * M_PI);
  real hydrodynamicRadius = 1.0;
  real temperature = 0.0;
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
  real gamma_n = 0.1; // Damping coefficient for normal direction
  real mass = 0.1 * gamma_n * gamma_n; // Mass of the particles (Stokes number is small ~ O(10^-2))
  real dt = 0.008 * gamma_n; // Time integration step
  // real gamma_n = 0.0;
  real gamma_t = 0.5 * gamma_n; // Deamping coefficient for tangential direction
  // real gamma_t = 0.0;
  real3 fext = make_real3(0.0001, 0.0, 0.0); // External force on interior particles
  // real gamma_n = 0.005;
  bool is2D = true;
  // real3 Kx = make_real3(0.0, 0.0, 0.0); // shear flow in x-direction whose gradient lies along y
  // real shearRate = 0.1; // shear rate for Lees-Edwards BC  
  // real strainrate = 2.0; // speed of the bottom "plate"

  // int Nsteps = 395978;
  // int Nwrite = 1250; // Write every (1/dt) steps (1 time unit)
  int Nwrite = int(1./dt); // Steps required for plate particle to travel its size 
  // int Nwrite = 1;
  int Nsteps = 500 * Nwrite;
  // int Nsteps = 1 * Nwrite;

};

struct UAMMD {
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<Integrator> integrator;
  Parameters par;
};

#endif