#ifndef UAMMD_FORCES_COMPUTEFORCES_CUH_
#define UAMMD_FORCES_COMPUTEFORCES_CUH_
#include <global/defines.h>
#include <uammd.cuh>
// #include <uammd_config.cuh>
#include "PairForces.cuh"
#include <Integrator/VelocityVerlet.cuh>
#include <Interactor/NeighbourList/CellList.cuh>
#include <Interactor/NeighbourList/VerletList.cuh>
#include <Interactor/Interactor.cuh>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include "PairForces.cuh"

using namespace uammd;

// A new way of using a neighbour list
template <class NeighbourContainer>
__global__ void processNeighboursContacts(
    real time, // Current simulation time
    NeighbourContainer ni, // Provides iterator with neighbours of a particle
    int numberParticles, Box box,
    real4 *vel, // Velocities in group indexing
    real4 *force, // Forces in group indexing
    real *energy, // Energies in group indexing
    real *virial,  // Virial in group indexing
    real dt, // Time step
    real kn, // Normal spring constant for overlap force
    real kt, // Tangential spring constant for overlap force
    real mu, // Coefficient of friction
    real gamma_n, // Damping coefficient for normal direction
    ContactManager* contact_mgr // Contact history manager

) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numberParticles)
    return;
  // Set ni to provide iterators for particle i
  ni.set(i);
  const real3 pi =
      make_real3(cub::ThreadLoad<cub::LOAD_LDG>(ni.getSortedPositions() + i));
  real3 f = real3();
  real e = 0;
  real v = 0; 
  // for(auto neigh: ni){ //This is equivalent to the while loop, although a tad
  // slower
  auto it = ni.begin(); // Iterator to the first neighbour of particle i
  // Note that ni.end() is not a pointer to the last neighbour, it just
  // represents "no more neighbours" and
  //  should not be dereferenced

  const int gli = ni.getGroupIndexes()[i];
  const real3 vi = make_real3(vel[gli]);
  while (it) { // it will cast to false when there are no more neighbours
    auto neigh = *it++; // The iterator can only be advanced and dereferenced
    int j = neigh.getGroupIndex();
    // const int glj = neigh.getGroupIndexes()[j];
    const real3 vj = make_real3(vel[j]);
    const real3 vij = vi - vj; // TODO: Account for PBCs in homogeneous shear flow using Lees-Edwards
    const real3 pj = make_real3(neigh.getPos());
    const real3 rij = box.apply_pbc(pj - pi);
    const real r2 = dot(rij, rij);
    if (r2 > 0 and r2 < (real(6.25))) {

      // TODO: reset the contact history if contact breaks using contact age
      // TODO: Account for different particle sizes here 
      // r2 < (Ri + Rj)^2
      // For now assume all particles have diameter 2 -> (1+1)^2 = 4
      if (r2 < 4.0){
        // Particles are in contact - get or create contact history
        ContactHistory* contact = contact_mgr->getContact(gli, j);
        
        // Check for existing contact history and update
        if(contact->contact_age > real(0.0)){

          // If the contact was last active in the previous timestep, reset the history
          if(contact->contact_age < (time-dt)){
            contact->xi = real3(0.0);
            contact->contact_time = real(0.0);
          }
          else{
            // Continuing contact - increment time and age
            contact->contact_time += dt;
            contact->xi += (vij - normal_component(vij)) * dt; 
          }
        }
        // Otherwise, this is a new contact with default values
        // Record the current time as the last active time 
        contact->contact_age = time;
        const real3 fmod = (force or virial) ? total_contact_force_ij(kn, kt, gamma_n, mu, dt, rij, vij, contact) : real3();
      }

      if (force)
        f += fmod;
      // if (energy)
      //   e += lj_energy(r2);
      // if (virial)
      //   v += dot(fmod, rij);
    }
  }

  if (force)
    force[gli] += make_real4(f);
  if (energy)
    energy[gli] += e;
  if (virial)
    virial[gli] += v;
}


// Define a custom interactor to compute the normal and tangential
// frictional forces at contact 
// Needs to be aware of the simulation time to update the tangential
class CustomContactInteractor : public ParameterUpdatable, public Interactor {
  using NeighbourList = VerletList;
  std::shared_ptr<NeighbourList> nl;
  std::shared_ptr<ParticleData> pd;
  std::unique_ptr<ContactManager> contact_mgr;  // Contact history manager
  real dt;
  real time = 0;
  real rcut; 
  real boxSize;
  real numberParticles; 
  real kn; // Normal spring constant for overlap force
  real kt; // Tangential spring constant for overlap force
  real mu; // Coefficient of friction
  real gamma_n; // Damping coefficient for normal direction

public: 
  CustomContactInteractor( UAMMD sim ) : 
    Interactor(sim.pd, "Custom"),
    pd(sim.pd),
    rcut(sim.par.rcut), 
    boxSize(sim.par.boxSize), 
    numberParticles(sim.par.numberParticles), 
    dt(sim.par.dt), 
    time(0),
    kn(sim.par.kn), 
    kt(sim.par.kt), 
    mu(sim.par.mu), 
    gamma_n(sim.par.gamma_n) 
     {
    nl = std::make_shared<VerletList>(sim.pd);
    // Initialize contact manager with estimated max contacts
    int max_contacts = numberParticles * 10;  // Estimate 10 contacts per particle
    contact_mgr = std::make_unique<ContactManager>(max_contacts);
  }

  virtual void updateSimulationTime(real newTime) override { time = newTime; }
  void sum(Computables comp, cudaStream_t st) override {
    Box box(boxSize);
    // Store copy of the old neighbor list 
    auto nl_old = nl;
    nl->update(box, rcut, st);
    
    // DON'T mark contacts inactive - we want to preserve history!
    // Instead, the kernel will mark contacts as active when found
    
    // NeighbourContainer can provide forward iterators with the neighbours of
    // each particle The drawback of it being a forward iterator is that it can
    // only be advanced, once you have asked for the next neighbour there is no
    // going back without starting from the first. With it=ni.begin() you can
    // only do it++, etc, there is no operator[] nor it--
    auto ni = nl->getNeighbourContainer();
    auto vel = pd->getVel(access::gpu, access::read).raw();
    auto force =
        comp.force
            ? pd->getForce(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;
    auto energy = comp.energy ? pd->getEnergy(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    auto virial = comp.virial ? pd->getVirial(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    processNeighboursContacts<<<numberParticles / 128 + 1, 128, 0, st>>>(
        time, ni, numberParticles, box, vel, force, energy, virial, dt, kn, kt, mu, gamma_n, contact_mgr.get());
  }
};

#endif // UAMMD_FORCES_COMPUTEFORCES_CUH_