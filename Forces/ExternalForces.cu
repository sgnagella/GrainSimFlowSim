#include "ExternalForces.cuh"


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