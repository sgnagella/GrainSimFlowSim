

/**
 * @file PairForces.cu
 * @brief CUDA kernels for computing pairwise forces between particles.
 *
 * This file contains CUDA device functions and kernels to compute
 * pairwise forces based on particle positions and velocities.
 *
 * The main function `wrap_contact_force_ij` calculates the contact force
 * between two particles considering normal and tangential components.
 *
 * The code assumes particles are represented in 3D space with real3 vectors.
 * The force calculations include overlap forces and frictional forces.
 * 
    * @author Sachit Nagella
/** */

#include <uammd.cuh>
#include "PairForces.cuh"
using namespace uammd;

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
        return mu * fn_magnitude * (xi / length(xi)); 
    }
    return static_force_ij(kt, xi); 
}

__device__ real3 total_contact_force_ij(
    real kn, real kt, 
    real gamma_n, 
    real mu, real dt, 
    real3 rij, real3 vij, 
    const ContactHistory &contact){
        real3 fn = overlap_force_ij(kn, rij); 
        fn += normal_frictional_force_ij(gamma_n, vij);
        // Update tangential displacement xi
        if( contact != nullptr && contact->is_active ){
            real3 ft = tangential_frictional_force_ij(kt, contact.xi, mu, fn);
        }
    return fn + ft;  
}
