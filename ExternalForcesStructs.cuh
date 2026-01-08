#ifndef EXTERNAL_FORCES_CUH
#define EXTERNAL_FORCES_CUH

#include <uammd.cuh>
#include <utils/vector.cuh>

using namespace uammd;

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

#endif