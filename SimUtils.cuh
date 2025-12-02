#ifndef SIMUTILS_CUH
#define SIMUTILS_CUH

#include <uammd.cuh>
#include <global/defines.h>

struct UAMMD {
  std::shared_ptr<ParticleData> pd;
  std::shared_ptr<Integrator> integrator;
  Parameters par;
};

#endif