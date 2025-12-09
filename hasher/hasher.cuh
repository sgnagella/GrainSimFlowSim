#ifndef HASHER_CUH
#define HASHER_CUH

// Heterogeneous hasher that can hash both cuco::pair and tuple types without conversion.
// The template allows it to accept any key type and extract the first two elements.
struct heterogeneous_hasher {
  template <typename Key>
  __device__ std::size_t operator()(Key const& key) const
  {
    auto const& ref  = thrust::raw_reference_cast(key);
    auto const major = cuda::std::get<0>(ref);  // Works for both pair.first and get<0>(tuple)
    auto const minor = cuda::std::get<1>(ref);  // Works for both pair.second and get<1>(tuple)
    return static_cast<std::size_t>(major * 131 + minor);
  }
};

// Heterogeneous equality functor that can compare cuco::pair and tuple types without conversion.
// The template allows it to accept any combination of key types and compare their first two
// elements.
struct heterogeneous_key_equal {
  template <typename LHS, typename RHS>
  __device__ bool operator()(LHS const& lhs, RHS const& rhs) const
  {
    auto const& left  = thrust::raw_reference_cast(lhs);
    auto const& right = thrust::raw_reference_cast(rhs);
    return (cuda::std::get<0>(left) == cuda::std::get<0>(right)) and  // Compare first elements
           (cuda::std::get<1>(left) == cuda::std::get<1>(right));     // Compare second elements
  }
};

#endif