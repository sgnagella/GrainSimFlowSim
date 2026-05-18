# GPU-Accelerated DEM: Pressure-Driven Granular Flow Between Parallel Plates

![Pressure-driven granular flow between parallel plates](renderings/pressure_flow_grains.gif)

*Caption*

A GPU-accelerated Discrete Element Method (DEM) simulation of 2D pressure-driven flow of a bidisperse dense granular packing confined between parallel plates. Grain dynamics are integrated using a modified [UAMMD](https://github.com/sgnagella/GrainSim) library (Newton–Euler integrator). Hysteretic contact forces (Luding 2008) are tracked via a custom GPU hash table with double-hashing and sequential probing, enabling nearly O(1) pair lookups and a ~100× speedup over naive linear search.

---

## Physical Setup

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  ← stationary plate (harmonic)│
 │  ○ ● ○ ● ○ ● ○ ● ○ ● ○ ● ○ ● ○ ●  ← bidisperse interior grains │
 │         ──────────────→  F_ext                                   │
 │  ○ ● ○ ● ○ ● ○ ● ○ ● ○ ● ○ ● ○ ●                               │
 │  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●  ← stationary plate (harmonic)│
 └──────────────────────────────────────────────────────────────────┘
```

| Parameter | Value |
|---|---|
| Total particles | 100,996 |
| Interior (flow) particles | 100,000 |
| Plate particles (each wall) | 498 |
| Box dimensions (L_x × L_y) | 996.9 × 502.1 × 1.0 |
| Aspect ratio L_x / L_y | 2 |
| External body force F_ext | 0.01 (x-direction) |
| Normal spring constant k_n | 1.0 |
| Tangential spring constant k_t | 2/7 |
| Coulomb friction coefficient μ | 0.5 |
| Normal damping γ_n | 0.1 |
| Tangential damping γ_t | γ_n / 2 |
| Plate harmonic stiffness k | 10,000 |
| Time step dt | 10⁻⁴ |
| Particle mass | γ_n³ (low-Stokes regime) |
| Temperature | 0 (athermal) |

- **Bidisperse packing**: two disk sizes to suppress crystallisation and maintain amorphous dense packing.
- **Plates**: rows of disk particles tethered to their initial positions by a harmonic spring potential, acting as rigid confining walls.
- **Boundary conditions**: periodic in x and y; confinement is provided by the plate spring force rather than hard walls.

---

## Contact Model

The force law follows **Luding (2008)**:

> S. Luding, "Cohesive, frictional powders: contact models for tension," *Granular Matter* **10**, 235–246 (2008). https://doi.org/10.1007/s10035-008-0099-x

For an overlapping pair (i, j) with separation vector **r**_ij (from j to i), overlap δ = (R_i + R_j) − |**r**_ij|, relative velocity **v**_ij, and accumulated tangential displacement **ξ**:

**Normal force:**
```
f_n = k_n δ r̂_ij  −  γ_n (v_ij · r̂_ij) r̂_ij
```

**Tangential force (hysteretic, Coulomb-limited):**
```
f_t = −k_t ξ  −  γ_t v_t,ij
```
with slip condition `|f_t| ≤ μ |f_n|`.

The tangential displacement **ξ** is integrated over the contact lifetime and reset when the contact breaks. Tracking **ξ** across timesteps requires persistent contact history — the central algorithmic challenge addressed by the hash table below.

---

## Hash Table for Contact History (`ShearGranularMedia2d.cu`)

### Why a hash table?

Each GPU thread handles one particle and must look up or create the contact history for every neighbour currently in contact. A naive linear scan over all stored contacts is O(N_contacts) per pair — prohibitive at ~10⁵ particles. The custom GPU hash table reduces this to amortised O(1).

### Data layout

Each entry in the hash table packs the 32-bit key and a 32-bit contact-array index into a single 64-bit word for lock-free atomic compare-and-swap:

```cuda
// ContactManager — excerpt from ShearGranularMedia2d.cu

static constexpr uint32_t EMPTY_KEY   = 0xFFFFFFFF;
static constexpr unsigned long long EMPTY_PACKED = 0xFFFFFFFFFFFFFFFFull;
static constexpr int MAX_PROBES = 512;

struct HashEntry {
    // Upper 32 bits: packed (i,j) key
    // Lower 32 bits: index into contacts[]
    unsigned long long packed;
};

HashEntry* hash_table;   // device array, size = hash_size (power of two)
ContactHistory* contacts; // device array, size = max_contacts
int* contact_count;       // device scalar, atomically incremented
int hash_size;            // next power of two above 2 × max_contacts
```

### Key packing

Particle IDs are always stored in canonical order (min, max) so the key is symmetric:

```cuda
__device__ uint64_t pack_key(int i, int j) {
    uint32_t min_id = min(i, j);
    uint32_t max_id = max(i, j);
    // Assumes particle IDs < 65536 (fits in 16 bits each)
    return (uint32_t(min_id) << 16) | max_id;
}
```

### Double hashing

Two independent hash functions eliminate primary clustering. `hash2` is forced to be odd so it steps through all slots in a power-of-two table:

```cuda
// Primary slot
__device__ uint64_t hash1(uint64_t key) {
    // MurmurHash3 finalizer
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (hash_size - 1);
}

// Stride (must be odd for power-of-two table — guarantees full coverage)
__device__ uint64_t hash2(uint64_t key) {
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    uint64_t stride = key & (hash_size - 1);
    return stride | 1;  // set lowest bit → always odd
}
```

### Sequential probing with atomic CAS

`getContact_v1` is called from every GPU thread during force evaluation. It reserves a contact-array slot up front, then walks the hash table until it either finds an existing record or claims an empty slot via `atomicCAS`:

```cuda
__device__ ContactHistory* getContact_v1(int i, int j) {
    uint64_t key    = pack_key(i, j);
    uint64_t slot   = hash1(key);
    uint64_t stride = hash2(key);

    // Reserve a contact slot atomically
    int my_idx = atomicAdd(contact_count, 1);
    if (my_idx >= max_contacts) {
        atomicSub(contact_count, 1);
        return nullptr;
    }

    for (int probe = 0; probe < MAX_PROBES; ++probe) {
        uint64_t current_slot = (slot + uint64_t(probe) * stride)
                                & (hash_size - 1);
        HashEntry* entry = &hash_table[current_slot];
        unsigned long long cur = entry->packed;

        // Case 1: key already present — return existing record
        if (unpack_key(cur) == key) {
            atomicSub(contact_count, 1);          // give back unused slot
            int idx = unpack_index(cur);
            contacts[idx].is_active = true;
            return &contacts[idx];
        }

        // Case 2: empty slot — try to claim it
        if (cur == EMPTY_PACKED) {
            unsigned long long desired = pack_entry(key, my_idx);
            unsigned long long old =
                atomicCAS(&entry->packed, EMPTY_PACKED, desired);

            if (old == EMPTY_PACKED) {
                // Successfully claimed — initialise new contact
                contacts[my_idx] = ContactHistory(i, j);
                contacts[my_idx].is_active = true;
                return &contacts[my_idx];
            }
            // Another thread beat us; check if it inserted the same key
            if (unpack_key(old) == key) {
                atomicSub(contact_count, 1);
                int idx = unpack_index(old);
                contacts[idx].is_active = true;
                return &contacts[idx];
            }
            // Different key — keep probing
        }
        // Case 3: occupied by different key — keep probing
    }

    atomicSub(contact_count, 1);
    return nullptr;  // table full or too many collisions
}
```

The hash table is initialised at 50% load factor (size = next power of two above 2 × max_contacts), keeping the expected probe length below 2.

---

## Global Memory Cleanup and Compaction

Without periodic maintenance the `contacts[]` array fills with stale entries — pairs that separated long ago but whose slots were never reclaimed — and `contact_count` grows monotonically until the table overflows. A four-phase host routine `h_cleanupContacts()` runs every `cleanup_interval = 1/dt` steps (once per simulation time unit) to recover this memory.

### Phase 1 — Eviction (`gpu_cleanupContacts`)

One GPU thread per contact-array slot checks whether `contact_age < time` (i.e., the pair was not in contact during the most recent force evaluation). If so, the thread locates the corresponding hash-table entry by re-running the same double-hash probe sequence and atomically swaps it back to `EMPTY_PACKED` via `atomicCAS`. The slot is then marked `is_active = false`.

```cuda
if (contact->contact_age < time) {
    // re-derive slot with hash1/hash2 ...
    atomicCAS(&entry->packed, desired, EMPTY_PACKED);
    contact->is_active = false;
}
```

### Phase 2 — Compaction (Thrust `partition`)

After eviction, the `contacts[]` array has gaps: active entries are interspersed with inactive ones. A Thrust `partition` on the device moves all active contacts to a dense prefix, eliminating the gaps without a full copy:

```cuda
auto new_end = thrust::partition(policy, d_begin, d_end,
    [] __device__ (const ContactHistory& c) { return c.is_active; });
int new_count = thrust::distance(d_begin, new_end);
```

### Phase 3 — Hash table rebuild (`gpu_rebuildHashTable`)

Because compaction changes every contact's index within `contacts[]`, the hash table — which stores `(key, index)` pairs — is now stale. It is wiped to `EMPTY_PACKED` and rebuilt from scratch: one thread per compacted slot re-inserts its `(key, new_index)` pair using the same double-hash/CAS logic as `getContact_v1`.

### Phase 4 — Counter reset

`contact_count` is overwritten with `new_count` so subsequent `atomicAdd` allocations start from the correct high-water mark, and the `cleanup_count` diagnostic counter is zeroed for the next interval.

Together these four phases bound memory growth: the occupied fraction of `contacts[]` after each cleanup reflects only the contacts that are geometrically active at that moment, keeping the hash table well below its overflow threshold for arbitrarily long runs.

---

## Repository Structure

```
.
├── ShearGranularMedia2d.cu      # Main simulation: integrator setup, force kernels,
│                                #   ContactManager hash table, I/O
├── PairForces.cuh               # Device functions: overlap, damping, tangential forces
├── ComputeForces.cuh            # CustomContactInteractor (Verlet list wrapper)
├── hasher/hasher.cuh            # Hash utility stubs (superseded by inline implementation)
├── hash.h                       # Minimal hash include
├── SimUtils.cuh                 # Miscellaneous simulation utilities
├── Makefile                     # NVCC build (C++17, -O3, links to GrainSim/UAMMD)
│
├── generate_lattice_2d.py       # 2D HCP lattice generator for plate particles
├── generate_initial_config.py   # Assembles full initial configuration
│
├── make_gsd.py                  # Converts particles3.dat → traj.gsd (HOOMD/OVITO)
├── truncate_gsd.py              # Extracts last N frames from a GSD trajectory
├── unwrap_traj.py               # Unwraps periodic images for MSD/diffusion analysis
├── msd.py                       # Mean-squared displacement calculation
├── write_stress_npz.py          # Parses stress.dat → .npz arrays
│
├── input/
│   ├── initial_positions.dat    # Initial (x, y, z) coordinates for all particles
│   └── radii.dat                # Per-particle radii (bidisperse distribution)
│
└── view_velocity_field.ovito    # OVITO pipeline for velocity-field visualisation
```

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [UAMMD / GrainSim](https://github.com/sgnagella/GrainSim) | Newton–Euler DEM integrator, particle data, Verlet list |
| CUDA ≥ 11 | GPU kernels, `atomicCAS`, Thrust |
| Thrust | Device vectors, partition, permutation iterators |
| CUB | `ThreadLoad<LOAD_LDG>` cache hints |
| Python ≥ 3.9 | Pre/post-processing scripts |
| NumPy, SciPy, Matplotlib | Analysis and plotting |
| [gsd](https://gsd.readthedocs.io/) | Trajectory file I/O (HOOMD format) |

---

## Building

```bash
# Set UAMMD_ROOT to your GrainSim clone
make UAMMD_ROOT=../../GrainSim
```

This compiles `ShearGranularMedia2d.cu` with `nvcc -std=c++17 -O3 --extended-lambda`.

---

## Running a Simulation

**1. Generate the initial configuration**

```bash
python generate_lattice_2d.py   # creates the HCP plate rows
python generate_initial_config.py  # writes input/initial_positions.dat
```

**2. Run the simulation**

```bash
./ShearGranularMedia2d
```

Output files written every `N_write = 1/dt = 1000` steps:

| File | Contents |
|---|---|
| `particles3.dat` | `x y z vx vy vz img_x img_y img_z` per particle per frame |
| `stress.dat` | Per-particle stress tensor components |

**3. Convert to GSD for visualisation**

```bash
python make_gsd.py          # → traj.gsd
python truncate_gsd.py      # → traj_truncated_last_1000.gsd  (steady-state frames)
```

Open `traj.gsd` in [OVITO](https://www.ovito.org/) or use the included `view_velocity_field.ovito` pipeline.

---

## References

- S. Luding, "Cohesive, frictional powders: contact models for tension," *Granular Matter* **10**, 235–246 (2008). https://doi.org/10.1007/s10035-008-0099-x
- R. P. Pelaez, UAMMD — *Universally Adaptable Multiscale Molecular Dynamics*, https://github.com/RaulPPelaez/UAMMD
