import numpy as np
import matplotlib.pyplot as plt

class OverlapLattice2D:
    """2D lattice generator with controlled particle overlap."""
    
    def __init__(self, radius=1.0, overlap_distance=None, overlap_fraction=None):
        """
        Initialize 2D lattice with overlap.
        
        Parameters:
        radius (float): Radius of particles
        overlap_distance (float): Absolute overlap distance δ
        overlap_fraction (float): Overlap as fraction of radius (e.g., 0.05 for δ=0.05R)
        
        Note: Provide either overlap_distance OR overlap_fraction, not both.
        """
        self.radius = radius
        
        if overlap_distance is not None and overlap_fraction is not None:
            raise ValueError("Provide either overlap_distance OR overlap_fraction, not both")
        elif overlap_distance is not None:
            self.delta = overlap_distance
        elif overlap_fraction is not None:
            self.delta = overlap_fraction * radius
        else:
            # Default: 5% overlap
            self.delta = 0.05 * radius
        
        # Center-to-center distance = 2R - δ
        self.spacing = 2 * self.radius - self.delta
        
        print(f"Lattice Configuration:")
        print(f"  Particle radius: {self.radius}")
        print(f"  Overlap distance δ: {self.delta}")
        print(f"  Overlap fraction: {self.delta/self.radius:.3f} (or {100*self.delta/self.radius:.1f}%)")
        print(f"  Center-to-center spacing: {self.spacing}")
    
    def square_lattice(self, nx, ny):
        """
        Generate square lattice with overlap.
        
        Parameters:
        nx, ny (int): Number of particles in x and y directions
        
        Returns:
        tuple: (x_centers, y_centers) - 1D arrays of coordinates
        """
        x = np.arange(nx) * self.spacing
        y = np.arange(ny) * self.spacing
        
        X, Y = np.meshgrid(x, y)
        
        return X.flatten(), Y.flatten()
    
    def hexagonal_lattice(self, nx, ny):
        """
        Generate hexagonal (HCP) lattice with overlap.
        
        Parameters:
        nx, ny (int): Number of particles in x direction and rows in y
        
        Returns:
        tuple: (x_centers, y_centers) - 1D arrays of coordinates
        """
        spacing_x = self.spacing
        spacing_y = self.spacing * np.sqrt(3) / 2
        
        x_centers = []
        y_centers = []
        
        for row in range(ny):
            y = row * spacing_y
            x_offset = self.spacing / 2 if row % 2 == 1 else 0
            
            for col in range(nx):
                x = col * spacing_x + x_offset
                x_centers.append(x)
                y_centers.append(y)
        
        return np.array(x_centers), np.array(y_centers)
    
    def rectangular_fill(self, width, height, lattice_type='hexagonal'):
        """
        Fill a rectangular region with particles.
        
        Parameters:
        width, height (float): Dimensions of region to fill
        lattice_type (str): 'square' or 'hexagonal'
        
        Returns:
        tuple: (x_centers, y_centers) - 1D arrays of coordinates
        """
        if lattice_type == 'square':
            nx = int(width / self.spacing) + 1
            ny = int(height / self.spacing) + 1
            return self.square_lattice(nx, ny)
        
        elif lattice_type == 'hexagonal':
            spacing_y = self.spacing * np.sqrt(3) / 2
            nx = int(width / self.spacing) + 2
            ny = int(height / spacing_y) + 1
            
            x_centers, y_centers = self.hexagonal_lattice(nx, ny)
            
            # Filter particles outside the region
            mask = (x_centers <= width) & (y_centers <= height)
            return x_centers[mask], y_centers[mask]
        
        else:
            raise ValueError("lattice_type must be 'square' or 'hexagonal'")

def plot_lattice(x_centers, y_centers, radius=1.0):
    """
    Simple plot of 2D lattice.
    
    Parameters:
    x_centers, y_centers: Particle center coordinates
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # Plot each particle
    for i, (x, y) in enumerate(zip(x_centers, y_centers)):
        circle = plt.Circle((x, y), radius, fill=False, edgecolor='blue', 
                          linewidth=1.0, alpha=0.8)
        ax.add_patch(circle)
        
        # Mark center
        ax.plot(x, y, 'ro', markersize=3)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    ax.set_aspect('equal')
    margin = radius * 1.5
    ax.set_xlim(np.min(x_centers) - margin, np.max(x_centers) + margin)
    ax.set_ylim(np.min(y_centers) - margin, np.max(y_centers) + margin)
    
    plt.show()
    
    return
    
def plot_lattice_with_overlap(x_centers, y_centers, radius, delta, 
                               title="2D Lattice with Overlap", 
                               show_overlap=True, show_connections=True):
    """
    Plot 2D lattice showing particle overlap.
    
    Parameters:
    x_centers, y_centers: Particle center coordinates
    radius (float): Particle radius
    delta (float): Overlap distance
    title (str): Plot title
    show_overlap (bool): Highlight overlap regions
    show_connections (bool): Draw lines between overlapping particles
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    spacing = 2 * radius - delta
    
    # Plot each particle
    for i, (x, y) in enumerate(zip(x_centers, y_centers)):
        circle = plt.Circle((x, y), radius, fill=False, edgecolor='blue', 
                          linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        
        # Mark center
        ax.plot(x, y, 'ro', markersize=3)
    
    # Show overlap regions
    if show_overlap:
        for i, (x1, y1) in enumerate(zip(x_centers, y_centers)):
            for j, (x2, y2) in enumerate(zip(x_centers, y_centers)):
                if i < j:
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Check if particles overlap
                    if dist < 2 * radius:
                        overlap_depth = 2 * radius - dist
                        
                        # Draw connection line
                        if show_connections and abs(dist - spacing) < 0.01:
                            ax.plot([x1, x2], [y1, y2], 'g-', alpha=0.3, linewidth=1)
                        
                        # Highlight overlap region (lens-shaped)
                        if overlap_depth > 0.001:  # Numerical tolerance
                            # Calculate intersection points
                            dx, dy = x2 - x1, y2 - y1
                            
                            # Midpoint weighted by radii
                            mid_x = (x1 + x2) / 2
                            mid_y = (y1 + y2) / 2
                            
                            # Draw a small circle at overlap region
                            overlap_marker = plt.Circle((mid_x, mid_y), delta/2, 
                                                       fill=True, color='red', 
                                                       alpha=0.3, edgecolor='red')
                            ax.add_patch(overlap_marker)
    
    # Set equal aspect and limits
    ax.set_aspect('equal')
    margin = radius * 1.5
    ax.set_xlim(np.min(x_centers) - margin, np.max(x_centers) + margin)
    ax.set_ylim(np.min(y_centers) - margin, np.max(y_centers) + margin)
    
    # Labels and grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    # Title with parameters
    overlap_pct = 100 * delta / radius
    ax.set_title(f'{title}\n{len(x_centers)} particles | R={radius} | δ={delta:.3f} ({overlap_pct:.1f}% of R) | spacing={spacing:.3f}', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def calculate_packing_fraction(x_centers, y_centers, radius, boundary_dims=None):
    """
    Calculate packing fraction (area occupied by particles / total area).
    
    Parameters:
    x_centers, y_centers: Particle coordinates
    radius: Particle radius
    boundary_dims: (width, height) of bounding box, or None to auto-calculate
    
    Returns:
    float: Packing fraction
    """
    n_particles = len(x_centers)
    particle_area = n_particles * np.pi * radius**2
    
    if boundary_dims is None:
        width = np.max(x_centers) - np.min(x_centers) + 2 * radius
        height = np.max(y_centers) - np.min(y_centers) + 2 * radius
    else:
        width, height = boundary_dims
    
    total_area = width * height
    packing_fraction = particle_area / total_area
    
    return packing_fraction

def verify_overlap(x_centers, y_centers, radius, delta):
    """
    Verify that particles have the correct overlap distance.
    
    Parameters:
    x_centers, y_centers: Particle coordinates
    radius: Particle radius
    delta: Expected overlap distance
    
    Returns:
    dict: Statistics about particle distances
    """
    expected_spacing = 2 * radius - delta
    distances = []
    
    for i in range(len(x_centers)):
        for j in range(i+1, len(x_centers)):
            dist = np.sqrt((x_centers[j] - x_centers[i])**2 + 
                          (y_centers[j] - y_centers[i])**2)
            distances.append(dist)
    
    distances = np.array(distances)
    
    # Find nearest neighbor distances
    nearest_neighbors = []
    for i in range(len(x_centers)):
        min_dist = float('inf')
        for j in range(len(x_centers)):
            if i != j:
                dist = np.sqrt((x_centers[j] - x_centers[i])**2 + 
                              (y_centers[j] - y_centers[i])**2)
                min_dist = min(min_dist, dist)
        nearest_neighbors.append(min_dist)
    
    nearest_neighbors = np.array(nearest_neighbors)
    
    stats = {
        'min_distance': np.min(distances),
        'mean_nearest_neighbor': np.mean(nearest_neighbors),
        'std_nearest_neighbor': np.std(nearest_neighbors),
        'expected_spacing': expected_spacing,
        'spacing_error': np.abs(np.mean(nearest_neighbors) - expected_spacing)
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    print("2D Lattice Generator with Particle Overlap")
    print("=" * 70)
    
    # Example 1: Hexagonal lattice with 5% overlap
    print("\n1. Hexagonal Lattice with δ = 0.05R")
    print("-" * 70)
    
    lattice1 = OverlapLattice2D(radius=1.0, overlap_fraction=0.05)
    x1, y1 = lattice1.hexagonal_lattice(6, 5)
    
    print(f"\nGenerated {len(x1)} particles")
    
    # Verify overlap
    stats1 = verify_overlap(x1, y1, lattice1.radius, lattice1.delta)
    print(f"\nVerification:")
    print(f"  Expected spacing: {stats1['expected_spacing']:.4f}")
    print(f"  Measured nearest neighbor: {stats1['mean_nearest_neighbor']:.4f} ± {stats1['std_nearest_neighbor']:.4f}")
    print(f"  Spacing error: {stats1['spacing_error']:.6f}")
    
    plot_lattice_with_overlap(x1, y1, lattice1.radius, lattice1.delta, 
                             "Hexagonal Lattice (δ = 0.05R)")
    
    # Example 2: Square lattice with 10% overlap
    print("\n2. Square Lattice with δ = 0.10R")
    print("-" * 70)
    
    lattice2 = OverlapLattice2D(radius=1.0, overlap_fraction=0.10)
    x2, y2 = lattice2.square_lattice(5, 5)
    
    print(f"\nGenerated {len(x2)} particles")
    
    stats2 = verify_overlap(x2, y2, lattice2.radius, lattice2.delta)
    print(f"\nVerification:")
    print(f"  Expected spacing: {stats2['expected_spacing']:.4f}")
    print(f"  Measured nearest neighbor: {stats2['mean_nearest_neighbor']:.4f}")
    
    plot_lattice_with_overlap(x2, y2, lattice2.radius, lattice2.delta,
                             "Square Lattice (δ = 0.10R)")
    
    # Example 3: Different overlap values comparison
    print("\n3. Comparison of Different Overlap Values")
    print("-" * 70)
    
    for overlap_frac in [0.01, 0.05, 0.10]:
        lattice = OverlapLattice2D(radius=1.0, overlap_fraction=overlap_frac)
        x, y = lattice.hexagonal_lattice(4, 4)
        packing = calculate_packing_fraction(x, y, lattice.radius)
        print(f"\nδ = {overlap_frac:.2f}R:")
        print(f"  Spacing = {lattice.spacing:.4f}")
        print(f"  Packing fraction ≈ {packing:.3f}")
    
    # Example 4: Fill rectangular region
    print("\n4. Fill Rectangular Region (10×10) with Hexagonal Packing")
    print("-" * 70)
    
    dim = 30.0; 20.0; 10.0
    lattice4 = OverlapLattice2D(radius=1.0, overlap_fraction=0.008)
    x4, y4 = lattice4.rectangular_fill(dim, dim, lattice_type='square')

    print(f"\nGenerated {len(x4)} particles")
    packing4 = calculate_packing_fraction(x4, y4, lattice4.radius, boundary_dims=(dim, dim))
    print(f"Packing fraction: {packing4:.3f}")
    
    plot_lattice_with_overlap(x4, y4, lattice4.radius, lattice4.delta,
                             "Rectangular Fill - Hexagonal Packing")
    
    
    print("\n" + "=" * 70)
    print("All lattices generated successfully!")
    print("Red dots = particle centers")
    print("Blue circles = particle boundaries")
    print("Green lines = connections between neighbors")
    print("Red regions = overlap zones")
    
    # Display number of particles total
    # Re-arrange the indexing to have the top and bottom rows first for easy identification
    print(f"\nTotal particles in rectangular fill: {len(x4)}")
    width4 = np.max(x4) - np.min(x4)
    height4 = np.max(y4) - np.min(y4)
    
    # Shift to center around (0,0)
    x4 = x4 - width4 / 2
    y4 = y4 - height4 / 2
    
    indices_top_row = np.where(y4 == np.max(y4))[0]
    indices_bottom_row = np.where(y4 == np.min(y4))[0]
    
    # For debugging, only store the top and bottom rows to test the moving plates
    x_plate = 2 * np.arange(-2, 3)
    y_plate_top = np.full_like(x_plate, np.max(y4))
    # y_plate_bottom = np.full_like(x_plate, np.min(y4))
    # y_plate_bottom = np.full_like(x_plate, np.max(y4) - 6.5)
    y_plate_bottom = np.full_like(x_plate, np.max(y4) - 6.25)
    coords_plate_top = np.column_stack((x_plate, y_plate_top))
    coords_plate_bottom = np.column_stack((x_plate, y_plate_bottom))
    coords_plate = np.vstack((coords_plate_top, coords_plate_bottom))
    coords_plate = np.hstack((coords_plate, np.zeros((coords_plate.shape[0], 1))))
    np.savetxt("plate_positions.dat", coords_plate, comments='')
    
    # Place one particle along the moving boundary for testing friction model 
    x_test_particle = np.array([0.0, 0.0])
    y_test_particle = np.array([np.max(y_plate_top)-1.50, np.max(y_plate_top)-3.50]) # introduce slight overlap
    coords_test_particle = np.column_stack((x_test_particle, y_test_particle, np.zeros_like(x_test_particle)))
    coords_total = np.vstack((coords_plate, coords_test_particle))
    
    # Visualize
    plot_lattice(coords_total[:,0], coords_total[:,1], radius=1.0)
    
    np.savetxt("test_particle_position.dat", coords_total, comments='')
    print(f"\n No. inidices in top row: {len(indices_top_row)}")
    print(f" No. inidices in bottom row: {len(indices_bottom_row)}")
    
    # Re-arrange so top row is first, then bottom row, then the rest
    remaining_indices = [i for i in range(len(x4)) if i not in indices_top_row and i not in indices_bottom_row]
    new_order = np.concatenate((indices_top_row, indices_bottom_row, remaining_indices))
    x4 = x4[new_order]
    y4 = y4[new_order]
    
    # Save the rectangular lattice
    np.savetxt("initial_config.dat", np.column_stack((x4, y4, np.zeros_like(x4))), comments='')
    
    # Plot the lattice
    plot_lattice(x4, y4, radius=1.0)
    
    # Obtain the edge-to-edge dimensions along x and y 
    width4 = np.max(x4) - np.min(x4) + 2 * lattice4.radius
    height4 = np.max(y4) - np.min(y4) + 2 * lattice4.radius
    
    # Write a README file for the initial configuration 
    with open('README.md', 'w') as f:
        f.write("# Initial Configuration for ShearGranularMedia3d Simulation\n\n")
        f.write("This file contains the initial configuration of particles for the shear granular media simulation.\n\n")
        f.write("## Particle Data\n")
        f.write("- Each row corresponds to a particle.\n")
        f.write("- The columns represent the x and y coordinates of the particle centers.\n\n")
        f.write("## Lattice Parameters\n")
        f.write(f"- Particle radius (R): {lattice4.radius}\n")
        f.write(f"- Overlap distance (δ): {lattice4.delta} ({100*lattice4.delta/lattice4.radius:.1f}% of R)\n")
        f.write(f"- Center-to-center spacing: {lattice4.spacing}\n")
        f.write(f"- Total number of particles: {len(x4)}\n")
        f.write(f"- Dimensions of rectangular region: {width4} x {height4}\n\n")
        f.write(f" - Packing fraction: {packing4:.3f}\n\n")
        f.write(f" No. of particles in top row: {len(indices_top_row)}\n")
        f.write(f" No. of particles in bottom row: {len(indices_bottom_row)}\n")
        f.write("## Notes\n")
        f.write("- The top and bottom rows of particles are intended to represent the moving plates in the simulation.\n")
        f.write("- The configuration is centered around (0,0) for convenience.\n")