import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Lattice3D:
    """3D lattice generator with customizable spacing and structure types."""
    
    def __init__(self, spacing=1.0):
        """
        Initialize the 3D lattice generator.
        
        Parameters:
        spacing (float or tuple): Lattice spacing. Can be:
            - Single float: uniform spacing in all directions
            - Tuple (a, b, c): different spacing for x, y, z directions
        """
        if isinstance(spacing, (int, float)):
            self.spacing = (spacing, spacing, spacing)
        elif isinstance(spacing, (list, tuple)) and len(spacing) == 3:
            self.spacing = tuple(spacing)
        else:
            raise ValueError("Spacing must be a number or 3-element tuple/list")
        
        self.a, self.b, self.c = self.spacing
    
    def simple_cubic(self, nx, ny, nz):
        """
        Generate simple cubic lattice.
        
        Parameters:
        nx, ny, nz (int): Number of unit cells in each direction
        
        Returns:
        numpy.ndarray: Array of shape (N, 3) with particle coordinates
        """
        x = np.arange(nx) * self.a
        y = np.arange(ny) * self.b
        z = np.arange(nz) * self.c
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        positions = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        return positions
    
    def body_centered_cubic(self, nx, ny, nz):
        """
        Generate body-centered cubic (BCC) lattice.
        
        Parameters:
        nx, ny, nz (int): Number of unit cells in each direction
        
        Returns:
        numpy.ndarray: Array of particle coordinates
        """
        # Corner atoms
        corners = self.simple_cubic(nx, ny, nz)
        
        # Body center atoms (offset by half lattice constant)
        x_centers = (np.arange(nx-1) + 0.5) * self.a
        y_centers = (np.arange(ny-1) + 0.5) * self.b
        z_centers = (np.arange(nz-1) + 0.5) * self.c
        
        if len(x_centers) > 0 and len(y_centers) > 0 and len(z_centers) > 0:
            X_c, Y_c, Z_c = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
            body_centers = np.column_stack([X_c.flatten(), Y_c.flatten(), Z_c.flatten()])
            
            positions = np.vstack([corners, body_centers])
        else:
            positions = corners
        
        return positions
    
    def face_centered_cubic(self, nx, ny, nz):
        """
        Generate face-centered cubic (FCC) lattice.
        
        Parameters:
        nx, ny, nz (int): Number of unit cells in each direction
        
        Returns:
        numpy.ndarray: Array of particle coordinates
        """
        positions = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    base_x, base_y, base_z = i * self.a, j * self.b, k * self.c
                    
                    # Corner atom
                    positions.append([base_x, base_y, base_z])
                    
                    # Face center atoms (if not on boundary)
                    if i < nx - 1 and j < ny - 1:  # xy face center
                        positions.append([base_x + self.a/2, base_y + self.b/2, base_z])
                    if i < nx - 1 and k < nz - 1:  # xz face center
                        positions.append([base_x + self.a/2, base_y, base_z + self.c/2])
                    if j < ny - 1 and k < nz - 1:  # yz face center
                        positions.append([base_x, base_y + self.b/2, base_z + self.c/2])
        
        return np.array(positions)
    
    def hexagonal_close_packed(self, nx, ny, nz):
        """
        Generate hexagonal close-packed (HCP) lattice.
        
        Parameters:
        nx, ny, nz (int): Number of unit cells in each direction
        
        Returns:
        numpy.ndarray: Array of particle coordinates
        """
        positions = []
        
        # HCP has two atoms per unit cell
        # Standard HCP has c/a ratio ≈ 1.633 for ideal packing
        
        for k in range(nz):
            z = k * self.c
            
            for j in range(ny):
                for i in range(nx):
                    # Base position
                    x_base = i * self.a
                    y_base = j * self.b * np.sqrt(3)
                    
                    # First atom at corner
                    positions.append([x_base, y_base, z])
                    
                    # Second atom offset (alternating layers)
                    if k % 2 == 1:  # Odd layers
                        x_offset = self.a/2 if j % 2 == 0 else 0
                        y_offset = self.b * np.sqrt(3)/2
                        positions.append([x_base + self.a/2 + x_offset, 
                                        y_base + y_offset, z + self.c/2])
                    else:  # Even layers
                        positions.append([x_base + self.a/2, 
                                        y_base + self.b * np.sqrt(3)/2, z + self.c/2])
        
        return np.array(positions)
    
    def custom_lattice(self, nx, ny, nz, basis_atoms):
        """
        Generate custom lattice with user-defined basis atoms.
        
        Parameters:
        nx, ny, nz (int): Number of unit cells in each direction
        basis_atoms (list): List of (x, y, z) fractional coordinates for basis atoms
                           Each coordinate should be between 0 and 1
        
        Returns:
        numpy.ndarray: Array of particle coordinates
        """
        positions = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    base_x = i * self.a
                    base_y = j * self.b
                    base_z = k * self.c
                    
                    for frac_x, frac_y, frac_z in basis_atoms:
                        x = base_x + frac_x * self.a
                        y = base_y + frac_y * self.b
                        z = base_z + frac_z * self.c
                        positions.append([x, y, z])
        
        return np.array(positions)

def plot_3d_lattice(positions, title="3D Lattice", particle_size=50, 
                    show_bonds=False, bond_cutoff=None):
    """
    Plot 3D lattice structure.
    
    Parameters:
    positions (numpy.ndarray): Array of particle coordinates
    title (str): Plot title
    particle_size (float): Size of particles in plot
    show_bonds (bool): Whether to show bonds between particles
    bond_cutoff (float): Maximum distance to draw bonds (if None, auto-calculate)
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot particles
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              s=particle_size, c='blue', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Plot bonds if requested
    if show_bonds:
        if bond_cutoff is None:
            # Auto-calculate reasonable bond cutoff
            distances = []
            for i in range(min(len(positions), 50)):  # Sample first 50 particles
                for j in range(i+1, min(len(positions), 50)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            if distances:
                bond_cutoff = np.min(distances) * 1.1  # 10% larger than nearest neighbor
        
        # Draw bonds
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= bond_cutoff:
                    ax.plot([positions[i, 0], positions[j, 0]], 
                           [positions[i, 1], positions[j, 1]], 
                           [positions[i, 2], positions[j, 2]], 
                           'gray', alpha=0.3, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\n{len(positions)} particles')
    
    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

# Example usage and demonstrations
if __name__ == "__main__":
    print("3D Lattice Generator")
    print("=" * 50)
    
    # Example 1: Simple cubic with uniform spacing
    print("1. Simple Cubic Lattice (spacing = 2.0)")
    lattice = Lattice3D(spacing=2.0)
    positions_sc = lattice.simple_cubic(4, 4, 3)
    print(f"Generated {len(positions_sc)} particles")
    print(f"Lattice parameters: a={lattice.a}, b={lattice.b}, c={lattice.c}")
    plot_3d_lattice(positions_sc, "Simple Cubic (a=2.0)", show_bonds=True)
    
    # Example 2: BCC with different spacing in each direction
    print(f"\n2. Body-Centered Cubic (spacing = (1.5, 1.5, 2.0))")
    lattice_bcc = Lattice3D(spacing=(1.5, 1.5, 2.0))
    positions_bcc = lattice_bcc.body_centered_cubic(3, 3, 3)
    print(f"Generated {len(positions_bcc)} particles")
    plot_3d_lattice(positions_bcc, "BCC Lattice (a=1.5, b=1.5, c=2.0)", show_bonds=True)
    
    # Example 3: FCC lattice
    print(f"\n3. Face-Centered Cubic (spacing = 1.0)")
    lattice_fcc = Lattice3D(spacing=1.0)
    positions_fcc = lattice_fcc.face_centered_cubic(3, 3, 3)
    print(f"Generated {len(positions_fcc)} particles")
    plot_3d_lattice(positions_fcc, "FCC Lattice (a=1.0)", show_bonds=True)
    
    # Example 4: Custom lattice (diamond structure)
    print(f"\n4. Custom Lattice - Diamond Structure (spacing = 1.0)")
    diamond_basis = [(0, 0, 0), (0.25, 0.25, 0.25)]
    positions_diamond = lattice_fcc.custom_lattice(3, 3, 3, diamond_basis)
    print(f"Generated {len(positions_diamond)} particles")
    plot_3d_lattice(positions_diamond, "Diamond Structure", show_bonds=True)
    
    # Show coordinate examples
    print(f"\n5. Coordinate Examples (Simple Cubic, first 10 particles):")
    for i in range(min(10, len(positions_sc))):
        x, y, z = positions_sc[i]
        print(f"Particle {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    print(f"\nAll lattices generated successfully!")
    print(f"You can modify spacing, lattice type, and size as needed.")