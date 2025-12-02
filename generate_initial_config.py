import numpy as np
import matplotlib.pyplot as plt
from generate_lattice import Lattice3D
from hcp import generate_hcp_lattice_rectangular

def main(): 
    # Parameters
    radius = 1.0  # Particle radius
    width, height = 50.0, 50.0  # Dimensions of the rectangular region

    # Generate HCP lattice in a rectangular region, these represent the top and bottom plates
    x_centers, y_centers = generate_hcp_lattice_rectangular(width, height, radius)
    x_centers = x_centers - width / 2
    y_centers = y_centers - height / 2
    print(f"Generated {len(x_centers)} particles in each plate")

    # Generate a cubic lattice that will go between the plates
    spacing = 3.5 * radius  # Spacing between particles in the cubic lattice
    lattice = Lattice3D(spacing=spacing)
    fcc_positions = lattice.face_centered_cubic(nx=10, ny=10, nz=10)
    fcc_width = np.max(fcc_positions[:, 0]) - np.min(fcc_positions[:, 0])
    fcc_height = np.max(fcc_positions[:, 1]) - np.min(fcc_positions[:, 1])
    print(f"Generated cubic lattice with dimensions: {fcc_width} x  {fcc_height} x {np.max(fcc_positions[:, 2]) - np.min(fcc_positions[:, 2])}")
    fcc_positions[:, 0] = fcc_positions[:, 0] - fcc_width / 2
    fcc_positions[:, 1] = fcc_positions[:, 1] - fcc_height / 2

    fcc_depth = np.max(fcc_positions[:, 2]) - np.min(fcc_positions[:, 2])
    fcc_positions[:, 2] = fcc_positions[:, 2] - fcc_depth / 2

    z_centers_bottom = np.min(fcc_positions[:, 2]) - np.sqrt(3)*radius
    z_centers_bottom *= np.ones_like(x_centers)
    z_centers_top = np.max(fcc_positions[:, 2]) + np.sqrt(3)*radius
    z_centers_top *= np.ones_like(x_centers)
    print(f"Generated {len(fcc_positions)} particles in the cubic lattice")

    initial_config = np.stack((x_centers, y_centers, z_centers_bottom), axis=-1)
    initial_config = np.vstack((initial_config, np.stack((x_centers, y_centers, z_centers_top), axis=-1)))
    initial_config = np.vstack((initial_config, fcc_positions))
    
    print(f"Total particles in the system: {len(initial_config)}")
    np.savetxt("initial_config.dat", initial_config, comments=' ')
    return
    

if __name__ == "__main__":
    main()