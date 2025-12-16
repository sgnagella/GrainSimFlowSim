from cProfile import label
import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

def displacement(pos1, pos2, box):
    """
    Calculate the displacement between two positions considering periodic boundary conditions.
    """
    delta = pos2 - pos1
    delta -= box * np.floor(delta / box + 0.5)
    return delta

def get_velocities(
    traj_file, 
    dt=1.0,
): 
    """
    Get the velocity field from a GSD trajectory file.
    """
    
    # Load the trajectory
    traj = gsd.hoomd.open(traj_file, 'r')    
    # Get the number of frames and particles
    num_frames = len(traj)
    num_particles = traj[0].particles.N
    
    # Initialize arrays to store positions and velocities
    velocities = np.zeros((num_frames, num_particles, 3))
    positions = np.zeros((num_frames, num_particles, 3))
    box = traj[0].configuration.box[:3]
    
    # Loop through each frame and extract positions and velocities
    for idx, frame in enumerate(traj[:-1]):
        positions[idx] = frame.particles.position
        positions_next = traj[idx + 1].particles.position
        velocities[idx] = displacement(positions[idx], positions_next, box) / dt

    velocities[-1] = displacement(positions[-2], positions[-1], box) / dt  # Last frame velocity (using last two frames)
    return velocities, positions, box

def avg_velocity_1D(velocities, axis=0): 
    """
    Average the velocity field along a specified axis.
    """
    return np.mean(velocities, axis=axis)

def bin_velocities_1D(
    positions, 
    velocities, 
    box, 
    bin_size=1.5,
    axis=0,
    direction=1
):
    """
        Bin the velocities into a grid along a specified axis and calculate the average velocity in each bin.
    """
    
    Nframes, Nparticles, _ = positions.shape
    min_coord = box[axis] * -0.5
    max_coord = box[axis] * 0.5
    
    bins = np.arange(min_coord, max_coord + bin_size, bin_size)
    binned_velocities = np.zeros((Nframes, len(bins) - 1, 3))
    
    for idx in range(Nframes):
        indices = np.digitize(positions[idx, :, axis], bins)
        counts = np.zeros(len(bins)-1)
        for jdx in range(Nparticles):
            bin_idx = indices[jdx] - 1
            if 0 <= bin_idx < len(bins) - 1:
                binned_velocities[idx, bin_idx] += velocities[idx, jdx]
                counts[bin_idx] += 1
        
        counts[counts==0] = 1
        binned_velocities[idx] /= counts[:, None]
        
    return binned_velocities, bins

def bin_velocities(
    positions,
    velocities,
    box, 
    dim=2,
    bin_size=1.5
):
    """
        Bin the velocities into a grid and calculate the average velocity in each bin.
    """
    Nframes, Nparticles, _ = positions.shape
    
    # Determine the bounds of the grid
    min_coords = box[:dim] * -0.5
    max_coords = box[:dim] * 0.5
    bins = [np.arange(min_coords[i], max_coords[i] + bin_size, bin_size) for i in range(dim)]

    # Initialize arrays to store binned velocities and counts
    input_shape = (Nframes,) + tuple(len(bins[i]) - 1 for i in range(dim)) + (dim,)
    binned_velocities = np.zeros( input_shape )

    input_shape = tuple( len(bins[i]) - 1 for i in range(dim) ) 
    
    # Bin the velocities
    for idx, frame_velocities in enumerate(velocities):
        indices = [np.digitize(positions[idx, :, i], bins[i]) - 1 for i in range(dim)]
        counts = np.zeros( input_shape )
        for jdx in range(frame_velocities.shape[0]):
            if dim == 2:
                x_idx, y_idx = indices[0][jdx], indices[1][jdx]
                binned_velocities[idx, x_idx, y_idx, 0] += frame_velocities[jdx, 0]
                binned_velocities[idx, x_idx, y_idx, 1] += frame_velocities[jdx, 1]
                counts[x_idx, y_idx] += 1
                
            elif dim == 3:
                x_idx, y_idx, z_idx = indices[0][jdx], indices[1][jdx], indices[2][jdx]
                binned_velocities[idx, x_idx, y_idx, z_idx] += frame_velocities[jdx]
                counts[x_idx, y_idx, z_idx] += 1
        

        # Avoid division by zero
        counts[counts == 0] = 1
        binned_velocities[idx] /= counts[..., None]

    return binned_velocities, bins

def plot_velocity_field_1D(
    avg_velocities, 
    bins, 
    idx=1, 
    save=False, 
    loglog=False,
    var=False):
    """
    Plot the average velocity field along a specified axis.
    """
    labels = ['Vx', 'Vy', 'Vz']
    if var:
        labels = [r'$\delta$' + label + r'$^2$' for label in labels]
        
    plt.figure(figsize=(10, 5))
    bins -= bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(
        centers, 
        avg_velocities[...,idx], 
        'ko', 
        markersize=5,
        label=labels[idx])
    plt.plot(centers, np.zeros_like(centers), 'b--')
    plt.xlabel('Position')
    plt.ylabel('Average Velocity')
    if loglog:
        # centers/= np.max(np.abs(centers))
        # plt.xscale('log')
        plt.xlim(right=centers[-3])
        plt.ylim(bottom=1e-5)
        plt.yscale('log')
    plt.legend()
    if save:
        save_label = f'avg_velocity_{labels[idx]}_logy_{loglog}.png'
        plt.savefig(save_label)
        data = {
            'centers': centers,
            'avg_velocities': avg_velocities[..., idx]
        }
        np.savez(f'avg_velocity_{labels[idx]}_var_{var}.npz', **data)
        
    plt.show()
        
def plot_velocity_field_1D_time_slices(
    avg_velocities, 
    bins, 
    idx=1, 
    slices=10,
    start_from=0,
    save=False
):
    """
    Plot the average velocity field along a specified axis for different time slices.
    """
    labels = ['Vx', 'Vy', 'Vz']
    cmap = plt.get_cmap('hot')
    colors = cmap(np.linspace(0, 1, len(bins) - 1))
    plt.figure(figsize=(10, 5))
    bins -= bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    times = np.arange(avg_velocities.shape[0])
    for time_idx in times[start_from::slices]:
        plt.plot(
            centers, 
            avg_velocities[time_idx, :, idx], 
            color=colors[time_idx % len(colors)],
            label=f'Time {time_idx}'
            )
        
    plt.xlabel('Position')
    plt.ylabel('Average Velocity')
    plt.legend()
    
    if save:
        plt.savefig(f'avg_velocity_{labels[idx]}_time_slices.png')
        
    plt.show()

    
def plot_velocity_field_1D_time(avg_velocities, bins, idx=1):
    """
    Plot the average velocity field along a specified axis over time.
    """
    labels = ['Vx', 'Vy', 'Vz']
    plt.figure(figsize=(10, 5))
    xmin = bins[0]
    bins -= xmin
    plt.plot((bins[:-1] + bins[1:]) / 2, avg_velocities[0], label=f'Time 0')
    plt.xlabel('Position')
    plt.ylabel('Average Velocity')
    plt.xscale('log')
    plt.yscale('log')
    def update(frame):
        plt.clf()
        plt.plot((bins[:-1] + bins[1:]) / 2, avg_velocities[frame], label=f'Time {frame}')
        plt.xlabel('Position')
        plt.ylabel('Average Velocity')
        plt.legend()
        plt.title(f'Average Velocity at Time {frame}')
        # xmin = bins[0]
        # bins -= xmin
        # plt.xscale('log')
        # plt.yscale('log')
        plt.pause(0.1)

    ani = animation.FuncAnimation(
        plt.gcf(), update, frames=range(avg_velocities.shape[0])
    )
    return ani

def plot_velocity_field(binned_velocities, bins):
    """
    Plot the velocity field using quiver plot.
    """
    X, Y, Z = np.meshgrid(
        (bins[0][:-1] + bins[0][1:]) / 2,
        (bins[1][:-1] + bins[1][1:]) / 2,
        (bins[2][:-1] + bins[2][1:]) / 2,
        indexing='ij'
    )
    
    U = binned_velocities[..., 0]
    V = binned_velocities[..., 1]
    W = binned_velocities[..., 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Velocity Field')
    plt.show()
    
def main():
    bin_size = 3.0
    traj_file = '../traj.gsd'  # Replace with your GSD trajectory file
    velocities, positions, box = get_velocities(traj_file)
    
    # binned_velocities, bins = bin_velocities(velocities, velocities, box, bin_size=2.5) # size (Nframes, Nx, Ny, 3)
    
    avg_velocities_x, bins = bin_velocities_1D(
        positions, 
        velocities, 
        box, 
        bin_size=bin_size, 
        axis=1
        ) # size (Nframes, Ny, 3)
    
    avg_velocities2_x, bins = bin_velocities_1D(
        positions, 
        velocities**2, 
        box, 
        bin_size=bin_size, 
        axis=1
        ) # size (Nframes, Ny, 3)
    
    print(avg_velocities_x.shape, avg_velocities2_x.shape)

    # avg_velocities_x = avg_velocity_1D(binned_velocities, axis=1) # size (Nframes, Ny, 3)
    # plot_velocity_field_1D(avg_velocities_x[100], bins=bins, idx=0)  # Plot Vx vs y
    
    # ani = plot_velocity_field_1D_time(avg_velocities_x[150:, :, 0], bins=bins, idx=0)  # Plot Vx vs y over time
    
    time_avg_velocities_x = np.mean(avg_velocities_x[160:], axis=0)
    time_avg_velocities2_x = np.mean(avg_velocities2_x[160:] - avg_velocities_x[160:]**2, axis=0)
    
    plot_velocity_field_1D(
        time_avg_velocities_x, 
        bins=bins, 
        idx=0, 
        save=True, 
        loglog=False, 
        var=False
        )  # Plot time-averaged Vx vs y
    
    plot_velocity_field_1D(
        time_avg_velocities2_x, 
        bins=bins, 
        idx=0, 
        save=True, 
        loglog=False,
        var=True
        )  # Plot time-averaged Vx^2 vs y on log-log scale
    
    plot_velocity_field_1D_time_slices(
        avg_velocities_x, 
        bins=bins, 
        slices=15, 
        start_from=160,
        idx=0,
        save=True,
    )
    plt.show()
    
if __name__ == "__main__":
    main()