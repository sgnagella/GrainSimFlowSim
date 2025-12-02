"""
    Script to unwrap trajecctories from 2D Brownian Dynamics simulations
    using HOOMD 
"""
import numpy as np
import gsd.hoomd

def get_unwrapped_positions_velocities(traj_file='trajectory.gsd'): 
    # reads the trajectory and gets the particle positions and velocities 
    # read the trajectory GSD file
    traj = gsd.hoomd.open(name=traj_file, mode='r')

    # see how many frames exist in the trajectory
    N_frames = len(traj)

    # get number of particles and box dimensions
    N_particles = traj[0].particles.N
    Lx = traj[0].configuration.box[0]
    Ly = traj[0].configuration.box[1]
    Lz = traj[0].configuration.box[2]
    boxvector = [Lx,Ly,Lz]
    
    # (naive double loop too slow for large data sets)
    unwrapped_traj = np.zeros((N_frames, N_particles, 3))  # for unwrapped positions (for 2D, z will be 0)
    
    for i in range(N_frames):
        print("Reading frame " + str(i+1) + " / " + str(N_frames))
        position = traj[i].particles.position
        image = traj[i].particles.image
        for j in range(N_particles):
            xyz = position[j] + boxvector*image[j]
            # unwrapped_traj[i, j, 0] = xyz[0]
            # unwrapped_traj[i, j, 1] = xyz[1]
            # unwrapped_traj[i, j, 2] = xyz[2]
            
            unwrapped_traj[i, j, :] = xyz  # store all three coordinates at once

    # save the unwrapped trajectory
    np.save('unwrapped_trajectory.npy', unwrapped_traj)
    print("Unwrapped trajectory saved to 'unwrapped_trajectory.npy'")
    return

