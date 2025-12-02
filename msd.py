# -*- coding: utf-8 -*-
"""
    Script to obtain the diffusivity from trajectories
    computed by 2D Brownian Dynamics simulation
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from unwrap_traj import get_unwrapped_positions_velocities

# Load the unwrapped trajectory
get_unwrapped_positions_velocities('traj.gsd')
unwrapped_traj = np.load('unwrapped_trajectory.npy')
print('Unwrapped trajectory shape: ', unwrapped_traj.shape)

# Plot the trajectory of the first particle as a check
plt.figure()
plt.plot(unwrapped_traj[:, 0, 0] - unwrapped_traj[0,0,0], marker='.', color='blue', label='x')
plt.plot(unwrapped_traj[:, 0, 1] - unwrapped_traj[0,0,1], marker='.', color='orange', label='y')
plt.plot(unwrapped_traj[:, 0, 2] - unwrapped_traj[0,0,2], marker='.', color='green', label='z')
plt.xlabel('Frame')
N_frames = unwrapped_traj.shape[0]
N_particles = unwrapped_traj.shape[1]
print('Number of frames: ', N_frames)
print('Number of particles: ', N_particles)

def getMSDTensor(r):
    # Uses window averaging 
    shifts = np.arange(len(r))
    msds = np.zeros((shifts.size,4)) # columns: <dx^2>, <dy^2>, <dx*dy>, <dz^2>
    
    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        diffs_x = diffs[:,0]
        diffs_y = diffs[:,1]
        diffs_z = diffs[:,2]
        sqdist_xx = diffs_x*diffs_x
        sqdist_yy = diffs_y*diffs_y
        sqdist_xy = diffs_x*diffs_y
        sqdist_zz = diffs_z*diffs_z
        msds[i,0] = sqdist_xx.mean()
        msds[i,1] = sqdist_yy.mean()
        msds[i,2] = sqdist_xy.mean()
        msds[i,3] = sqdist_zz.mean()
    
    return msds


def compute_msd_tensor_unwrapped(traj): 
    """From the unwrapped trajectory, compute the MSD tensor
    """

    msd = np.zeros((N_frames, 4))  # columns: <dx^2>, <dy^2>, <dx*dy>, <dz^2>
    for ii in range(N_particles):
        print(f"Analyzing particle {ii+1}/{N_particles}")
        r = traj[:,ii,:]        # Trajectory of shape (N_frames, N_particles, 4),
        msd += getMSDTensor(r)  # Compute MSD from trajectory of one particle at a time (N_frames, 4)

    msd /= N_particles
    return msd

msd = compute_msd_tensor_unwrapped(unwrapped_traj)
flag = True
times = np.arange(N_frames)
msdovr2 = msd/2
Dxx = np.polyfit(times, msdovr2[:,0], 1)[0]
Dyy = np.polyfit(times, msdovr2[:,1], 1)[0]
Dxy = np.polyfit(times, msdovr2[:,2], 1)[0]
Dzz = np.polyfit(times, msdovr2[:,3], 1)[0]
print(f"Dxx = {Dxx}, Dyy = {Dyy}, Dxy = {Dxy}, Dzz = {Dzz}")
plt.figure()
plt.plot(times, msdovr2)
plt.xlabel('Time')
plt.ylabel('MSD')
plt.show()