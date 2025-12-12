import os
import numpy as np
import gsd.hoomd

def is_float(string):
    """ True if given string is float else False"""
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def convert_string_to_float(string):
    """ Converts string to float if possible else returns string"""
    try:
        return float(string)
    except ValueError:
        return string


def read_dat_file(filename):
    print(f"Reading data from {filename}")
    data = []
    with open(filename, 'r') as f:
        d = f.readlines()[1:] # Skip header line
        num_lines = len(d)
        print(f"Number of lines in the file: {num_lines}")
        for idx, i in enumerate(d):
            print(f"Reading line {idx+1}/{num_lines}", end='\r')
            # if idx > 1: 
            #     exit(1)
            k = i.rstrip().split(' ')
            data.append(np.array([convert_string_to_float(i) if is_float(i) else np.nan for i in k][:3]))
            print(data[-1].shape)
            # print(data)

    print("\nFinished reading data.")
    print(data[-10:])
    return np.asarray(data, dtype=np.float32)

def read_dat_file_simple(filename, usecols=(0,1,2,-3, -2, -1)):
    """Loads the first 3 columns of a .dat file into a numpy array using numpy.loadtxt."""
    print(f"Reading data from {filename} (simple method)")
    data = np.loadtxt(filename, comments='#', usecols=usecols)
    print(f"Loaded data shape: {data.shape}")
    print(data[:10])
    print(data[-10:])
    return data

def write_gsd(filename, radii_file, Nparticles, box, dt, Nsteps, particle_groups=None):
    """Writes a GSD file with Nsteps frames of Nparticles in a box of given size.
    The particles are randomly placed in the box and have random velocities.
    """
    traj = read_dat_file_simple(filename)
    
    radii = read_dat_file_simple(radii_file, usecols=(0,))
    radii_unique = np.unique(radii)
    print(f"Unique radii: {radii_unique}")
    
    if particle_groups is None:
        particle_groups = {"A": range(0, Nparticles)}
    else:
        # Assign different particle groups for each radii if they are not already assigned
        for radius in radii_unique:
            indices = np.where(radii == radius)[0]
            particle_groups[f"R{radius}"] = indices.tolist()
        
    # typeids = []
    # for t in particle_groups.keys():
    #     typeids += list(particle_groups[t])
    
    typeids = np.zeros_like(radii, dtype=np.int32)
    for idx, (group_name, indices) in enumerate(particle_groups.items()):
        typeids[indices] = idx
        
    types = list(particle_groups.keys())

    with gsd.hoomd.open(name="traj.gsd", mode='w') as f:
        for step in range(Nsteps):
            frameidx = int(Nparticles * step)
            if frameidx + Nparticles > traj.shape[0]:
                print(f"Reached end of trajectory data at step {step}. Stopping.")
                break
            snap = gsd.hoomd.Frame()
            snap.particles.N = Nparticles
            snap.configuration.box = [box[0], box[1], box[2], 0, 0, 0]
            snap.particles.position = traj[frameidx:frameidx+Nparticles, :3]
            snap.particles.velocity = traj[frameidx:frameidx+Nparticles, 3:6]
            snap.particles.image = traj[frameidx:frameidx+Nparticles, -3:]
            # if np.any( snap.particles.image == -1 ):
            #     print("Image flag contains -1")
                
            snap.particles.diameter = 2 * radii[:Nparticles]
            # snap.particles.velocity = traj[frameidx:frameidx+int(Nparticles), 3:6]
            snap.particles.typeid = np.array(typeids, dtype=np.int32)
            snap.particles.types = types
            snap.configuration.step = step
            snap.configuration.dimensions = 3
            snap.configuration.dt = dt
            f.append(snap)
            
        f.flush()
        
    
    return

if __name__ == "__main__":
    dat_file ="particles3.dat"; "particles2.dat"; "particles.dat"
    radii_file = "input/radii.dat"
    box = [38.01401138305664, 42.782547, 0.0]
    Nparticles = 286
    platePaticles = 15
    dt = 0.0008; 0.001
    Nsteps = int(200*1584/1584)
    # Nsteps = int(312500/1250)
    # Nsteps = int(100000/1250)
    particle_groups = {"A": [0]*(platePaticles), "B": [1]*(platePaticles), "C": [2]*(Nparticles - 2*platePaticles)} ;  {"A": [0]*(Nparticles//2), "B": [1]*(Nparticles//2)}; {"A":[0]*Nparticles}; 
    write_gsd(dat_file, radii_file, Nparticles, box, dt, Nsteps, particle_groups)