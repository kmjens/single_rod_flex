## Simulate single flexicles with single rod inside
# Try to flatten spherocylinders to explore phase space accoring to Haichao Wu experiments

## Boilerplate:
import argparse
import datetime
import freud
import hoomd
import matplotlib
import os
import random
import signac

import gsd.hoomd

import numpy as np
import pandas as pd
import pyvista as pv

from utility import print_state, print_state_to_file

def Run_implementation(job, communicator):
    
    #############################################
    ## Statepoints from init file:
    #############################################
    
    kT      = job.cached_statepoint['kT']
    R       = job.cached_statepoint['R']
    N_mesh  = int(job.cached_statepoint['N_mesh'])
    dt      = job.cached_statepoint['dt']
    v0      = job.cached_statepoint['v0']
    runtime = job.cached_statepoint['runtime']
    simseed = job.cached_statepoint['seed']
    L       = R * 5 # box size
    gamma   = job.cached_statepoint['gamma']
    gamma_r = [gamma/3.0,gamma/3.0,gamma/3.0]
    mesh_gamma = 5
    
    k_bend  = job.cached_statepoint['k_bend']
    k_bond  = 4 * k_bend # based on general ratio lipid bilayers have
    k_area  = 1000
    
    # Adding active particles:
    N_active    = int(job.cached_statepoint['N_active'])
    ratio_len   = job.cached_statepoint['ratio_len'] # defines sigma
    num_filler  = job.cached_statepoint['num_filler']
    filler_diam_ratio = job.cached_statepoint['filler_diam_ratio']
    gravity_strength  = np.abs(job.cached_statepoint['gravity_strength'])
    N_particles = N_mesh + N_active
    
    # Adding torque:
    rand_orient     = job.cached_statepoint['rand_orient']
    active_angle    = job.cached_statepoint['active_angle']
    torque_mag      = job.cached_statepoint['torque_mag']
    ideal_buffer    = 0.5
    
    mesh_sigma   = 1
    sigma        = (2/3) * R / ratio_len
    filler_sigma = filler_diam_ratio * sigma
    rod_size     = sigma * 3
    
    #############################################
    ## Set up sim:
    #############################################
    
    device = hoomd.device.CPU(num_cpu_threads=communicator.num_ranks)
    sim = hoomd.Simulation(device=device)
    sim.seed = simseed

    #############################################
    ## Particle placement:
    #############################################
    
    # Mesh:
    x,y,z = fibonacci_sphere(num_pts=N_mesh, R=R)
    mesh_position = np.column_stack((x,y,z))
    mesh_orient = np.zeros((N_mesh,4),dtype=float)
    mesh_orient[:,0] = 1
    mesh_typeid = [0] * N_mesh
    mesh_diam = [0.333 * mesh_sigma] * N_mesh
    mesh_MoI = np.zeros((N_mesh,3),dtype=float)
    
    # A Beads:
    A_position = [np.array([0,0,0])]
    A_orient = np.zeros((len(A_position),4),dtype=float)
    A_orient[:,0] = 1
    A_typeid = np.ones(len(A_position),dtype=int)
    A_diam = np.ones(len(A_position),dtype=int)
    A_MoI = np.zeros((len(A_position),3),dtype=float)
    A_MoI[:,0] = 0
    A_MoI[:,1] = 1.0/12*5*(rod_size*sigma)**2
    A_MoI[:,2] = 1.0/12*5*(rod_size*sigma)**2
    
    # Consituents:


    #############################################
    ## Set up frame
    #############################################
    
    position = np.append(mesh_position,A_position,axis=0)
    orientation = np.append(mesh_orient,A_orient,axis=0)
    typeid = np.append(mesh_typeid,A_typeid,axis=0)
    diameter = np.append(mesh_diam,A_diam,axis=0)
    moment_inertia = np.append(mesh_MoI,A_MoI,axis=0)

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = position[0:N_particles]
    frame.particles.orientation = orientation[0:N_particles]
    frame.particles.typeid = typeid[0:N_particles]
    frame.particles.diameter = diameter[0:N_particles]
    frame.particles.moment_inertia = moment_inertia[0:N_particles]
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['mesh','A','A_const','A_filler']

    with gsd.hoomd.open(name=job.fn('initial.gsd'), mode='w') as f:
       f.append(frame)

    state = sim.create_state_from_gsd(filename=job.fn('initial.gsd'))
    f = gsd.hoomd.open(name=job.fn('initial.gsd'),mode='r')
    frame = f[0]

def Run(*jobs):
    """Execute actions on directories in parallel using HOOMD-blue."""
    processes_per_directory = os.environ['ACTION_PROCESSES_PER_DIRECTORY']
    communicator = hoomd.communicator.Communicator()
    Run_implementation(jobs[communicator.partition], communicator)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()
    
    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)
