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

from utility import *


def Run_implementation(job, communicator):

    #############################################
    ## Statepoints from init file
    #############################################
    print('\nStarting simulation.')
    print('job: ', job)
    print('statepoints:\n', job.sp, '\n\n')
    
    SP = JobParser(job)

    # Calculate sigmas
    sigma        = 1
    mesh_sigma   = SP.mesh_sigma_rat * sigma
    flattener_sigma = SP.flattener_sigma_rat * sigma

    # Calculate gammas
    gamma   = 6 * np.pi * sigma
    gamma_r = [gamma/3.0,gamma/3.0,gamma/3.0]
    mesh_gamma  = 5

    # Calculate particle and mesh scaling:
    rod_length   = SP.aspect_rat * sigma
    bead_spacing = (rod_length / 2) - (sigma / 2)
    sphero_vol   = (sigma ** 3) * (3 * rod_length - 1) / 4 # approx as spherocylinder
    cylinder_vol = 2 * np.pi**2 * (sigma / 2)**3
    vol_diff     = cylinder_vol - sphero_vol
    R            = (SP.freedom_rat * rod_length) / 2
    L            = R * 5 # box size


    N_active    = int(job.cached_statepoint['N_active'])
    num_flattener  = job.cached_statepoint['num_flattener'] # num on one active particle
    N_flattener    = 2 * num_flattener * N_active # including all active particles
    num_beads   = int(job.cached_statepoint['num_beads']) #number of beads including the center particle in rigid body
    num_const_beads = int(num_beads - 1) # neglecting middle particle
    N_bead      = num_const_beads * N_active
    N_particles = N_active + N_bead + N_flattener

    TriArea = SP.TriArea
    num_tri = int(4 * np.pi * R**2 / TriArea)
    N_mesh = num_tri + 2

    # Buoyant force and gravitational force
    BG = BuoyancyAndGravity(R, N_mesh, cylinder_vol)
    F_const_rod     = BG.F_const_rod
    mass_rod        = BG.mass_rod

    with open(job.fn('Run.out.in_progress'), 'w') as file:
        file.write('Initializing sim seed: ' + str(SP.simseed) + '\n')
    

    #############################################
    ## Set up simulation object
    #############################################

    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device)
    sim.seed = SP.simseed

    state = sim.create_state_from_gsd(filename=job.fn('final_init_frame.gsd'))
    bond_types = sim.state.bond_types
    print("Pre-existing bonds: ",bond_types)


    f = gsd.hoomd.open(name=job.fn('initial.gsd'),mode='r')
    frame = f[0]

    rigid = hoomd.md.constrain.Rigid()

    #############################################
    ## Set up filters and integrator
    #############################################

    filter_all  = hoomd.filter.All()
    filter_free = hoomd.filter.Rigid(("center","free"))

    integrator = hoomd.md.Integrator(
            dt=SP.dt,
            rigid=rigid,
            integrate_rotational_dof=True)
    sim.operations.integrator = integrator

    langevin = hoomd.md.methods.Langevin(filter=filter_free, kT=SP.kT)
    langevin.gamma.default = gamma
    langevin.gamma_r.default = [gamma,gamma,gamma]
    integrator.methods.append(langevin)


    #############################################
    ## Add potentials
    #############################################

    ideal_buffer = 0.5
    cell = hoomd.md.nlist.Cell(buffer=ideal_buffer, exclusions=['body'])
    
    # Add wall:
    wall = [hoomd.wall.Plane(origin=(0, 0, -R-sigma), normal=(0, 0, 1))]
    wlj = hoomd.md.external.wall.LJ(walls=wall)
    wlj.params[['A','A_const','A_flattener']] = {"epsilon": 0.0, "sigma": 1.0, "r_cut": 0.}
    integrator.forces.append(wlj)


    #############################################
    ## Initialize the simulation
    #############################################

    snap = sim.state.get_snapshot()
    print('after potentials:', snap.particles.diameter)

    # GSD logger:
    logger = hoomd.logging.Logger(['particle','constraint'])
    gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(500)), #int(2000)
                               filename=job.fn('active.gsd'),
                               logger=logger, mode='wb',
                               dynamic=['momentum','property','attribute/particles/diameter'],
                               filter=filter_all)
    gsd_oper.write_diameter = True
    sim.operations += gsd_oper
    

    # Add gravity:
    print('\nAdding gravity...')
    mass_mesh_particle = 1
    gravity = hoomd.md.force.Constant(filter=hoomd.filter.All())
    gravity.constant_force['A'] = (0,0,F_const_rod)
    gravity.constant_force['A_const','A_flattener'] = (0,0,0)
    gravity.constant_torque['A','A_const','A_flattener'] = (0,0,0)

    integrator.forces.append(gravity)


    #############################################
    ## Run the simulation
    #############################################
    
    # Add rod active force:
    print('\nAdding active force...')
    active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
    active.active_force['A'] = (SP.fA,0,0)
    active.active_torque['A'] = (0,0,0)
    integrator.forces.append(active)
    
    fA = SP.fA
    Pe = sigma * fA / SP.kT


    print('\nCurrent state:')
    print_state(sigma, flattener_sigma, N_particles, num_flattener, N_active, num_beads, bead_spacing, SP.aspect_rat, SP.freedom_rat, Pe, SP.torque_mag,  mass_rod, F_const_rod, job)
    
    # Initialize
    sim.run(0)
    print('Successfully ran for 0 timestep.\n')

    
    print('\nRunning simulation...')
    while sim.timestep < SP.runtime:
        sim.run(10000)
        gsd_oper.flush()
        print('step: ', sim.timestep)


    os.rename(job.fn('Run.out.in_progress'), job.fn('Run.out'))

    print('Simulation complete.')

    snap = sim.state.get_snapshot()

def Run(*jobs):
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
