## Boilerplate
import argparse
import datetime
import freud
import hoomd
import matplotlib
import os
import json
import random
import signac

import gsd.hoomd

import numpy as np
import pandas as pd
import pyvista as pv

from utility import *


def Setup_implementation(job, communicator):

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
    visc    = job.cached_statepoint['visc']
    gamma   = 6 * np.pi * sigma
    gamma_r = [gamma/3.0, gamma/3.0, gamma/3.0]

    mesh_gamma  = 6 * np.pi * mesh_sigma
    mesh_gamma_r = [mesh_gamma/3.0, mesh_gamma/3.0, mesh_gamma/3.0]
    
    # Particle num and mesh scaling
    N_active       = int(job.cached_statepoint['N_active'])
    num_flattener  = job.cached_statepoint['num_flattener'] # num on one active particle
    N_flattener    = 2 * num_flattener * N_active # including all active particles
    num_beads      = int(job.cached_statepoint['num_beads']) #number of beads including the center particle in rigid body
    num_const_beads = int(num_beads - 1) # neglecting middle particle
    N_bead   = num_const_beads * N_active
    
    rod_length   = SP.aspect_rat * sigma
    bead_spacing = (SP.aspect_rat - 1) / (num_const_beads - 1)
    sphero_vol   = (sigma ** 3) * (3 * rod_length - 1) / 4 # approx as spherocylinder
    cylinder_vol = rod_length * np.pi * (sigma / 2) ** 2
    vol_diff     = cylinder_vol - sphero_vol
    R            = (SP.freedom_rat * rod_length) / 2
    L            = R * 5 # box size

    TriArea = SP.TriArea
    num_tri = int(4 * np.pi * R**2 / TriArea)
    N_mesh = num_tri + 2
    N_particles = N_active + N_bead + N_flattener
    
    # Active, buoyant, and gravitational forces
    fA = SP.fA
    Pe = sigma * fA / SP.kT
    print('Peclet Number = ', Pe)

    BG = BuoyancyAndGravity(R, N_mesh, cylinder_vol)
    F_const_mesh    = BG.F_const_mesh
    F_const_rod     = BG.F_const_rod
    mass_rod        = BG.mass_rod
    mass_mesh_bead  = BG.flex_mass / N_mesh
    print('mass_rod = ', mass_rod)
    print('mass_mesh_bead = ', mass_mesh_bead)

    with open(job.fn('Setup.out.in_progress'), 'w') as file:
        file.write('Initializing sim seed: ' + str(SP.simseed) + '\n')

    with open(job.fn('Initialization.out.in_progress'), 'w') as file:
        file.write('Initializing sim seed: ' + str(SP.simseed) + '\n')

    #############################################
    ## Set up simulation object
    #############################################

    #device = hoomd.device.CPU(num_cpu_threads=communicator.num_ranks)
    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device)
    sim.seed = SP.simseed


    #############################################
    ## Particle placement
    #############################################


    # Rod position:
    A_position = [np.array([0,0,0])] * N_active
    A_orient = np.zeros((len(A_position),4),dtype=float)
    A_orient[:,0] = 1
    A_typeid = np.ones(len(A_position),dtype=int)
    A_diam = [sigma] * N_active
    A_mass = [mass_rod] * N_active
    A_MoI = np.zeros((len(A_position),3),dtype=float) # kg*m2 -- need to check unit convs?
    A_MoI[:,0] = 0
    A_MoI[:,1] = 1.0 / 12 * 5 * (rod_length * sigma)**2
    A_MoI[:,2] = 1.0 / 12 * 5 * (rod_length * sigma)**2


    #############################################
    ## Set up frame
    #############################################

    position = A_position
    orientation = A_orient
    typeid = A_typeid
    diameter = A_diam
    moment_inertia = A_MoI
    mass = A_mass

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_active
    frame.particles.mass = mass 
    frame.particles.position = position[0:frame.particles.N]
    frame.particles.orientation = orientation[0:frame.particles.N]
    frame.particles.typeid = typeid[0:frame.particles.N]
    frame.particles.diameter = diameter[0:frame.particles.N]
    frame.particles.moment_inertia = moment_inertia[0:frame.particles.N]
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['A','A_const','A_flattener']

    with gsd.hoomd.open(name=job.fn('initial.gsd'), mode='w') as f:
       f.append(frame)

    state = sim.create_state_from_gsd(filename=job.fn('initial.gsd')) 
    f = gsd.hoomd.open(name=job.fn('initial.gsd'),mode='r')
    frame = f[0]


    #############################################
    # Construct rod rigid bodies
    #############################################
    # Initialize lists
    bead_type_list = []
    bead_pos_list = []
    bead_orient_list = []
    bead_diam = []
    bead_mass = []

    flattener_type_list = []
    flattener_pos_list = []
    flattener_orient_list = []
    flattener_diam = []
    flattener_mass = []

    # Loop over all active rods
    for i in range(N_active):
        # Beads along the rod (excluding central particle)
        bead_type_list.extend(['A_const'] * num_const_beads)
        bead_pos_list.extend(get_bead_pos(rod_length, sigma, num_const_beads, rod_index=i))
        bead_orient_list.extend([(1, 0, 0, 0)] * num_const_beads)
        bead_diam.extend([sigma] * num_const_beads)
        bead_mass.extend([0] * num_const_beads)
        
        # Flattener particles for this rod
        flattener_type_list.extend(['A_flattener'] * (2 * num_flattener))
        flattener_pos_list.extend(get_flattener_pos(num_flattener, sigma, flattener_sigma, rod_length, rod_index=i))
        flattener_orient_list.extend([(1, 0, 0, 0)] * (2 * num_flattener))
        flattener_diam.extend([flattener_sigma] * (2 * num_flattener))
        flattener_mass.extend([0] * (2 * num_flattener))

    # Combine constituent particles
    const_type_list = bead_type_list + flattener_type_list
    const_pos_list = bead_pos_list + flattener_pos_list
    const_orient_list = bead_orient_list + flattener_orient_list

    assert len(const_type_list) == len(const_pos_list) == len(const_orient_list)

    # Create rigid body objects
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": const_type_list,
        "positions":         const_pos_list,
        "orientations":      const_orient_list
    }
    rigid.create_bodies(sim.state)

    # Combine with central particle arrays (already in `diameter`, `mass`)
    diameter = np.append(diameter, bead_diam + flattener_diam)
    mass = np.append(mass, bead_mass + flattener_mass)

    # Create initial GSD
    snapshot = sim.state.get_snapshot()
    frame = gsd.hoomd.Frame()
    frame.particles.N = len(snapshot.particles.position)
    frame.particles.mass = mass
    frame.particles.position = snapshot.particles.position
    frame.particles.orientation = snapshot.particles.orientation
    frame.particles.moment_inertia = snapshot.particles.moment_inertia
    frame.particles.typeid = snapshot.particles.typeid
    frame.particles.diameter = diameter
    frame.particles.types = snapshot.particles.types
    frame.particles.body = snapshot.particles.body  # needed for rigid bodies
    frame.configuration.box = snapshot.configuration.box

    with gsd.hoomd.open(name=job.fn('initial_wRigid.gsd'), mode='w') as f:
        f.append(frame)

    # Reload simulation state
    sim = hoomd.Simulation(device=device)
    sim.seed = SP.simseed
    state = sim.create_state_from_gsd(filename=job.fn('initial_wRigid.gsd'))
    snap = sim.state.get_snapshot()    

    #############################################
    ## Set up filters and integrator
    #############################################

    filter_all  = hoomd.filter.All()
    filter_rigid = hoomd.filter.Rigid(("center","free"))

    integrator = hoomd.md.Integrator(
            dt=SP.dt,
            rigid=rigid,
            integrate_rotational_dof=True)
    sim.operations.integrator = integrator

    langevin = hoomd.md.methods.Langevin(filter=filter_rigid, kT=SP.kT)
    langevin.gamma.default = gamma
    langevin.gamma_r.default = gamma_r
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
    gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(10000)), #int(2000)
                               filename=job.fn('Initialize.gsd'),
                               logger=logger, mode='wb',
                               dynamic=['momentum','property','attribute','attribute/particles/diameter'],
                               filter=filter_all)
    gsd_oper.write_diameter = True
    sim.operations += gsd_oper
    sim.state.thermalize_particle_momenta(filter=filter_all, kT=SP.kT)
    
    # Initialize
    sim.run(0)
    print('Successfully ran for 0 timestep.\n')

    snap = sim.state.get_snapshot()
    print('particle size:', snap.particles.diameter)

    sim.run(5000)

    # Add gravity:
    print('\nAdding gravity...')
    gravity = hoomd.md.force.Constant(filter=hoomd.filter.All())
    gravity.constant_force['A'] = (0,0,F_const_rod)
    gravity.constant_force['A_const','A_flattener'] = (0,0,0)
    gravity.constant_torque['A','A_const','A_flattener'] = (0,0,0)

    integrator.forces.append(gravity)


    #############################################
    ## Run the simulation
    #############################################
    print('\nFinish equilibrating simulation...')
    while sim.timestep < (SP.equiltime - 1000):
        sim.run(1000)
        gsd_oper.flush()
        print('step: ', sim.timestep)

    # Add rod active force:
    print('\nAdding active force...')
    active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
    active.active_force['A'] = (SP.fA,0,0)
    active.active_torque['A'] = (0,0,0)
    integrator.forces.append(active)

    sim.run(999)
    
    final_timestep = sim.timestep+1
    final_frame_writer = hoomd.write.GSD(trigger=hoomd.trigger.On(final_timestep),
                                        filename=job.fn("final_init_frame.gsd"),
                                        logger=logger, mode='wb',
                                        dynamic=['momentum','property','attribute','attribute/particles/diameter'],
                                        filter=filter_all)

    final_frame_writer.write_diameter = True
    sim.operations += final_frame_writer
    
    sim.run(1)
    gsd_oper.flush()
    print('step: ', sim.timestep)

    print('\nCurrent state:')
    print_state(sigma, flattener_sigma, N_particles, num_flattener, N_active, num_beads, bead_spacing, SP.aspect_rat, SP.freedom_rat, Pe, SP.torque_mag, mass_rod, F_const_rod, job)

    
    os.rename(job.fn('Initialization.out.in_progress'), job.fn('Initilization.out'))

    print('Initialization complete.')
    '''
    gsd_run = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(10000)), #int(2000)
                               filename=job.fn('Run.gsd'),
                               logger=logger, mode='wb',
                               dynamic=['momentum','property','attribute','attribute/particles/diameter'],
                               filter=filter_all)
    gsd_demo.write_diameter = True
    sim.operations += gsd_run
    
    print('Running for demo...')
    while sim.timestep < SP.runtime:
        sim.run(10000)
        gsd_oper.flush()
        print('step: ', sim.timestep)
    '''


    os.rename(job.fn('Setup.out.in_progress'), job.fn('Setup.out'))
    print('Setup complete.')


def Setup(*jobs):
    processes_per_directory = os.environ['ACTION_PROCESSES_PER_DIRECTORY']
    communicator = hoomd.communicator.Communicator()
    Setup_implementation(jobs[communicator.partition], communicator)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)
