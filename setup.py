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
    gamma   = job.cached_statepoint['gamma']
    gamma_r = [gamma/3.0,gamma/3.0,gamma/3.0]
    mesh_gamma  = 5

    # Calculate particle and mesh scaling:
    rod_length   = SP.aspect_rat * sigma
    bead_spacing = (aspect_rat - 1) / (num_const_beads - 1)
    sphero_vol   = (sigma ** 3) * (3 * rod_length - 1) / 4 # approx as spherocylinder
    cylinder_vol = rod_length * np.pi * (sigma / 2) ** 2
    vol_diff     = cylinder_vol - sphero_vol
    R            = (SP.freedom_rat * rod_length) / 2
    L            = R * 5 # box size

    # Calculate particle numbers:
    TriArea = SP.TriArea
    num_tri = int(4 * np.pi * R**2 / TriArea)
    N_mesh = num_tri + 2
    
    #N_mesh      = int(np.ceil(4 * np.pi * R**2 * 0.8))
    N_active    = int(job.cached_statepoint['N_active'])
    num_flattener  = job.cached_statepoint['num_flattener'] # num on one active particle
    N_flattener    = 2 * num_flattener * N_active # including all active particles
    num_beads   = int(job.cached_statepoint['num_beads']) #number of beads including the center particle in rigid body
    num_const_beads = int(num_beads - 1) # neglecting middle particle
    N_bead      = num_const_beads * N_active
    N_particles = N_mesh + N_active + N_bead + N_flattener

    # Buoyant force and gravitational force
    BG = BuoyancyAndGravity(R, N_mesh, cylinder_vol)
    F_const_mesh    = BG.F_const_mesh
    F_const_rod     = BG.F_const_rod
    mass_rod        = BG.mass_rod
    mass_mesh_bead  = BG.flex_mass / N_mesh
    print('mass_rod = ', mass_rod)
    print('mass_mesh_bead = ', mass_mesh_bead)

    with open(job.fn('Setup.out.in_progress'), 'w') as file:
        file.write('Initializing sim seed: ' + str(SP.simseed) + '\n')


    #############################################
    ## Set up simulation object
    #############################################

    device = hoomd.device.CPU(num_cpu_threads=communicator.num_ranks)
    sim = hoomd.Simulation(device=device)
    sim.seed = SP.simseed


    #############################################
    ## Particle placement
    #############################################

    # Mesh:
    x,y,z = fibonacci_sphere(num_pts=N_mesh, R=R)
    mesh_position = np.column_stack((x,y,z))
    mesh_orient = np.zeros((N_mesh,4),dtype=float)
    mesh_orient[:,0] = 1
    mesh_typeid = [0] * N_mesh
    mesh_diam = [mesh_sigma] * N_mesh
    mesh_MoI = np.zeros((N_mesh,3),dtype=float)
    #mesh_mass = [1] * N_mesh
    mesh_mass = [mass_mesh_bead] * N_mesh

    mesh = pv.PolyData(mesh_position)
    faces = mesh.delaunay_3d().extract_geometry().faces.reshape((-1, 4))
    triangle_points = []
    for face in faces:
        triangle_points.append(face[1:])
    triangle_tags = np.vstack((triangle_points))

    mesh_obj = hoomd.mesh.Mesh()
    mesh_obj.types = ["mesh"]
    mesh_obj.triangulation = dict(type_ids = [0] * len(triangle_tags),
          triangles = triangle_tags)

    # Rod position:
    A_position = [np.array([0,0,0])]
    A_orient = np.zeros((len(A_position),4),dtype=float)
    A_orient[:,0] = 1
    A_typeid = np.ones(len(A_position),dtype=int)
    A_diam = [sigma] * N_active
    #A_mass = [mass_rod] * N_active
    A_mass = [1] * N_active
    A_MoI = np.zeros((len(A_position),3),dtype=float) # kg*m2 -- need to check unit convs?
    A_MoI[:,0] = 0
    A_MoI[:,1] = 1.0 / 12 * 5 * (rod_length * sigma)**2
    A_MoI[:,2] = 1.0 / 12 * 5 * (rod_length * sigma)**2


    #############################################
    ## Set up frame
    #############################################

    position = np.append(mesh_position,A_position,axis=0)
    orientation = np.append(mesh_orient,A_orient,axis=0)
    typeid = np.append(mesh_typeid,A_typeid,axis=0)
    diameter = np.append(mesh_diam,A_diam,axis=0)
    moment_inertia = np.append(mesh_MoI,A_MoI,axis=0)
    mass = np.append(mesh_mass, A_mass, axis=0)

    frame = gsd.hoomd.Frame()
    frame.particles.N = N_mesh + N_active
    frame.particles.mass = mass 
    frame.particles.position = position[0:frame.particles.N]
    frame.particles.orientation = orientation[0:frame.particles.N]
    frame.particles.typeid = typeid[0:frame.particles.N]
    frame.particles.diameter = diameter[0:frame.particles.N]
    frame.particles.moment_inertia = moment_inertia[0:frame.particles.N]
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['mesh','A','A_const','A_flattener']

    with gsd.hoomd.open(name=job.fn('initial.gsd'), mode='w') as f:
       f.append(frame)

    state = sim.create_state_from_gsd(filename=job.fn('initial.gsd')) 
    f = gsd.hoomd.open(name=job.fn('initial.gsd'),mode='r')
    frame = f[0]


    #############################################
    # Construct rod rigid bodies
    #############################################

    bead_type_list = ['A_const'] * N_bead
    bead_pos_list = get_bead_pos(rod_length, sigma, num_const_beads)
    bead_orient_list = [(1,0,0,0)] * N_bead
    bead_diam = [sigma] * N_bead
    bead_mass = [0] * N_bead

    flattener_type_list = ['A_flattener'] * (2 * num_flattener)
    flattener_pos_list = get_flattener_pos(num_flattener, sigma, flattener_sigma, rod_length)
    flattener_orient_list = [(1,0,0,0)] * (2 * num_flattener)
    flattener_diam = [flattener_sigma] * N_flattener
    flattener_mass = [0] * N_flattener

    const_type_list = bead_type_list + flattener_type_list
    const_pos_list = bead_pos_list + flattener_pos_list
    const_orient_list = bead_orient_list +flattener_orient_list

    assert len(const_type_list) == len(const_pos_list) == len(const_orient_list)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": const_type_list,
        "positions":         const_pos_list,
        "orientations":      const_orient_list
    }
    rigid.create_bodies(sim.state)

    diameter = np.append(diameter, bead_diam, axis=0)
    diameter = np.append(diameter, flattener_diam, axis=0)

    mass = np.append(mass, bead_mass, axis=0)
    mass = np.append(mass, flattener_mass, axis=0)
    print('check diam: ', diameter)
    print('check mass: ', mass)
    
    # Create initial gsd
    snapshot = sim.state.get_snapshot()
    frame = gsd.hoomd.Frame()
    frame.particles.N = len(snapshot.particles.position)
    frame.particles.mass = mass
    frame.particles.position = snapshot.particles.position
    frame.particles.orientation = snapshot.particles.orientation
    frame.particles.typeid = snapshot.particles.typeid
    frame.particles.diameter = diameter
    frame.particles.types = snapshot.particles.types
    frame.particles.body = snapshot.particles.body # need to add this line to have rigid body diams
    frame.configuration.box = snapshot.configuration.box

    with gsd.hoomd.open(name=job.fn('initial_wRigid.gsd'), mode='w') as f:
        f.append(frame)
    
    sim = hoomd.Simulation(device=device)
    sim.seed = SP.simseed
    state = sim.create_state_from_gsd(filename=job.fn('initial_wRigid.gsd'))

    snap = sim.state.get_snapshot()
    print('diam after initial gsd:', snap.particles.diameter)
    print('mass after initial gsd:', snap.particles.mass)

    #############################################
    ## Set up filters and integrator
    #############################################

    filter_all  = hoomd.filter.All()
    filter_mesh = hoomd.filter.Type(['mesh'])
    filter_rigid = hoomd.filter.Rigid(("center","free"))

    integrator = hoomd.md.Integrator(
            dt=SP.dt,
            rigid=rigid,
            integrate_rotational_dof=True)
    sim.operations.integrator = integrator

    langevin = hoomd.md.methods.Langevin(filter=filter_rigid, kT=SP.kT)
    langevin.gamma.default = gamma
    langevin.gamma_r.default = [gamma,gamma,gamma]
    integrator.methods.append(langevin)

    #############################################
    ## Add potentials
    #############################################

    ideal_buffer = 0.5
    cell = hoomd.md.nlist.Cell(buffer=ideal_buffer, exclusions=['meshbond','body'])

    # Expanded LJ:
    ExpLJ = hoomd.md.pair.ExpandedLJ(nlist=cell, mode="shift", default_r_cut=0)

    unit_sigma = sigma
    deltas = np.ones((3,3))
    sigmas = [sigma, mesh_sigma, flattener_sigma]
    for i in range(len(sigmas)):
        for j in range(len(sigmas)):
            deltas[i][j] = (sigmas[i] + sigmas[j])/2 - unit_sigma

    ExpLJ.params.default = dict(epsilon=0, sigma=unit_sigma, delta=0)
    ExpLJ.params[('mesh','mesh')] = dict(epsilon=1,
                                         sigma=unit_sigma,
                                         delta=deltas[1][1])
    ExpLJ.params[('mesh','A'),
                 ('mesh','A_const')] = dict(epsilon=1,
                                            sigma = unit_sigma,
                                            delta=deltas[0][1])
    ExpLJ.params[('A_flattener','A'),
                 ('A_flattener','A_const')] = dict(epsilon=1,
                                                sigma=unit_sigma,
                                                delta=deltas[0][2])
    ExpLJ.params[('A_flattener','mesh')] = dict(epsilon=1,
                                             sigma=unit_sigma,
                                             delta=deltas[1][2])
    ExpLJ.params[('A_flattener','A_flattener')] = dict(epsilon=0,
                                                 sigma=unit_sigma,
                                                 delta=deltas[2][2])

    ExpLJ.r_cut[('A','A'),
                ('A_const','A'),
                ('A_const','A_const')] = 2**(1.0/6.) * (unit_sigma) + deltas[0][0]
    ExpLJ.r_cut[('mesh','A'),
                ('mesh','A_const')] = 2**(1.0/6)*(unit_sigma)+deltas[0][1]
    ExpLJ.r_cut[('mesh','mesh')] = 2**(1.0/6.)*(unit_sigma)+deltas[1][1]
    ExpLJ.r_cut[('A_flattener','A'),
                ('A_flattener','A_const')] = 2**(1.0/6)*unit_sigma + deltas[0][2]
    ExpLJ.r_cut[('A_flattener','mesh')] = 2**(1.0/6)*(unit_sigma) + deltas[1][2]
    ExpLJ.r_cut[('A_flattener','A_flattener')] = 0

    integrator.forces.append(ExpLJ)

    # Apply tethering potential to mesh:
    l_min, l_c1, l_c0, l_max = get_tether_params(frame, triangle_tags)

    mesh_bond_potential = hoomd.md.mesh.bond.Tether(mesh_obj)
    mesh_bond_potential.params["mesh"] = dict(
            k_b=SP.k_bond,
            l_min=l_min,
            l_c1=l_c1,
            l_c0=l_c0,
            l_max=l_max)
    integrator.forces.append(mesh_bond_potential)

    # Helfrich bending potential:
    helfrich_potential = hoomd.md.mesh.bending.Helfrich(mesh_obj)
    helfrich_potential.params["mesh"] = dict(k=SP.k_bend)
    integrator.forces.append(helfrich_potential)

    # Area conservation potential:
    k_area = SP.k_area_i
    TriArea = 4 * np.pi * R**2 / len(faces)
    area_potential = hoomd.md.mesh.conservation.TriangleArea(mesh_obj)
    area_potential.params.default = dict(k=k_area, A0=TriArea)
    integrator.forces.append(area_potential)

    # Add wall:
    wall = [hoomd.wall.Plane(origin=(0, 0, -R-sigma), normal=(0, 0, 1))]
    wlj = hoomd.md.external.wall.LJ(walls=wall)
    wlj.params['mesh'] = {"sigma": unit_sigma, "epsilon": 1.0, "r_cut": 2**(1/6)*unit_sigma}
    wlj.params[['A','A_const','A_flattener']] = {"epsilon": 0.0, "sigma": 1.0, "r_cut": 0.}
    integrator.forces.append(wlj)


    #############################################
    ## Initialize the simulation
    #############################################

    snap = sim.state.get_snapshot()
    print('after potentials:', snap.particles.diameter)

    # GSD logger:
    logger = hoomd.logging.Logger(['particle','constraint'])
    gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(2000)), #int(2000)
                               filename=job.fn('Setup.gsd'),
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

    print('Equilibrating mesh...')
    sim.run(5000)

    while k_area < SP.k_area_f:
        snapshot = sim.state.get_snapshot()
        if snapshot.communicator.rank == 0:
            all_positions = snapshot.particles.position
            R_vert_avg = np.mean(all_positions[snapshot.particles.typeid == 0])
        TriArea = 4 * np.pi * R_vert_avg**2 / len(faces)
        k_area *= 2
        area_potential.params.default = dict(k=k_area, A0=TriArea)
        sim.run(5000)

    k_area = SP.k_area_f
    area_potential.params.default = dict(k=k_area, A0=TriArea)
    print('k_area set to: ', k_area)
    sim.run(5000)

    # Calculate fake gravity for debugging:
    #F_const_rod = 0
    #F_const_mesh = 0 


    # Add gravity:
    print('\nAdding gravity...')
    mass_mesh_particle = 1
    gravity = hoomd.md.force.Constant(filter=hoomd.filter.All())
    gravity.constant_force['A'] = (0,0,F_const_rod)
    gravity.constant_force['A_const','A_flattener'] = (0,0,0)
    gravity.constant_force['mesh'] = (0,0,F_const_mesh)
    gravity.constant_torque['mesh','A','A_const','A_flattener'] = (0,0,0)

    integrator.forces.append(gravity)

    #############################################
    ## Run the simulation
    #############################################
    print('\nFinish equilibrating simulation...')
    while sim.timestep < (SP.equiltime - 10000):
        sim.run(10000)
        gsd_oper.flush()
        print('step: ', sim.timestep)

    
    # Add rod active force:
    print('\nAdding active force...')
    active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
    active.active_force['A'] = (SP.v0,0,0)
    active.active_torque['A'] = (0,0,0)
    integrator.forces.append(active)

    sim.run(9999)
    
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
    print_state(sigma, mesh_sigma, flattener_sigma, N_particles, num_flattener, N_active, num_beads, bead_spacing, N_mesh, R, SP.aspect_rat, SP.freedom_rat, deltas, SP.torque_mag, mass_mesh_bead, mass_rod, F_const_rod, F_const_mesh, job)

    
    os.rename(job.fn('Setup.out.in_progress'), job.fn('Setup.out'))

    print('Initialization complete.')
    '''
    print('Running for demo...')
    while sim.timestep < SP.runtime:
        sim.run(100000)
        gsd_oper.flush()
        print('step: ', sim.timestep)

    print('Simulation complete.')
    '''

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
