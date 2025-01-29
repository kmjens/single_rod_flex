## Simulate single flexicles with single rod inside
# Flatten spherocylinders to explore shape phase space

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

    kT      = job.cached_statepoint['kT']
    R       = job.cached_statepoint['R']
    N_mesh  = int(job.cached_statepoint['N_mesh'])
    dt      = job.cached_statepoint['dt']
    v0      = job.cached_statepoint['v0']
    simseed = job.cached_statepoint['seed']
    L       = R * 5 # box size
    gamma   = job.cached_statepoint['gamma']
    gamma_r = [gamma/3.0,gamma/3.0,gamma/3.0]
    mesh_gamma  = 5
    runtime     = job.cached_statepoint['runtime']
    equiltime   = job.cached_statepoint['equiltime']
    
    k_bend   = job.cached_statepoint['k_bend']
    k_bond   = 4 * k_bend # based on general ratio lipid bilayers have
    k_area_f = job.cached_statepoint['k_area_f']
    k_area   = job.cached_statepoint['k_area_i']
    
    # Adding active particles:
    N_active    = int(job.cached_statepoint['N_active'])
    num_beads   = int(job.cached_statepoint['num_beads']) #number of beads including the center particle in rigid body
    num_const_beads = int(num_beads - 1)
    aspect_rat  = job.cached_statepoint['aspect_rat'] # aspect ratio of rod length to diam
    freedom_rat = job.cached_statepoint['freedom_rat'] # ratio of flex diam to rod length
    num_filler  = job.cached_statepoint['num_filler'] # num on one active particle
    N_filler    = 2 * num_filler * N_active # including all active particles
    filler_diam_ratio = job.cached_statepoint['filler_diam_ratio']
    gravity_strength  = np.abs(job.cached_statepoint['gravity_strength']) 
    gravity_ratio = np.abs(job.cached_statepoint['gravity_ratio'])
    N_bead  = num_const_beads * N_active 
    
    # with R and sigma  dependence switched
    sigma        = 1
    mesh_sigma   = 0.333 * sigma
    rod_length   = aspect_rat * sigma
    bead_spacing = (rod_length / 2) - (sigma / 2) 
    R            = (freedom_rat * rod_length) / 2
    filler_sigma = filler_diam_ratio * sigma
    sphero_vol   = (sigma ** 3) * (3 * rod_length - 1) / 4 # approx as spherocylinder
    ideal_buffer = 0.5
    N_mesh       = int(np.ceil(4 * np.pi * R**2 * 0.8))

    N_particles = N_mesh + N_active
    
    
    # Adding torque:
    rand_orient     = job.cached_statepoint['rand_orient']
    active_angle    = job.cached_statepoint['active_angle']
    torque_mag      = job.cached_statepoint['torque_mag']
    
    #############################################
    ## Real units:
    #############################################
    
    # Conversion Factors
    mass_conv = 3.13e-5         # kg per 1 sim units 
    len_conv = 3e-6             # m per 1 sim units (ie 3um)
    energy_conv = 2.0709735e-20 # J per 1 sim units (ie 5kT)
    time_conv = 116.2           # sec per 1 sim units

    # Experimental values
    rho_rod = 1.12              # kg/m^3
    rho_outer_fluid = 1.014     # kg/m^3
    rho_inner_fluid = 1.025     # kg/m^3
    rho_mesh_V = 0.88           # kg/m^3
    mesh_thickness = 6.31e-9    # m
    mesh_area_density = rho_mesh_V * mesh_thickness # kg/m^2

    # convert params to sim units
    g = 9.8 * (time_conv)**2 / mass_conv

    rho_rod *= len_conv**3 / mass_conv
    rho_outer_fluid *= len_conv**3 / mass_conv
    rho_inner_fluid *= len_conv**3 / mass_conv
    
    V_rod = 10 * np.pi * (1 / 2)**2 * (10/3)    # sim units
    mass_rod = V_rod * rho_rod                  # sim units
    
    SA_flex = 4 * np.pi * R**2
    V_flex = (4 / 3) * np.pi * R**3
    mesh_area_density *= len_conv**2 / mass_conv
    mesh_mass = mesh_area_density * SA_flex
    mesh_particle_mass = mesh_mass / N_mesh
    
    # Buoyant force and gravitational force
    F_grav_mesh = g * mesh_particle_mass
    F_grav_rod = g * mass_rod
    F_boy_mesh = g * rho_outer_fluid * V_flex
    F_boy_rod = g * rho_inner_fluid * V_rod
    F_const_mesh = F_boy_mesh - F_grav_mesh
    F_const_rod = F_boy_rod - F_grav_rod
    
    print('mesh particle mass: ', mesh_particle_mass)
    print('F_grav_mesh: ', F_grav_mesh, ' F_grav_rod: ', F_grav_rod)
    print('F_boy_mesh: ', F_boy_mesh, ' F_boy_rod: ', F_boy_rod)
    print('F_const_mesh: ', F_boy_mesh, ' F_const_rod: ', F_boy_rod)


    with open(job.fn('Run.out.in_progress'), 'w') as file:
        file.write('Initializing sim seed: ' + str(simseed) + '\n')


    #############################################
    ## Set up simulation object
    #############################################
    
    device = hoomd.device.CPU(num_cpu_threads=communicator.num_ranks)
    sim = hoomd.Simulation(device=device)
    sim.seed = simseed


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
    A_MoI = np.zeros((len(A_position),3),dtype=float)
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


    #############################################    
    # Construct rod rigid bodies
    #############################################    
    
    bead_type_list = ['A_const'] * N_bead
    bead_pos_list = get_bead_pos(bead_spacing, sigma, num_const_beads) 
    bead_orient_list = [(1,0,0,0)] * N_bead
    bead_diam = [sigma] * N_bead 
    
    filler_type_list = ['A_filler'] * (2 * num_filler)
    filler_pos_list = get_filler_pos(num_filler, sigma, filler_sigma, rod_length)
    filler_orient_list = [(1,0,0,0)] * (2 * num_filler)
    filler_diam = [filler_sigma] * N_filler

    const_type_list = bead_type_list + filler_type_list
    const_pos_list = bead_pos_list + filler_pos_list
    const_orient_list = bead_orient_list +filler_orient_list

    assert len(const_type_list) == len(const_pos_list) == len(const_orient_list)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": const_type_list,
        "positions":         const_pos_list,
        "orientations":      const_orient_list
    }
    rigid.create_bodies(sim.state)

    diameter = np.append(diameter, bead_diam, axis=0)
    diameter = np.append(diameter, filler_diam, axis=0)
    print('check diam: ', diameter)

    
    # Create initial gsd
    snapshot = sim.state.get_snapshot()
    frame = gsd.hoomd.Frame()
    frame.particles.N = len(snapshot.particles.position)
    frame.particles.position = snapshot.particles.position
    frame.particles.orientation = snapshot.particles.orientation
    frame.particles.typeid = snapshot.particles.typeid
    frame.particles.diameter = diameter[0:frame.particles.N]
    frame.particles.types = snapshot.particles.types
    frame.configuration.box = snapshot.configuration.box
    
    with gsd.hoomd.open(name=job.fn('initial_wRigid.gsd'), mode='w') as f:
        f.append(frame)


    #############################################    
    ## Set up filters and integrator
    #############################################

    filter_all  = hoomd.filter.All()
    filter_mesh = hoomd.filter.Type(['mesh'])
    filter_free = hoomd.filter.Rigid(("center","free"))
    
    integrator = hoomd.md.Integrator(
            dt=dt, 
            rigid=rigid, 
            integrate_rotational_dof=True)
    sim.operations.integrator = integrator

    langevin = hoomd.md.methods.Langevin(filter=filter_free, kT=kT)
    langevin.gamma.default = gamma
    langevin.gamma_r.default = [gamma,gamma,gamma]
    integrator.methods.append(langevin)


    #############################################
    ## Add potentials
    #############################################
    
    cell = hoomd.md.nlist.Cell(buffer=ideal_buffer, exclusions=['meshbond','body'])

    # Expanded LJ:
    ExpLJ = hoomd.md.pair.ExpandedLJ(nlist=cell, mode="shift", default_r_cut=0)
    
    unit_sigma = sigma
    deltas = np.ones((3,3))
    sigmas = [sigma, mesh_sigma, filler_sigma]
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
    ExpLJ.params[('A_filler','A'),
                 ('A_filler','A_const')] = dict(epsilon=1,
                                                sigma=unit_sigma, 
                                                delta=deltas[0][2])
    ExpLJ.params[('A_filler','mesh')] = dict(epsilon=1, 
                                             sigma=unit_sigma, 
                                             delta=deltas[1][2])
    ExpLJ.params[('A_filler','A_filler')] = dict(epsilon=0, 
                                                 sigma=unit_sigma, 
                                                 delta=deltas[2][2])

    ExpLJ.r_cut[('A','A'),
                ('A_const','A'),
                ('A_const','A_const')] = 2**(1.0/6.) * (unit_sigma) + deltas[0][0]
    ExpLJ.r_cut[('mesh','A'),
                ('mesh','A_const')] = 2**(1.0/6)*(unit_sigma)+deltas[0][1]
    ExpLJ.r_cut[('mesh','mesh')] = 2**(1.0/6.)*(unit_sigma)+deltas[1][1]
    ExpLJ.r_cut[('A_filler','A'),
                ('A_filler','A_const')] = 2**(1.0/6)*unit_sigma + deltas[0][2]
    ExpLJ.r_cut[('A_filler','mesh')] = 2**(1.0/6)*(unit_sigma) + deltas[1][2]
    ExpLJ.r_cut[('A_filler','A_filler')] = 0

    integrator.forces.append(ExpLJ)

    # Apply tethering potential to mesh:
    l_min, l_c1, l_c0, l_max = get_tether_params(frame, triangle_tags)

    mesh_bond_potential = hoomd.md.mesh.bond.Tether(mesh_obj)
    mesh_bond_potential.params["mesh"] = dict(
            k_b=k_bond,
            l_min=l_min,
            l_c1=l_c1,
            l_c0=l_c0,
            l_max=l_max)
    integrator.forces.append(mesh_bond_potential)

    # Helfrich bending potential:
    helfrich_potential = hoomd.md.mesh.bending.Helfrich(mesh_obj)
    helfrich_potential.params["mesh"] = dict(k=k_bend)
    integrator.forces.append(helfrich_potential)

    # Area conservation potential:
    TriArea = 4 * np.pi * R**2 / len(faces)
    area_potential = hoomd.md.mesh.conservation.TriangleArea(mesh_obj)
    area_potential.params.default = dict(k=k_area, A0=TriArea)
    integrator.forces.append(area_potential)

    # Add wall:
    wall = [hoomd.wall.Plane(origin=(0, 0, -R-sigma), normal=(0, 0, 1))]
    wlj = hoomd.md.external.wall.LJ(walls=wall)
    wlj.params['mesh'] = {"sigma": unit_sigma, "epsilon": 1.0, "r_cut": 2**(1/6)*unit_sigma}
    wlj.params[['A','A_const','A_filler']] = {"epsilon": 0.0, "sigma": 1.0, "r_cut": 0.}
    integrator.forces.append(wlj)
    

    #############################################
    ## Initialize the simulation
    #############################################

    # Initialize: 
    sim.state.thermalize_particle_momenta(filter=filter_all, kT=kT)
    sim.run(0)
    print('Successfully ran for 0 timestep.\n')
    
    print('Equilibrating mesh...')
    sim.run(5000)
    
    while k_area < k_area_f:
        snapshot = sim.state.get_snapshot()
        if snapshot.communicator.rank == 0:
            all_positions = snapshot.particles.position
            R_vert_avg = np.mean(all_positions[snapshot.particles.typeid == 0])
        TriArea = 4 * np.pi * R_vert_avg**2 / len(faces)
        k_area *= 2
        area_potential.params.default = dict(k=k_area, A0=TriArea)
        sim.run(5000)
    
    k_area = k_area_f
    area_potential.params.default = dict(k=k_area, A0=TriArea)
    print('k_area set to: ', k_area)
    sim.run(5000)
    
    # Add gravity:
    print('\nAdding gravity...')
    mass_mesh_particle = 1
    gravity = hoomd.md.force.Constant(filter=hoomd.filter.All())
    #gravity.constant_force['A'] = (0,0,-gravity_strength)
    gravity.constant_force['A'] = (0,0,F_const_rod)
    gravity.constant_force['A'] = (0,0,0)
    gravity.constant_force['A_const','A_filler'] = (0,0,0)
    #gravity.constant_force['mesh'] = (0,0,-gravity_strength)
    gravity.constant_force['mesh'] = (0,0,F_const_mesh)
    gravity.constant_force['mesh'] = (0,0,0)
    gravity.constant_torque['mesh','A','A_const','A_filler'] = (0,0,0)

    integrator.forces.append(gravity)
    sim.run(10000)

    # Add rod active force:
    print('\nAdding active force...')
    active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
    active.active_force['A'] = (v0,0,0)    
    active.active_torque['A'] = (0,0,0)
    integrator.forces.append(active)

    #############################################
    ## Run the simulation
    #############################################
    print('\nFinish equilibrating simulation...')
    while sim.timestep < equiltime:
        sim.run(10000)
        print('step: ', sim.timestep)

    print('\nCurrent state:')
    print_state(sigma, mesh_sigma, filler_sigma, num_filler, N_active, num_beads, bead_spacing, N_mesh, R, aspect_rat, freedom_rat, deltas, gravity_strength, torque_mag, job)
    
    # GSD logger:
    logger = hoomd.logging.Logger(['particle','constraint'])
    gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(200)), #int(2000)
                               filename=job.fn('active.gsd'),
                               logger=logger, mode='wb',
                               dynamic=['momentum','property','attribute/particles/diameter'],
                               filter=filter_all)
    gsd_oper.write_diameter = True
    sim.operations += gsd_oper

    print('\nRunning simulation...')
    while sim.timestep < runtime:
        sim.run(10000)
        gsd_oper.flush()
        print('step: ', sim.timestep)

    os.rename(job.fn('Run.out.in_progress'), job.fn('Run.out'))
    
    print('Simulation complete.')

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
