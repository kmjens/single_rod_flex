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
    k_area_f = job.cached_statepoint['k_area']
    k_area   = 0
    
    # Adding active particles:
    N_active    = int(job.cached_statepoint['N_active'])
    ratio_len   = job.cached_statepoint['ratio_len'] # defines sigma
    num_filler  = job.cached_statepoint['num_filler'] # num on one active particle
    N_filler    = num_filler * N_active # including all active particles
    filler_diam_ratio = job.cached_statepoint['filler_diam_ratio']
    gravity_strength  = np.abs(job.cached_statepoint['gravity_strength']) 
    gravity_ratio = np.abs(job.cached_statepoint['gravity_ratio'])
    N_particles = N_mesh + N_active
    
    rod_size_int = int(job.cached_statepoint['rod_size_int'])
    num_bead = ((rod_size_int - 1) * 2)
    N_bead  = num_bead * N_active 
    
    mesh_sigma   = 1
    sigma        = (2/rod_size_int) * R / ratio_len
    rod_size     = sigma * rod_size_int
    filler_sigma = filler_diam_ratio * sigma
    sphero_vol   = (sigma ** 3) * (3 * rod_size - 1) / 4 # approx as spherocylinder
    ideal_buffer = 0.5

    # Adding torque:
    rand_orient     = job.cached_statepoint['rand_orient']
    active_angle    = job.cached_statepoint['active_angle']
    torque_mag      = job.cached_statepoint['torque_mag']
    
    #############################################
    ## Real units:
    #############################################
    
    g = 9.8 # m/s^2
    g *= 1e12 # sim units
    NA = 6.022e23

    # Mesh mass
    SA_flex = 4 * np.pi * R**2 # sim units (nm^2)?
    V_flex = (4 / 3) * np.pi * R**3
    mesh_area_density = 5.5528e-3  # g/m^2 (from exp)
    mesh_area_density *= NA * 1e-18 # AMU/nm^2 (sim units?)
    mesh_mass = mesh_area_density * SA_flex # sim units
    mesh_particle_mass = mesh_mass / N_mesh
    print('mesh particle mass: ', mesh_particle_mass)
    
    # Particle mass
    density_rod = 1.12 # g/cm^3 (from exp)
    V_rod = np.pi * (3 / 2)**2 *10 # um^3 (from exp)
    mass_rod = density_rod *1e12 * V_rod
    
    # Adjusted mesh mass
    '''
    Because I set my sim up opposite to exp. 
    In my sim I change the rod size -- but in exp they change the flexicle size.
    Need to edit later. Until then:
    This finds an adjusted effective mass of the flexicle (ie. scales the whole system.
    '''
    SA_flex_adjusted = 4 * np.pi * (rod_size * ratio_len/2)**2
    V_flex_adjusted = (4 / 3) * np.pi * (rod_size * ratio_len/2)**3
    mesh_mass_adjusted = mesh_area_density * SA_flex_adjusted
    mesh_particle_mass_adjusted = mesh_mass_adjusted / N_mesh
    print('adjusted mesh particle mass: ', mesh_particle_mass_adjusted)

    F_grav_mesh = g * mesh_particle_mass_adjusted
    F_grav_rod = g * mass_rod
    print('F_grav_mesh: ', F_grav_mesh, ' F_grav_rod: ', F_grav_rod)

    # Buoyancy
    rho_outer_fluid = 1.014 # g/cm^3 (from exp)
    rho_outer_fluid *= 1e-21 * NA # AMU/nm^3 (sim units?)
    F_boy_mesh = g * rho_outer_fluid * V_flex_adjusted # note adjusted

    rho_inner_fluid = 1.025 # g/cm^3 (from exp)
    rho_inner_fluid *= 1e-21 * NA # AMU/nm^3 (sim units?) 
    F_boy_rod = g * rho_inner_fluid * V_rod
    print('F_boy_mesh: ', F_boy_mesh, ' F_boy_rod: ', F_boy_rod)

    F_const_mesh = F_boy_mesh - F_grav_mesh
    F_const_rod = F_boy_rod - F_grav_rod
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
    A_MoI[:,1] = 1.0/12*5*(rod_size*sigma)**2
    A_MoI[:,2] = 1.0/12*5*(rod_size*sigma)**2


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
    
    bead_type_list = ['A_const'] * num_bead
    bead_pos_list = get_bead_pos(rod_size_int, sigma) 
    bead_orient_list = [(1,0,0,0)] * num_bead
    bead_diam = [sigma] * N_bead 
    
    filler_type_list = ['A_filler'] * num_filler
    filler_pos_list = get_filler_pos(num_filler, sigma, filler_sigma, rod_size)
    filler_orient_list = [(1,0,0,0)] * num_filler
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
    
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=['meshbond','body'])

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
    k_area = 50
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
    #mass_frac = mass_mesh_particle * sphero_vol / (4/3* np.pi * mesh_sigma**3)
    gravity = hoomd.md.force.Constant(filter=hoomd.filter.All())
    gravity.constant_force['A'] = (0,0,-gravity_strength)
    gravity.constant_force['A_const','A_filler'] = (0,0,0)
    gravity.constant_force['mesh'] = (0,0,-gravity_strength)
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
    print_state(sigma, mesh_sigma, filler_sigma, num_filler, N_active, deltas, gravity_strength, torque_mag, job)
    
    # GSD logger:
    logger = hoomd.logging.Logger(['particle','constraint'])

    class RadiusLogger:
        '''
        Custom Logger to save radius data.
        '''
        def __init__(self, sim):
            self.sim = sim

        @hoomd.logging.log(category="particles")
        def diameter(self):
            snapshot = self.sim.state.snapshot
            return snapshot.particles.diameters / 2

    radius_logger = RadiusLogger(sim)
    logger.add(radius_logger, quantities=["Radius"])
    gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(200)), #int(2000)
                               filename=job.fn('active.gsd'),
                               logger=logger, mode='wb',
                               dynamic=['momentum','property'],
                               filter=filter_all)
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
