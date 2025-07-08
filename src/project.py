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


    # Collect triangle Data
    with open(job.fn("triangles.json"), "r") as f:
        tri_data = json.load(f)

    triangle_tags = tri_data["triangles"]
    type_ids = tri_data.get("type_ids", [0] * len(triangle_tags))
    num_tri = len(triangle_tags)
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
    mass_mesh_bead  = BG.mesh_mass / N_mesh

    with open(job.fn('Run.out.in_progress'), 'w') as file:
        file.write('Initializing sim seed: ' + str(SP.simseed) + '\n')
    

    #############################################
    ## Set up simulation object
    #############################################

    device = hoomd.device.CPU(num_cpu_threads=communicator.num_ranks)
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
    filter_mesh = hoomd.filter.Type(['mesh'])
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

   
    # Set up mesh from previous directory
    mesh_obj = hoomd.mesh.Mesh()
    mesh_obj.types = ["mesh"]
    mesh_obj.triangulation = dict(
        type_ids=type_ids,
        triangles=triangle_tags
    )

    
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
    k_area = SP.k_area_f
    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        all_positions = snapshot.particles.position
        R_vert_avg = np.mean(all_positions[snapshot.particles.typeid == 0])
    TriArea = 4 * np.pi * R_vert_avg**2 / num_tri
    
    area_potential = hoomd.md.mesh.conservation.TriangleArea(mesh_obj)
    area_potential.params.default = dict(k=k_area, A0=TriArea)
    integrator.forces.append(area_potential)
    print('k_area set to: ', k_area)
    

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
    gravity.constant_force['mesh'] = (0,0,F_const_mesh)
    gravity.constant_torque['mesh','A','A_const','A_flattener'] = (0,0,0)

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
    print_state(sigma, mesh_sigma, flattener_sigma, N_particles, num_flattener, N_active, num_beads, bead_spacing, N_mesh, R, SP.aspect_rat, SP.freedom_rat, Pe, deltas, SP.torque_mag, mass_mesh_bead, mass_rod, F_const_rod, F_const_mesh, job)
    
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
