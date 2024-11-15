## Simulate flexicles with rods of various rod number density and bending rigidity.

# First: Vary N_active and k_bend
# Next: Maybe also vary rod_size and v0.

## Boilerplate:
import numpy as np
import time
import gsd.hoomd
import hoomd
import matplotlib
import itertools
import freud
import pandas as pd
#from stl import mesh
import fresnel
import random
import pyvista as pv
from utility import print_state, print_state_to_file, check_for_nan_in_positions, get_filler_pos, q_2_cart, get_rand_orient, fibonacci_sphere, get_active_pos_A, calc_tether_bond_lengths, plot_tether_energies, find_mesh_radius_avg_and_std
import os
import argparse
import signac
import datetime

## Read in statepoints from job
def Run_implementation(job):
    
    # If the product already exists, there is no work to do.
    # Was seeing error with the following lines: unsure why:
    #if job.isfile('Run.out'):
    #    continue

    # Open a temporary file so that the action is not completed early or on error.
    with open(job.fn('Run.out.in_progress'), 'w') as file:
        file.write('Initialization begun for seed: '+str(job.cached_statepoint['seed'])+'\n')
    print("Writing statepoints for job seed: ", job.cached_statepoint['seed'])
    os.system('signac find | xargs signac diff > workspace/statepoints.txt')
    
    current_time = datetime.datetime.now()
    with open(job.fn('output.log'),'a') as file:
        file.write('Simulation started at time: '+str(current_time))
    
    # Collect values of state points from init file.
    kT = job.cached_statepoint['kT']
    R = job.cached_statepoint['R'] 
    N_mesh = int(job.cached_statepoint['N_mesh'])
    gamma = job.cached_statepoint['gamma']
    k_bend = job.cached_statepoint['k_bend']
    k_area = job.cached_statepoint['k_area']
    dt = job.cached_statepoint['dt']
    N_active = int(job.cached_statepoint['N_active'])
    sigma = job.cached_statepoint['sigma']
    ratio_len = job.cached_statepoint['ratio_len'] # couples to sigma (and overides) if N_active is 1
    v0 = job.cached_statepoint['v0']
    runtime = job.cached_statepoint['runtime'] 
    simseed = job.cached_statepoint['seed']
    rand_orient = job.cached_statepoint['rand_orient']
    active_angle = job.cached_statepoint['active_angle']
    torque_mag = job.cached_statepoint['torque_mag']
    num_filler = job.cached_statepoint['num_filler']
    filler_diam_ratio = job.cached_statepoint['filler_diam_ratio']
    gravity_strength = np.abs(job.cached_statepoint['gravity_strength'])

    # Other necessary calculable parameters:
    gamma_r = [gamma/3.0,gamma/3.0,gamma/3.0]
    mesh_gamma = 5
    mesh_sigma = sigma
    k_bond = 4*k_bend # based on general ratio lipid bilayers have
    L = R*5 # box size
    ideal_buffer = 0.5

    # Re-calc sigma if N_active = 1:
    if N_active == 1:
        print('N_active =1 so sigma rescaling')
        print('OLD sigma: ', sigma)

        sigma = (2/3) * R / ratio_len
        rod_size = sigma * 3
        filler_sigma = filler_diam_ratio * sigma

        print('NEW sigma: ', sigma)
        print('rod_size: ', rod_size)
        print('filler_sigma', filler_sigma)
    elif N_active == 0:
        filler_sigma = 0
        rod_size = 3
    else:
        rod_size = 3
        sigma = sigma
        filler_sigma = filler_diam_ratio * sigma
        print('rod_size: ', rod_size)
        print('filler_sigma', filler_sigma)

    ## Make Simulation Object:
    CPU = hoomd.device.CPU()
    sim = hoomd.Simulation(device=CPU)
    sim.seed = simseed 
    current_time = datetime.datetime.now() 
    
    # Write a log file
    with open(job.fn('output.log'), 'a') as file:
        file.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\n'.format('kT: '+str(kT),'R: '+str(R),'N_mesh: '+str(N_mesh),'gamma: '+str(gamma),'k_bend: '+str(k_bend),'k_area: '+str(k_area),'dt: '+str(dt),'N_active: '+str(N_active),'sigma: '+str(sigma),'rod_size: '+str(rod_size),'v0: '+str(v0),'runtime: '+str(runtime)))
        file.write('Sim begun at time: '+str(current_time)+'\n ')
        file.write('Beginning init for N_active: '+str(N_active)+' and k_bend: '+str(k_bend)+' with seed: '+str(sim.seed)+'\n')


    ## Mesh initial placement and properties
    # Place mesh on sphere
    x,y,z = fibonacci_sphere(num_pts=N_mesh, R=R)
    mesh_position = np.column_stack((x,y,z))

    # Set initial orientations and typeids of mesh etc...
    # Set initial orientations and typeids of mesh etc...
    mesh_orient = np.zeros((N_mesh,4),dtype=float)
    mesh_orient[:,0] = 1
    mesh_typeid = [0] * N_mesh
    mesh_diam = [0.333 * mesh_sigma] * N_mesh
    mesh_MoI = np.zeros((N_mesh,3),dtype=float)

    ## Active rod initial position and properties
    sphero_vol = sigma**3*(3*rod_size-1)/4
    rho_rod = (N_active * sphero_vol) / (3/4 * np.pi * R**3) #volume fraction filled by rods)
    N_particles = N_mesh + N_active

    # place rods inside flexicle
    if N_active == 1:
        A_position = [np.array([0,0,0])]
    else:
        x = np.arange(R-(rod_size+sigma)/2, -R+(rod_size-sigma)/2, -rod_size*sigma)  # os_ves_min_y + D_ves/2 + D_inner/2
        y = np.arange(-R+1, R, sigma) # pos_ves_max_x - D_ves/2 - D_inner/2
        z = np.arange(-R+1, R, sigma)  # pos_ves_max_x - D_ves/2 - D_inner/2
        pos_in = list(itertools.product(x, y, z))
        np.random.shuffle(pos_in)

        pos_in = np.array(pos_in)
    
        # check if rods inside flexicle to start:
        A_position = []
        zahl = 0 # zahl in German means number haha
        for pi in pos_in:
            pp = mesh_position-pi

            if np.linalg.norm(pi) > R:
                continue

            is_it_in = True
            for jj in range(-rod_size+1,rod_size):
                pp1 = np.linalg.norm(pp+[jj*sigma*0.5,0,0],axis=1)
                if np.any(pp1 < sigma):
                    is_it_in = False
                    break

            if is_it_in:
                A_position.append(pi)
                zahl+=1

            if zahl == N_active:
                break

    # Orientation of rods:
    A_orient = np.zeros((len(A_position),4),dtype=float)
    A_orient[:,0] = 1

    if rand_orient in ['True','true','TRUE']:
        for i in range(N_active):
            overlap = True
            # NOTE: THIS IS NOT YET WORKING FOR rand_orient = true
            while overlap == True:
                A_orient[i] = get_rand_orient()
                # Now check if neighboring active particles overlap treating them like cylinders:
                for j in range(i-1):
                    dj = q_2_cart(A_orient[j])
                    di = q_2_cart(A_orient[i])
                
                    # Calc shortest distance between 2 lines in 3D space:
                    n = np.cross(dj,di)
                    if np.linalg.norm(n) < 1e-6:
                        #lines are //
                        ax_dist = np.linalg.norm(np.cross(di, A_position[j] - A_position[i])) / np.linalg.norm(di)
                    else:
                        ax_dist = np.abs(np.dot((A_position[j] - A_position[i]), n)) / np.linalg.norm(n)

                    # Calc projections of cylinders onto line between their centers
                    center_line = A_position[j] - A_position[i]
                    proj_i = np.abs(np.dot(center_line, di))
                    proj_j = np.abs(np.dot(center_line, dj))

                    ## If the distance between axes is greater than the sum of the radii, they don't overlap
                    if ax_dist >= 2*sigma:
                        overlap = False

                    # Check if the projections of the cylinders onto the line between centers overlap
                    overlap_len = np.linalg.norm(A_position[j] - A_position[i]) - (proj_i + proj_j) / 2
                    if overlap_len <= rod_size:
                        overlap = True
                    else:
                        overlap = False

        with open(job.fn('output.log'),'a') as file:
            file.write('\nInitial rod orientation randomized.\n')
    elif rand_orient in ['False','false','FALSE']:
        with open(job.fn('output.log'),'a') as file:     
            file.write('\nInitial rod orientation set to [1,0,0,0] for all.\n')
    else:
        with open(job.fn('output.log'),'a') as file:
            file.write('\nUnable to identify rand_orient selection.\nInitial rod orientation set to [1,0,0,0] for all by default.\n')
    
    # A Type IDs
    A_typeid = np.ones(len(A_position),dtype=int)
    A_diam = np.ones(len(A_position),dtype=int)
    A_MoI = np.zeros((len(A_position),3),dtype=float)

    # Moment of inertia of a spherocylinder
    A_MoI[:,0] = 0
    A_MoI[:,1] = 1.0/12*5*(rod_size*sigma)**2
    A_MoI[:,2] = 1.0/12*5*(rod_size*sigma)**2

    ## Set up frame
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

    print('Initialization for seed ',sim.seed,' completed!')
    current_time = datetime.datetime.now() 
    with open(job.fn('output.log'),'a') as file:
        file.write('Initialization for seed '+str(sim.seed)+' completed at time:'+str(current_time)+'\n')

    current_time = datetime.datetime.now()
    with open(job.fn('output.log'),'a') as file:
        file.write('Simulation started at time: '+str(current_time)+'\n')


    state = sim.create_state_from_gsd(filename=job.fn('initial.gsd'))
    f = gsd.hoomd.open(name=job.fn('initial.gsd'),mode='r')
    frame = f[0]

    ## Describe rods as rigid bodies
    #constituent particle type, pos, and orient 
    if num_filler == 0:
        filler_sigma = 0
        const_type_list = ['A_const','A_const','A_const','A_const']
        const_pos_list = [(sigma,0,0), (0.5*sigma,0,0), (-0.5*sigma,0,0), (-sigma,0,0)]
        const_orient_list = [(1,0,0,0),(1,0,0,0),(1,0,0,0),(1,0,0,0)]
        
    else:
        bead_type_list = ['A_const','A_const','A_const','A_const']
        bead_pos_list = [(sigma,0,0), (0.5*sigma,0,0), (-0.5*sigma,0,0), (-sigma,0,0)]
        bead_orient_list = [(1,0,0,0),(1,0,0,0),(1,0,0,0),(1,0,0,0)]
        
        filler_pos_list = get_filler_pos(num_filler, sigma, filler_sigma)
        filler_type_list = ['A_filler'] * num_filler
        filler_orient_list = [(1,0,0,0)] * num_filler

        const_type_list = bead_type_list + filler_type_list
        const_pos_list = bead_pos_list + filler_pos_list
        const_orient_list = bead_orient_list +filler_orient_list
    
        # Check both position lists
        if check_for_nan_in_positions(bead_pos_list):
            print("Error in bead positions.")
        else:
            print("Bead positions looking ok!")

        if check_for_nan_in_positions(filler_pos_list):
            print("Error in filler positions.")
        else:
            print("Filler positions looking ok!")

    assert len(const_type_list) == len(const_pos_list) == len(const_orient_list)
    
    #construct the rigid bodies
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types":const_type_list,
        "positions": const_pos_list,
        "orientations":const_orient_list
    }
    rigid.create_bodies(sim.state)

    print('Rigid bodies successfully created.')

    
    # Save initial gsd with rigid bodiesL
    # Get the current simulation state (after rigid bodies have been created)
    snapshot = sim.state.get_snapshot()

    # Only proceed if running on the root rank (snapshot only exists on rank 0 in parallel execution)
    if snapshot.communicator.rank == 0:
        # Create a GSD frame
        frame = gsd.hoomd.Frame()

        # Set particle data in the frame
        frame.particles.N = len(snapshot.particles.position)
        frame.particles.position = snapshot.particles.position
        frame.particles.orientation = snapshot.particles.orientation
        frame.particles.typeid = snapshot.particles.typeid
        frame.particles.types = snapshot.particles.types

        # Set the box dimensions
        frame.configuration.box = snapshot.configuration.box

        # Open the GSD file and append the frame
        with gsd.hoomd.open(name=job.fn('initial_wRigid.gsd'), mode='w') as f:
            f.append(frame)
    
    ## Set up filters and Langevin Integrator
    filter_all = hoomd.filter.All()
    filter_mesh = hoomd.filter.Type(['mesh'])
    filter_free = hoomd.filter.Rigid(("center","free"))

    integrator = hoomd.md.Integrator(dt=dt,rigid=rigid, integrate_rotational_dof=True)
    sim.operations.integrator = integrator

    langevin = hoomd.md.methods.Langevin(filter=filter_free, kT=kT) # integrate only free particles 
    langevin.gamma['mesh'] = mesh_gamma
    langevin.gamma['A'] = gamma
    langevin.gamma_r['A'] = gamma_r
    integrator.methods.append(langevin)

    ## Create simulation mesh
    mesh_position = frame.particles.position[0:N_mesh]
    mesh = pv.PolyData(mesh_position)

    # Delaunay triangulation and surface extraction:
    triangulated = mesh.delaunay_3d()
    surface = triangulated.extract_geometry()
    faces = surface.faces.reshape((-1, 4))
    triangle_points = []

    for face in faces:
        if face[0] == 3:  # Check that the face is a triangle
            triangle_points.append(face[1:])

    triangle_tags = np.vstack((triangle_points))
    N_triangles = len(triangle_tags)

    mesh_obj = hoomd.mesh.Mesh()
    mesh_obj.types = ["mesh"]
    mesh_obj.triangulation = dict(type_ids = [0] * N_triangles,
          triangles = triangle_tags)

    ## Add potentials
    # Apply tethering potential to mesh:
    triangle_cartesian_positions = frame.particles.position[triangle_tags]
    global_max_edge = np.linalg.norm(triangle_cartesian_positions[:,0]-triangle_cartesian_positions[:,2],axis=1).max()
    global_min_edge = np.linalg.norm(triangle_cartesian_positions[:,0]-triangle_cartesian_positions[:,2],axis=1).min()

    avg_edge_len = (global_max_edge + global_min_edge)/2 # pot is very sensitive to this

    print("\nTETHERING POT PARAMS:\n")
    print('glob max edge: ', global_max_edge)
    print('glob min edge: ', global_min_edge)
    print('avg edge len: ', avg_edge_len)

    l_min   = (2/3) * avg_edge_len
    l_c1    = 0.85 * avg_edge_len
    l_c0    = 1.15 * avg_edge_len
    l_max   = (4/3) * avg_edge_len

    print('l_min: ', l_min,', l_c1: ', l_c1,', l_c0: ', l_c0,', l_max: ', l_max)

    mesh_bond_potential = hoomd.md.mesh.bond.Tether(mesh_obj)
    mesh_bond_potential.params["mesh"] = dict(k_b=k_bond, l_min=l_min, l_c1=l_c1,
                                     l_c0=l_c0, l_max=l_max)
    integrator.forces.append(mesh_bond_potential)

    # WCA pot between all particles:
    cell = hoomd.md.nlist.Cell(buffer=0.4, exclusions=['meshbond'])

    # Expanded LJ
    ExpLJ = hoomd.md.pair.ExpandedLJ(nlist=cell, default_r_cut=0) #, mode="shift")

    # Get deltas
    unit_sigma = sigma
    
    deltas = np.ones((3,3))
    sigmas = [sigma, mesh_sigma, filler_sigma]
    for i in range(len(sigmas)):
        for j in range(len(sigmas)):
            deltas[i][j] = (sigmas[i] + sigmas[j])/2 - unit_sigma

    delta_AA = deltas[0][0] 
    delta_Am = deltas[0][1] 
    delta_Af = deltas[0][2] 
    delta_fm = deltas[1][2] 
    delta_ff = deltas[2][2]  
    delta_mm = deltas[1][1] 

    ExpLJ.params.default  = dict(epsilon=0, sigma=unit_sigma, delta=0)
    ExpLJ.params[('mesh','mesh')]   = dict(epsilon=1, sigma=unit_sigma, delta=delta_mm)
    ExpLJ.params[('mesh','A'),('mesh','A_const')]   = dict(epsilon=1, sigma = unit_sigma, delta=delta_Am)
    ExpLJ.params[('A_filler','A'),('A_filler','A_const')]   = dict(epsilon=1, sigma=unit_sigma, delta=delta_Af)
    ExpLJ.params[('A_filler','A_filler')]   = dict(epsilon=1, sigma=unit_sigma, delta=delta_ff)
    ExpLJ.params[('A_filler','mesh')]       = dict(epsilon=1, sigma=unit_sigma, delta=delta_fm)
    
    ExpLJ.r_cut[('A','A'),('A_const','A'),('A_const','A_const')] = 2**(1.0/6.)*(unit_sigma) + delta_AA
    ExpLJ.r_cut[('mesh','A'),('mesh','A_const')] = 2**(1.0/6)*(unit_sigma)+delta_Am
    ExpLJ.r_cut[('mesh','mesh')] = 2**(1.0/6.)*(unit_sigma)+delta_mm
    ExpLJ.r_cut[('A_filler','A'),('A_filler','A_const')] =2**(1.0/6)*unit_sigma + delta_Af
    ExpLJ.r_cut[('A_filler','A_filler')]=2**(1.0/6)*(unit_sigma) + delta_ff
    ExpLJ.r_cut[('A_filler','mesh')]=2**(1.0/6)*(unit_sigma) + delta_fm

    integrator.forces.append(ExpLJ)

    # Print state info for easy reference
    print_state(sigma, mesh_sigma, filler_sigma, num_filler, N_active, deltas, gravity_strength, torque_mag, job)
    state_log_file = job.fn("print_state.log")
    print_state_to_file(sigma, mesh_sigma, filler_sigma, num_filler, N_active, deltas, gravity_strength, torque_mag, state_log_file, job)
    state_log_file = job.fn("output.log")
    print_state_to_file(sigma, mesh_sigma, filler_sigma, num_filler, N_active, deltas, gravity_strength, torque_mag, state_log_file, job)

    # Helfrich bending potential to mesh:
    helfrich_potential = hoomd.md.mesh.bending.Helfrich(mesh_obj)
    helfrich_potential.params["mesh"] = dict(k=k_bend)
    #integrator.forces.append(helfrich_potential)

    # Area conservation:
    TriArea = 4*np.pi*R**2/len(faces) # Area of triangle = surface area of mesh / number of triangles
    area_potential = hoomd.md.mesh.conservation.TriangleArea(mesh_obj)
    area_potential.params.default = dict(k=0,A0=TriArea)
    #integrator.forces.append(area_potential)

    # Active forces to rods:
    active = hoomd.md.force.Active(filter=hoomd.filter.Type(['A']))
    active.active_force['A'] = (v0,0,0)

    if not(torque_mag == 0 and active_angle == 0):
        aa_rad = active_angle * np.pi / 180
        T_norm = 1 / (np.sqrt(1**2 + np.sin(aa_rad)))
        T_hat = (T_norm, np.sin(aa_rad) * T_norm, 0) 
        active.active_torque['A'] = tuple(i * torque_mag for i in T_hat)
    else:
        active.active_torque['A'] = (0,0,0)
    # Will add to integrator after initialization
    
    if gravity_strength == 0:
        print('\nGravity off.')
        with open(job.fn('output.log'), 'a') as file:
            file.write('Gravity off.' +'.\n')
    else:
        # Add wall geometry:
        wall_pos = - L + R
        bottom = hoomd.wall.Plane(origin=(0, 0, wall_pos), normal=(0, 0, 1))
        
        wall_lj = hoomd.md.external.wall.LJ(walls=[bottom])
        wall_lj.params.default = dict(sigma=sigma, epsilon=0, r_cut=(sigma)* 2.0 ** (1 / 6))
        wall_lj.params['mesh'] = dict(sigma=mesh_sigma, epsilon=1.0, r_cut=(mesh_sigma)* 2.0 ** (1 / 6))

        sim.operations.integrator.forces.append(wall_lj)
        mass_mesh_particle = 1
        mass_frac = mass_mesh_particle * sphero_vol / (4/3* np.pi * mesh_sigma**3) # m1/m2 = V1/V2 if same density

        gravity = hoomd.md.force.Constant(filter=hoomd.filter.All())
        gravity.constant_force['A'] = (0,0,-mass_frac*gravity_strength)
        gravity.constant_force['A_const','A_filler'] = (0,0,0)
        gravity.constant_force['mesh'] = (0,0,-gravity_strength)
        gravity.constant_torque['mesh','A','A_const','A_filler'] = (0,0,0)

        integrator.forces.append(gravity)

        print('\nGravity has been turned on.\nG strength = ',str(gravity_strength), '.\nWall located at z = ',wall_pos)
        with open(job.fn('output.log'), 'a') as file:
            file.write('Gravity has been turned on.\n G strength = ' +str(gravity_strength)+'\nWall located at z = '+ str(wall_pos) +'.\n')


    ## Run the Sim:
    # GSD logger:
    logger = hoomd.logging.Logger(['particle','constraint'])

    # This gsd writer saves ALL particles
    gsd_oper = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(int(1)), filename=job.fn('active.gsd'),logger=logger,mode='wb',dynamic=['momentum','property'],filter=filter_all)
    sim.operations += gsd_oper

    # slowly increase area conservation strength to prevent simulation to blow up
    print('\n Begin sim by running for 5000 timesteps with k_area = 0 and active forces off.')
    
    
    sim.run(0)
    gsd_oper.flush()
    sim.run(5)
    gsd_oper.flush()
    sim.run(5000)
    gsd_oper.flush()

    
    print('\nGetting average area of triangles on mesh after equilibration')
    # Get position of type ids and average them to get average distance from 0. 
    snapshot = sim.state.get_snapshot()
    # Check if the snapshot is on the CPU
    if snapshot.communicator.rank == 0:  # Ensure we're accessing the data on the root rank
        # Get the positions of all particles
        all_positions = snapshot.particles.position
        R_vert_avg = np.mean(all_positions[snapshot.particles.typeid == 0])
    
    # use this to calc TriArea:
    TriArea = 4*np.pi*R_vert_avg**2/len(faces) # Area of triangle = surface area of mesh / number of triangles
    print('Average mesh triangle area set to: ', TriArea)

    print('Gradually increase k_area.')
    k_area = 100
    area_potential.params.default = dict(k=k_area,A0=TriArea)
    while k_area < job.cached_statepoint['k_area']:
        with open(job.fn('output.log'), 'a') as file:
            file.write('k_area: '+str(k_area)+'\n')
        sim.run(5000)
        gsd_oper.flush()
        k_area *= 2
        area_potential.params.default = dict(k=k_area,A0=TriArea)
        sim.run(15000)
        Ravg, Rstd  = find_mesh_radius_avg_and_std(sim)
        print('timestep: ', sim.timestep,'\n')
        with open(job.fn('output.log'), 'a') as file:
            file.write('timestep: '+str(sim.timestep)+'\n')
            file.write('radius: '+ str(Ravg)+' and std: '+str(Rstd)+'\n')
    print('k_area set to: ', k_area)
    
    print('Turning on the rod active force at timestep: ', sim.timestep, '\n')
    integrator.forces.append(active)
    while sim.timestep < runtime:
        sim.run(10000)
        gsd_oper.flush()
        print('timestep: ', sim.timestep,'\n')
        with open(job.fn('output.log'), 'a') as file:
            file.write('timestep: '+str(sim.timestep)+'\n')

    with open(job.fn('output.log'), 'a') as file:
        file.write('Simulation completed at timestep: '+str(sim.timestep)+'\n')

    print('\nSimulation seed '+str(sim.seed)+' completed at timestep: '+str(sim.timestep)+'\n')
    
    current_time = datetime.datetime.now()
    # Sim run complete! Rename the temporary file to the product file.
    with open(job.fn('Run.out.in_progress'), 'w') as file:
        file.write('Simulation Run completed for seed: '+str(job.cached_statepoint['seed'])+' at timestep: '+str(sim.timestep)+' and time: '+str(current_time)+'\n')
    
    os.rename(job.fn('Run.out.in_progress'), job.fn('Run.out'))

def Run(*jobs):
    '''
    Process jobs in paralllel with the mpi4py packages.
    The number of ranks must be equal to the number of directories.
    '''
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != len(jobs):
        message = 'Number of ranks does not match number of directories.'+str(MPI.COMM_WORLD.Get_size())
        raise RuntimeError(message)

    rank = MPI.COMM_WORLD.Get_rank()
    Run_implementation(jobs[rank])

if __name__ == '__main__':
    # Parse the command line arguments: python action.py --action <ACTION> [DIRECTORIES]
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()
    
    # Open the signac jobs
    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    
    # Call the action
    globals()[args.action](*jobs)
