import json
import matplotlib
import os
import signac

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from itertools import combinations
from collections import defaultdict

# Currently not in use:

class JobParser:
    '''
    Collect variables defined in init.py.
    Store experimental values.
    Calculation sim to exp conversion factors.
    Perform a few basic calculations on statepoints.
    '''
    def __init__(self, job):

        self.kT      = job.cached_statepoint['kT']
        self.R       = job.cached_statepoint['R']
        self.N_mesh  = int(job.cached_statepoint['N_mesh'])
        self.dt      = job.cached_statepoint['dt']
        self.v0      = job.cached_statepoint['v0']
        self.simseed = job.cached_statepoint['seed']
        self.runtime     = job.cached_statepoint['runtime']
        self.equiltime   = job.cached_statepoint['equiltime']

        self.k_bend   = job.cached_statepoint['k_bend']
        self.k_bond   = 4 * self.k_bend # based on general ratio lipid bilayers have
        self.k_area_f = job.cached_statepoint['k_area_f']
        self.k_area_i = job.cached_statepoint['k_area_i']
        self.TriArea  = job.cached_statepoint['TriArea']

        # Particle and mesh size scaling:
        self.aspect_rat      = job.cached_statepoint['aspect_rat'] # aspect ratio of rod length to diam
        self.freedom_rat     = job.cached_statepoint['freedom_rat'] # ratio of flex diam to rod length
        self.flattener_sigma_rat = job.cached_statepoint['flattener_sigma_rat']
        self.mesh_sigma_rat = job.cached_statepoint['mesh_sigma_rat']

        # Optionally adding torque:
        self.rand_orient     = job.cached_statepoint['rand_orient']
        self.active_angle    = job.cached_statepoint['active_angle']
        self.torque_mag      = job.cached_statepoint['torque_mag']


class BuoyancyAndGravity:
    ''':
    Store experimental values and conversion factors.
    Calculation sim to exp conversion factors.
    Perform a few basic calculations on statepoints.
    '''
    def __init__(self, R, N_mesh, cylinder_vol):
        #############################################
        ## Real units:
        #############################################
        
        # Conversion Factors
        self.mass_conv = 7.863e-14       # kg per 1 sim units (ie approx mass of one rod)
        self.len_conv = 3e-6             # m per 1 sim units (ie 3um, approx rod width)
        self.energy_conv = 2.0709735e-20 # J per 1 sim units (ie 5kT)
        self.time_conv = self.len_conv * self.mass_conv**(1/2) / (self.energy_conv**(1/2))           # sec per 1 sim units
        print('time unit conversion: ', self.time_conv, 'sec/sim unit')

        # Experimental values
        self.rho_rod = 1120              # kg/m^3
        self.rho_outer_fluid = 1014      # kg/m^3
        self.rho_inner_fluid = 1025      # kg/m^3
        self.rho_mesh_V = 880            # kg/m^3
        self.mesh_thickness = 6.31e-9    # m
        self.mesh_area_density = self.rho_mesh_V * self.mesh_thickness # kg/m^2

        # convert params to sim units
        self.g = 9.8 * (self.time_conv)**2 / self.len_conv

        self.rho_rod *= self.len_conv**3 / self.mass_conv
        self.rho_outer_fluid *= self.len_conv**3 / self.mass_conv
        self.rho_inner_fluid *= self.len_conv**3 / self.mass_conv

        self.mass_rod = cylinder_vol * self.rho_rod        # sim units

        self.SA_flex = 4 * np.pi * R**2
        self.V_flex = (4 / 3) * np.pi * R**3
        self.mesh_area_density *= self.len_conv**2 / self.mass_conv
        self.mesh_mass = self.mesh_area_density * self.SA_flex
        self.flex_mass = self.mesh_mass + self.rho_inner_fluid * self.V_flex

        #self.rho_mesh_and_fluid = ((self.rho_inner_fluid * self.V_flex) + self.mesh_mass) / self.V_flex# rho considering mass of mesh and mass of fluid

        # Buoyant force and gravitational force
        self.F_grav_mesh = self.g * self.flex_mass / N_mesh
        self.F_grav_rod = self.g * self.mass_rod
        self.F_boy_mesh = self.g * self.rho_outer_fluid * self.V_flex / N_mesh
        self.F_boy_rod = self.g * self.rho_inner_fluid * cylinder_vol
        self.F_const_mesh = self.F_boy_mesh - self.F_grav_mesh
        self.F_const_rod = self.F_boy_rod - self.F_grav_rod
        print('F_grav_mesh: ', self.F_grav_mesh, ' F_grav_rod: ', self.F_grav_rod)
        print('F_boy_mesh: ', self.F_boy_mesh, ' F_boy_rod: ', self.F_boy_rod)
        print('F_const_mesh: ', self.F_const_mesh, ' F_const_rod: ', self.F_const_rod)


def get_tether_params(frame, triangle_tags):
    triangle_cartesian_positions = frame.particles.position[triangle_tags]
    global_max_edge = np.linalg.norm(triangle_cartesian_positions[:,0]-triangle_cartesian_positions[:,2],axis=1).max()
    global_min_edge = np.linalg.norm(triangle_cartesian_positions[:,0]-triangle_cartesian_positions[:,2],axis=1).min()

    avg_edge_len = (global_max_edge + global_min_edge)/2

    l_min   = (2/3) * avg_edge_len
    l_c1    = 0.85 * avg_edge_len
    l_c0    = 1.15 * avg_edge_len
    l_max   = (4/3) * avg_edge_len

    return(l_min, l_c1, l_c0, l_max)

def print_state(sigma, mesh_sigma, flattener_sigma, N_particles,  num_flattener, N_active, num_beads, bead_spacing, N_mesh, R, aspect_rat, freedom_rat, deltas, torque_mag, mass_mesh_bead, mass_rod, F_const_rod, F_const_mesh, job):
    print('sigma: ', sigma)
    print('mesh_sigma: ', mesh_sigma)
    print('flattener_sigma: ', flattener_sigma)

    print('\n')
    print('N_particles: ', N_particles)
    print('num_flattener: ', num_flattener)
    print('N_active: ', N_active)
    print('num_beads: ', num_beads)
    print('bead_spacing:', bead_spacing)
    print('N_mesh: ', N_mesh)
    print('mesh R: ', R)
    
    print('\n')
    print('aspect_rat: ', aspect_rat)
    print('freedom_rat: ', freedom_rat)
    
    print('\n')
    print('mass_mesh_bead: ',mass_mesh_bead)
    print('mass_rod: ',mass_rod)
    print('F_const_rod: ',F_const_rod)
    print('F_const_mesh: ',F_const_mesh)

    print('\n')
    print('deltas:')
    print('AA: ', deltas[0][0], '\nAm: ', deltas[0][1], '\nAf: ', deltas[0][2], '\nfm: ',deltas[1][2], '\nff: ', deltas[2][2],'\nmm: ', deltas[1][1])
    
    if torque_mag == 0:
        print('torque off')
    else:
        print('torque_mag: ', torque_mag)
    
    state_log_file = job.fn('state.log')
    with open(state_log_file, "w") as f:
        print('job: ', job, file=f)
        print('statepoints: ', job.sp, '\n\n', file=f)
        print('N_particles: ', N_particles, file=f)
        print('sigma: ', sigma, file=f)
        print('mesh_sigma: ', mesh_sigma, file=f)
        print('flattener_sigma: ', flattener_sigma, file=f)
        print('num_flattener: ', num_flattener, file=f)
        print('N_active: ', N_active, file=f)
        print('num_beads: ', num_beads, file=f)
        print('bead_spacing:', bead_spacing, file=f)
        print('N_mesh: ', N_mesh, file=f)
        print('mesh R: ', R, file=f)
        print('aspect_rat: ', aspect_rat,file=f)
        print('freedom_rat: ', freedom_rat,file=f)

        print('\n',file=f)
        print('mass_mesh_bead: ',mass_mesh_bead,file=f)
        print('mass_rod: ',mass_rod,file=f)
        print('F_const_rod: ',F_const_rod,file=f)
        print('F_const_mesh: ',F_const_mesh,file=f)

        print('deltas:', file=f)
        print('AA: ', deltas[0][0], ' Am: ', deltas[0][1], ' Af: ', deltas[0][2], 
              ' fm: ', deltas[1][2], ' ff: ', deltas[2][2], ' mm: ', deltas[1][1], file=f)
        if torque_mag == 0:
            print('torque off', file=f)
        else:
            print('torque_mag: ', torque_mag)

def get_bead_pos(rod_length, sigma, num_const_beads):
    """
    Place beads symmetrically around the origin (0, 0, 0).
    If `num_const_beads` is odd, include an additional bead at the end.
    """
    x_end = (rod_length - sigma) /2
    x_start = - x_end

    x_coords = []
    num_total_beads = num_const_beads
    half_beads = num_total_beads // 2

    spacing = (x_end - x_start) / (num_const_beads - 1)

    for i in range(half_beads):
        x_coords.append(x_start + i * spacing)
        x_coords.append(x_end - i * spacing)

    x_coords = list(sorted(set(x_coords)))

    # Insert zero bead and remove if needed
    x_coords.insert(len(x_coords) // 2, 0)
    bead_pos_list = [(x, 0, 0) for x in x_coords]
    if (0, 0, 0) in bead_pos_list:
        bead_pos_list.remove((0, 0, 0))

    return bead_pos_list


def get_flattener_pos(num_flattener,sigma,flattener_sigma,rod_length):
    '''
    Returns flattener_sigma (ie. r_flattener*2)
    Returns the positions of the flattener particles that
    flatten the spherocylinder relative to 
    the origin of the rigid body as a list [[x1,y1,z1],[x2,y2,z2],...]

    Takes in number of flattener particles and diameter of rod spheros.
    '''
    if num_flattener == 0:
        return []
    else:
        ref_theta = 2 * np.pi / num_flattener

        x = rod_length/2 - flattener_sigma/2
        a = sigma/2 - flattener_sigma/2 # radius of circle to rotate flattener points on (// to xy-plane)

        theta = np.linspace(0, 2 * np.pi, num_flattener)
        neg_flattener_pos = [[-x,round(a*np.sin(i),10),round(a*np.cos(i),10)] for i in theta]
        pos_flattener_pos = [[x,round(a*np.sin(i),10),round(a*np.cos(i),10)] for i in theta]
        flattener_pos = neg_flattener_pos + pos_flattener_pos

    return flattener_pos



def fibonacci_sphere(num_pts, R):
    '''
    Returns points on the surface of a sphere. 
    num_pts is the number of points on the sphere.
    R is the radius of my desired mesh. 
    '''

    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)

    return x, y, z

def get_rand_orient():
    x1,x2,x3,x4 = np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)
    
    S1 = x1**2 + x2**2
    S2 = x3**2 + x4**2
    
    while S1 > 1:
        x1,x2 = np.random.uniform(-1,1), np.random.uniform(-1,1)   
        S1 = x1**2 + x2**2

    while S2 > 1:
        x3,x4 = np.random.uniform(-1,1), np.random.uniform(-1,1)   
        S2 = x3**2 + x4**2

    c = x3 * np.sqrt((1-S1)/S2)
    d = x4 * np.sqrt((1-S1)/S2)
    q = [x1,x2,c,d]
    return q 

def q_2_cart(q):
    """Convert a quaternion into a direction vector (unit vector)."""
    w, x, y, z = q
    return np.array([
        2 * (x*z + w*y),
        2 * (y*z - w*x),
        1 - 2 * (x**2 + y**2)
    ])

def get_active_pos_A(rho, radius):
    num_points = int(rho * (radius**3))
    points = []
    for i in range(num_points):
        # Generate random points in a cube of side length 2*radius
        x = np.linspace(-radius+(sigma/2), radius-(sigma/2), num_points)
        y = np.random.uniform(-radius+(sigma/2), radius-(sigma/2), num_points)
        z = np.random.uniform(-radius+(sigma/2), radius-(sigma/2), num_points)
        
        #Calculate the distance of the points from the origin
        distances = np.sqrt(x**2 + y**2 + z**2)
        
        # Select points that lie within the sphere
        mask = distances <= radius
        selected_points = np.vstack((x[mask], y[mask], z[mask])).T
        
        # Add selected points to the list
        points.extend(selected_points)
        points = np.array(points)
    
    #points = np.array(points[:num_points]) # Trim to the desired number of points
    
    return points

def calc_tether_bond_lengths(fmin,fmax,fc1,fc0,k_b):
    '''
    Calculate parameters for tether bond length.
    This function is not optimized right now -- work needs to be done to determine how to do this the best.
    '''
    l_min = global_min_edge*fmin
    l_max = global_max_edge*fmax 
    l_c1 = global_min_edge*fc1
    l_c0 = global_max_edge*fc0

    print('shortest edge length: ', global_min_edge)
    print('longest edge length: ', global_max_edge)
    print('l_max: ',l_max)
    print('l_c0: ',l_c0)
    print('l_c1: ',l_c1)
    print('l_min: ',l_min)
    print('k_bond: ',k_b)
    
    return(l_min,l_max,l_c1,l_c0)


def find_U_tether_bond(r,tether_bond_params):
    '''
    Find total energy of the tether bond given a particular r between two particles.
    Needed to plot the tether energies.
    '''
    l_min = tether_bond_params[0]
    l_c1 = tether_bond_params[1]
    l_c0 = tether_bond_params[2]
    l_max = tether_bond_params[3]
    k_bond = tether_bond_params[4]
    
    if r > l_c0:
        U_att = (k_bond*np.exp(1/(l_c0-r))) / (l_max-r)
    elif r <= l_c0:
        U_att = 0
    
    if r < l_c1:
        U_rep = (k_bond*np.exp(1/(r-l_c1))) / (r-l_min)
    elif r >= l_c1:
        U_rep = 0
    
    U_tot = U_att + U_rep

    return(U_att, U_rep, U_tot)

def plot_tether_energies(tether_bond_params):
    '''
    Reformat U_att, U_rep, and U_tot to be plotted as a function of r
    '''
    l_min = tether_bond_params[0]
    l_c1 = tether_bond_params[1]
    l_c0 = tether_bond_params[2]
    l_max = tether_bond_params[3]
    k_bond = tether_bond_params[4]

    r_values = np.linspace(0, 2, 200)

    U_att_values = []
    U_rep_values = []
    U_tot_values = []
    for r in r_values:
        U_att, U_rep, U_tot = find_U_tether_bond(r, tether_bond_params)
        U_att_values.append(U_att)
        U_rep_values.append(U_rep)
        U_tot_values.append(U_tot)

    plt.figure(figsize=(5, 4))
    plt.plot(r_values, U_att_values, label='U_att', color='blue')
    plt.plot(r_values, U_rep_values, label='U_rep', color='red')
    plt.plot(r_values, U_tot_values, label='U_tot', color='green')
    plt.xlabel('r')
    plt.ylabel('Potential')
    plt.title('Mesh Bond Potential vs. Distance r')
    plt.legend()
    plt.gca().set_facecolor('white') 
    plt.grid(True, color='lightgrey', linestyle =':')
    plt.axvline(x=l_min, color='grey',linestyle ='--',linewidth=1,label='l_min')
    plt.axvline(x=l_max, color='grey',linestyle ='--',linewidth=1,label='l_max')
    plt.text(l_max-0.03,max(U_tot_values)+5,'l_max',color='grey')
    plt.text(l_min-0.03,max(U_tot_values)+5,'l_min',color='grey')
    plt.axvline(x=l_c0, color='grey',linestyle ='--',linewidth=1,label='l_min')
    plt.axvline(x=l_c1, color='grey',linestyle ='--',linewidth=1,label='l_max')
    plt.text(l_c0-0.03,max(U_tot_values)+5,'l_c0',color='grey')
    plt.text(l_c1-0.03,max(U_tot_values)+5,'l_c1',color='grey')
    plt.xlim(l_min-0.1,l_max+0.1)
    plt.ylim(-5,max(U_tot_values)+10)
    plt.show()
    

def find_mesh_radius_avg_and_std(sim):
    '''
    Given: pos, a numpy.ndarray with each particles catersian coordinates:
    Outputs:
        AVGrad: average radius of flexicle if assume spherical.
        STDrad: standard deviation of mesh points from the avg radius value.
    '''   
    snap = sim.state.get_snapshot()
    pos = snap.particles.position    
    
    rad = np.zeros(len(pos))
    for j in range(len(pos)):
        for i in list([0,1,2]):
            rad[j] += pos[j][i]**2
        rad[j] = np.sqrt(rad[j])
    AVGrad = np.mean(rad)
    STDrad = np.std(rad)
    return AVGrad, STDrad


