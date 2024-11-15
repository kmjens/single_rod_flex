import numpy as np
import matplotlib.pyplot as plt

def print_state(sigma, mesh_sigma, filler_sigma, num_filler, N_active, deltas, gravity_strength, torque_mag, job):
    print('job: ', job)
    print('statepoints: ', job.sp, '\n\n')
    print('sigma: ', sigma)
    print('mesh_sigma: ', mesh_sigma)
    print('filler_sigma: ', filler_sigma)
    print('num_filler: ', num_filler)
    print('N_active: ', N_active)
    print('deltas:')
    print('AA: ', deltas[0][0], ' Am: ', deltas[0][1], ' Af: ', deltas[0][2], ' fm: ',deltas[1][2], ' ff: ', deltas[2][2],' mm: ', deltas[1][1])
    if gravity_strength == 0:
        print('gravity off')
    if torque_mag == 0:
        print('torque off')
    
def print_state_to_file(sigma, mesh_sigma, filler_sigma, num_filler, N_active, deltas, gravity_strength, torque_mag, state_log_file, job):
    with open(state_log_file, "w") as f:
        print('job: ', job, file=f)
        print('statepoints: ', job.sp, '\n\n', file=f)
        print('sigma: ', sigma, file=f)
        print('mesh_sigma: ', mesh_sigma, file=f)
        print('filler_sigma: ', filler_sigma, file=f)
        print('num_filler: ', num_filler, file=f)
        print('N_active: ', N_active, file=f)
        print('deltas:', file=f)
        print('AA: ', deltas[0][0], ' Am: ', deltas[0][1], ' Af: ', deltas[0][2], 
              ' fm: ', deltas[1][2], ' ff: ', deltas[2][2], ' mm: ', deltas[1][1], file=f)
        if gravity_strength == 0:
            print('gravity off', file=f)
        if torque_mag == 0:
            print('torque off', file=f)

def check_for_nan_in_positions(pos_list):
    for i, pos in enumerate(pos_list):
        if any(np.isnan(p) or np.isinf(p) for p in pos):
            print(f"NaN or infinity found in position {i}: {pos}")
            return True
    return False

def get_filler_pos(num_filler,sigma,filler_sigma):
    '''
    Returns filler_sigma (ie. r_filler*2)
    Returns the positions of the filler particles that
    flatten the spherocylinder relative to 
    the origin of the rigid body as a list [[x1,y1,z1],[x2,y2,z2],...]

    Takes in number of filler particles and diameter of rod spheros.
    (warning that this is not generalized for more than 4 constituent particles making up the rod rigid body -- ie only works for rod_size = 3*sigma)
    '''
    ref_theta = 2 * np.pi / num_filler

    x = (3/2)*sigma - filler_sigma/2
    a = sigma/2 - filler_sigma/2 # radius of circle to rotate filler points on (// to xy-plane)

    theta = np.linspace(0, 2 * np.pi, num_filler)
    filler_pos = [[x,round(a*np.sin(i),10),round(a*np.cos(i),10)] for i in theta]

    return filler_pos


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
