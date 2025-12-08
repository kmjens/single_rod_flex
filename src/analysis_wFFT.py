import argparse
import freud
import glob
import hoomd
import json
import matplotlib
import os
import signac
import plotly

import gsd
import gsd.hoomd
import gsd.pygsd

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utility import *
from plotting_utility import *

from scipy.fft import fft, fftfreq
from scipy.stats import linregress
from scipy.spatial.transform import Rotation as Rot

from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

def Analysis_implementation(job, communicator):
    #############################################
    ## Unit Conversions
    #############################################
    SP = JobParser(job)

    # the following relations assume that sigma = 1
    R = (SP.freedom_rat * SP.aspect_rat) / 2
    N_mesh = int(4 * np.pi * R**2 / SP.TriArea) + 2
    cylinder_vol = SP.aspect_rat * np.pi * (1 / 2) ** 2
    
    BG = BuoyancyAndGravity(R, N_mesh, cylinder_vol)
    
    speed_conv = BG.len_conv / BG.time_conv # m/s per 1 speed sim unit
    speed_conv_mm = speed_conv*1e3
    time_conv_ms = BG.time_conv *1e3
    len_conv_um = BG.len_conv *1e6
    time_conv_min = BG.time_conv /60
    
    step_to_sec = BG.time_conv * SP.dt

    #############################################
    ## Collect positions
    #############################################
    
    # Open trajectory:
    f = gsd.pygsd.GSDFile(open(job.fn('Run.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)
    initial_frame = traj[0]
    
    num_frames = len(traj)
    box = initial_frame.configuration.box
    box_instance = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2], xy=box[3], xz=box[4], yz=box[5])
    
    time_indices = np.arange(len(traj))
    timesteps = time_indices * 10000
    timesteps_exp = timesteps*step_to_sec
    
    # Set up for ffts:
    freq = fftfreq(len(timesteps_exp), d=(timesteps_exp[1] - timesteps_exp[0]))  # Frequency values
    fft_xlim_max = timesteps_exp[-1]/5
    
    # Identify type indices
    typeid_array = initial_frame.particles.typeid
    mesh_indices   = np.where(typeid_array == 0)[0]
    active_indices = np.where(typeid_array == 1)[0]
    n_mesh   = len(mesh_indices)
    n_active = len(active_indices)
    
    # Allocate arrays for mesh particles
    mesh_positions       = np.empty((num_frames, n_mesh, 3), dtype=float)
    mesh_images          = np.empty((num_frames, n_mesh, 3), dtype=float)
    mesh_unwrapped       = np.empty((num_frames, n_mesh, 3), dtype=float)
    mesh_com_unwrapped   = np.empty((num_frames, 3), dtype=float)
    mesh_com_velocity    = np.empty((num_frames, 3), dtype=float)
    mesh_com_v_norms     = np.empty((num_frames, 1), dtype=float)
    mesh_normalized_v    = np.empty((num_frames, 3), dtype=float)
    
    # Allocate arrays for active particles
    active_positions       = np.empty((num_frames, n_active, 3), dtype=float)
    active_images          = np.empty((num_frames, n_active, 3), dtype=float)
    active_unwrapped       = np.empty((num_frames, n_active, 3), dtype=float)
    active_com_unwrapped   = np.empty((num_frames, 3), dtype=float)
    active_com_velocity    = np.empty((num_frames, 3), dtype=float)
    active_com_v_norms     = np.empty((num_frames, 1), dtype=float)
    active_normalized_v    = np.empty((num_frames, 3), dtype=float)
    active_com_orientation = np.empty((num_frames, 4), dtype=float)  # quaternion (x, y, z, w)
    
    # Initialize variables for displacement and distance calculations
    active_total_distance = 0
    mesh_total_distance = 0
    
    disp_active = [0]  # Starting displacement is 0
    disp_mesh = [0]
    active_total_distances = [0]
    mesh_total_distances = [0]
    
    # Rotational Autocorrelation stuff
    #3 because I calculate it for l = 2,4,6 to get diff symmetry orders
    racf = np.empty((num_frames, 3), dtype=complex)
    previous_ori = 'FALSE'

    # Loop through trajectory
    for i, frame in enumerate(traj):
        pos = frame.particles.position
        img = frame.particles.image
        tid = frame.particles.typeid
        vel = frame.particles.velocity
        ori = frame.particles.orientation
    
        #timesteps.append(frame.configuration.step)
    
        # Mesh particles (typeid == 0)
        mesh_mask = (tid == 0)
        mesh_pos  = pos[mesh_mask]
        mesh_img  = img[mesh_mask]
        mesh_vel  = vel[mesh_mask]
    
        mesh_positions[i]     = mesh_pos
        mesh_images[i]        = mesh_img
        mesh_unwrapped[i]     = box_instance.unwrap(mesh_pos, mesh_img)
        mesh_com_unwrapped[i] = np.mean(mesh_unwrapped[i], axis=0)
        mesh_com_velocity[i]  = np.mean(mesh_vel, axis=0)
        mesh_com_v_norms[i]   = np.linalg.norm(mesh_com_velocity[i])
        mesh_normalized_v[i]  = mesh_com_velocity[i] / mesh_com_v_norms[i]
    
        # Active particles (typeid == 1)
        active_mask = (tid == 1)
        active_pos  = pos[active_mask]
        active_img  = img[active_mask]
        active_vel  = vel[active_mask]
        active_ori  = ori[active_mask]
    
        active_positions[i]     = active_pos
        active_images[i]        = active_img
        active_unwrapped[i]     = box_instance.unwrap(active_pos, active_img)
        active_com_unwrapped[i] = np.mean(active_unwrapped[i], axis=0)
        active_com_velocity[i]  = np.mean(active_vel, axis=0)
        active_com_v_norms[i]   = np.linalg.norm(active_com_velocity[i])
        active_normalized_v[i]  = active_com_velocity[i] / active_com_v_norms[i]
    
        # Orientation (average quaternion)
        if len(active_ori) > 1:
            r = Rot.from_quat(active_ori)
            mean_rot = r.mean()
            active_com_orientation[i] = mean_rot.as_quat()
        else:
            active_com_orientation[i] = active_ori[0]
    
        # Displacement and total distance calculations
        if i > 0:
            # Displacements
            disp_active.append(np.linalg.norm(active_com_unwrapped[i] - active_com_unwrapped[0]))
            disp_mesh.append(np.linalg.norm(mesh_com_unwrapped[i] - active_com_unwrapped[0]))
    
            # Total distance and MSDs
            vector_active = active_com_unwrapped[i] - active_com_unwrapped[i-1]
            active_total_distance += np.linalg.norm(vector_active)
            active_total_distances.append(active_total_distances[-1] + np.linalg.norm(vector_active))
    
            vector_mesh = mesh_com_unwrapped[i] - mesh_com_unwrapped[i-1]
            mesh_total_distance += np.linalg.norm(vector_mesh)
            mesh_total_distances.append(mesh_total_distances[-1] + np.linalg.norm(vector_mesh))
    
        ## Rotational Autocorrelation Stuff
        # If there are previous quaternions, rotational autocorrelation with freud
        count = 0
        active_ori /= np.linalg.norm(active_ori, axis=1, keepdims=True)
        for l in [int(2),int(4),int(6)]:
            rot_auto = freud.order.RotationalAutocorrelation(l)
            if previous_ori == 'TRUE':
                rot_auto.compute(ref_orientations=prev_ori, orientations=active_ori)
                racf[i,count] = rot_auto.particle_order
                #print(rot_auto.particle_order)
            
            count += 1
        prev_ori = active_ori
        previous_ori = 'TRUE'
    
    active_init_pos = active_com_unwrapped[0]  # Initialize with the first frame's position
    com_init_pos = mesh_com_unwrapped[0]
    
    # Correlation between displacements
    disp_corr = np.corrcoef(disp_active, disp_mesh)[0, 1]
    dist_corr = np.corrcoef(active_total_distances, mesh_total_distances)[0 ,1]
    
    # Convert lists to arrays for further use
    disp_active = np.array(disp_active)
    disp_mesh = np.array(disp_mesh)
    active_total_distances = np.array(active_total_distances)
    mesh_total_distances = np.array(mesh_total_distances)
    
    print("Finished processing mesh and active particle COM positions, velocities, orientations, displacements, and total distances.")

    #####################################################
    ## Plot and save velocity and orientation information
    #####################################################

    cmap_blue = plt.get_cmap('Blues')
    cmap_red = plt.get_cmap('Reds')
    norm_t = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

    plot_rotational_autocorrelation_function(timesteps, racf, job)

    quats = active_com_orientation / np.linalg.norm(active_com_orientation, axis=1, keepdims=True)
    local_active_dir = np.tile([1, 0, 0], 1)
    
    directions = []
    rotations = []
    
    for q in range(len(quats)):
        rot = Rot.from_quat([quats[q][1], quats[q][2], quats[q][3], quats[q][0]])
        rotations.append(rot)
        directions.append(rot.apply(local_active_dir))  # 'apply` works per quaternion
    
    directions = np.array(directions)  # Convert directions to numpy array after loop
    
    # Spherical coordinates
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arccos(z)         # 0 (north pole) to pi (south pole)
    phi = np.arctan2(y, x)       # -pi to pi
    phi_unwrapped = np.unwrap(phi) # Unwrap bc time-series lines

    plot_orientation_path_on_S2(timesteps,x,y,z,job)

    plot_orientation_2D_versus_time(timesteps_exp, phi, phi_unwrapped, theta, job)

    ## FFTs of orientation signals
    phi_fft = fft(np.degrees(phi))
    phi_unwrapped_fft = fft(np.degrees(phi_unwrapped))
    theta_fft = fft(np.degrees(theta))
    
    plot_FFT_of_orientation_data(freq, fft_xlim_max, phi_fft, phi_unwrapped_fft, theta_fft, job)
    
    # values to record in analysis data
    fft_orient_phi_data = extract_FFT_data(phi_fft, freq)
    fft_orient_phi_unwrapped_data = extract_FFT_data(phi_unwrapped_fft, freq)
    fft_orient_theta_data = extract_FFT_data(theta_fft, freq)

    
    ## Autocorrelation function
    acf = quaternion_acf(active_com_orientation)
    acf_fft = fft(acf)
    acf_fft_data  = extract_FFT_data(acf_fft, freq) #save to analysis

    plot_autocorrelation_function(timesteps_exp, acf, job)
    plot_autocorrelation_function_FFT(timesteps_exp, acf, acf_fft, freq, fft_xlim_max, job)

    cmap=cmap_blue
    plot_contrasted_active_angular_dist(active_normalized_v, directions, cmap, job)
    plot_orientation_on_S2_for_active(phi,theta,job)
    plot_1D_orientation_hists_for_active(phi,theta,job)
    plot_v_distrib_on_S2(active_normalized_v, mesh_normalized_v, job)


    orientations = active_normalized_v
    suptitle='Distribution of Active Particle Velocity Direction'
    cmap = plt.cm.Blues
    plot_2D_w_1D_velocoity_hist(orientations, suptitle, cmap, job)

    orientations = mesh_normalized_v
    suptitle='Distribution of Mesh Velocity Direction'
    cmap = plt.cm.Reds
    plot_2D_w_1D_velocoity_hist(orientations, suptitle, cmap, job)


    plot_COM_velocity(timesteps, active_com_v_norms, mesh_com_v_norms, speed_conv_mm, step_to_sec, job)

    active_v_fft = fft(mesh_com_v_norms)
    mesh_v_fft = fft(active_com_v_norms)
    fft_active_com_v_data  = extract_FFT_data(active_v_fft, freq)
    fft_mesh_com_v_data = extract_FFT_data(mesh_v_fft, freq)
    plot_COM_velocity_FFT(freq, active_com_v_norms, mesh_com_v_norms, fft_xlim_max, job)


    AM = AlignmentMetrics()
    AM.compute(mesh_com_velocity, active_com_velocity,
                  mesh_com_unwrapped, active_com_unwrapped,
                  active_com_orientation)

    results = AM.summary_stats()
    plot_alignment_metrics(AM, timesteps, step_to_sec, job)

    # FFT of inner product signals
    AM.compute_FFT_AM()

    ## Save to analysis
    inner_v_fft_data  = extract_FFT_data(AM.inner_v_fft, freq)
    inner_v_fft_xy_data = extract_FFT_data(AM.inner_v_fft_xy, freq)
    inner_ori_fft_data = extract_FFT_data(AM.inner_ori_fft, freq)
    inner_ori_fft_xy_data = extract_FFT_data(AM.inner_ori_fft_xy, freq)
    inner_pos_fft_data = extract_FFT_data(AM.inner_pos_fft, freq)
    inner_normpos_fft_data = extract_FFT_data(AM.inner_normpos_fft, freq)
    inner_normpos_fft_xy_data = extract_FFT_data(AM.inner_normpos_fft_xy, freq)
    
    plot_alignment_metrics_FFT(AM, freq, timesteps, step_to_sec, fft_xlim_max, job)

    #############################################
    ## Plot and save x-y plane trajectories
    #############################################
        
    mesh_x = mesh_com_unwrapped[:, 0]
    mesh_y = mesh_com_unwrapped[:, 1]
    mesh_z = mesh_com_unwrapped[:, 2]
    
    active_x = active_com_unwrapped[:, 0]
    active_y = active_com_unwrapped[:, 1]
    active_z = active_com_unwrapped[:, 2]
    
    # mid colors for legend
    mid_color_mesh = cmap_red(norm_t(timesteps[int(3*len(timesteps)/4)]))  
    mid_color_active = cmap_blue(norm_t(timesteps[int(3*len(timesteps)/4)])) 
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('white')
    
    sc_mesh = ax.scatter(mesh_x*len_conv_um, mesh_y*len_conv_um, 
                        c=timesteps, cmap=cmap_red, 
                        norm=norm_t, label="Mesh Center of Mass", s=9, alpha=0.7, edgecolor='none')
    sc_active = ax.scatter(active_x*len_conv_um, active_y*len_conv_um, 
                        c=timesteps, cmap=cmap_blue, 
                        norm=norm_t, label="Active Particle", s=9, alpha=0.7, edgecolor='none')
    
    
    cbar_com = fig.colorbar(sc_mesh, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
    cbar_com.set_label('Simulation Timestep')
    cbar_active = fig.colorbar(sc_active, orientation='vertical', fraction=0.05, pad=0.02)
    cbar_active.ax.yaxis.set_ticks([])
    
    
    legend_mesh = Line2D([0], [0], marker='o', color='w', markerfacecolor=mid_color_mesh, markersize=9, label="Mesh Center of Mass")
    legend_active = Line2D([0], [0], marker='o', color='w', markerfacecolor=mid_color_active, markersize=9, label="Active Particle")
    
    ax.legend(handles=[legend_mesh, legend_active], loc='best')
    
    ax.set_xlabel(r"x position ($\mu$m)")
    ax.set_ylabel(r"y position ($\mu$m)")
    ax.set_title(f"x-y trajectory\nJob ID: {job}")
    
    ax.axis('equal')
    ax.grid()
    
    plt.show()
    fig.savefig(job.fn('xy_traj.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print('XY Trajectories plotted.')


    #############################################
    ## Z-Position Analysis
    #############################################

    active_z_exp = []
    mesh_z_exp = []
    
    for i in range(len(active_z)):
        active_z_exp.append((active_z[i]+R+1)*len_conv_um)
        mesh_z_exp.append((mesh_z[i]+R+1)*len_conv_um)
    
    z_diff = (mesh_z - active_z)*len_conv_um
    
    # Set limits of z_diff plot to be related to height of mesh
    lim = R
    if max(z_diff) > abs(min(z_diff)):
        if max(z_diff) > lim:
            lim = max(z_diff)
    else:
        if abs(min(z_diff)) > lim:
            lim = abs(min(z_diff))
    
    plot_z_position_data(active_z_exp, mesh_z_exp, z_diff, timesteps, lim, step_to_sec, job)
    
    # FFT of inner profuct signals
    diff_z_fft = fft(z_diff)
    mesh_z_fft = fft(mesh_z_exp)
    active_z_fft = fft(active_z_exp)
    
    ## Save to analysis
    diff_z_fft_data  = extract_FFT_data(diff_z_fft, freq)
    mesh_z_fft_data  = extract_FFT_data(mesh_z_fft, freq)
    active_z_fft_data  = extract_FFT_data(active_z_fft, freq)
    
    plot_z_position_FFT(freq, mesh_z_fft, active_z_fft, diff_z_fft, fft_xlim_max, job)
    
    
    #############################################
    ## Spherical deformation
    #############################################
    final_pos = mesh_unwrapped[-1]
    axes = [(max(final_pos[:, 0]) - min(final_pos[:, 0]))* len_conv_um,
            (max(final_pos[:, 1]) - min(final_pos[:, 1]))* len_conv_um,
            (max(final_pos[:, 2]) - min(final_pos[:, 2]))* len_conv_um]
        
    # Mesh length, width, height aspect ratios:
    ratio_xz = axes[0] / axes[2]
    ratio_yz = axes[1] / axes[2]
    ratio_xy = axes[0] / axes[1]
    
    # Polar eccentricity: 
    '''
    e = 0 is perfect sphere.
    e --> 1 with infinite deformation
    Here, we assume symmetry along x-y plane and take average.
    We assume z axis is shorter than x-y because of gravity.
    Else, we set e to NaN.
    '''
    max_ax = (axes[0] + axes[1])/2 # assume xy symmetry
    min_ax = axes[2] # set to axis_z
    eccen = 1 - (min_ax * min_ax) / (max_ax * max_ax)
    
    if eccen < 0:
        eccen = np.nan  # Set to NaN
    else:
        eccen = np.sqrt(eccen)
    
    # Volume deviations:
    R_avg = (axes[0] * axes[1] * axes[2])**(1/3)
    V_sphere = (4/3) * np.pi * R_avg**3
    V_actual = (4/3) * np.pi * axes[0] * axes[1] * axes[2]
    V_diff_sqr = (V_actual - V_sphere)**2
    
    # Sphericity: (assume oblate spheriod)
    '''
    Phi = 1 for perfect sphere
    Phi > 1 as deformations increase.
    Here, we assume symmetry along the x-y plane.
    We asume z-axis is shorted than x-y because of gravity. (ie. oblate)
    Else, we set Phi to NaN
    '''
    SA_sphere = (4 * np.pi)**(1/3) * ((3 * V_actual)**(2/3))
    SA_actual = 2 * np.pi * max_ax**2 * (1 + (min_ax/(max_ax * eccen)) * np.arcsin(eccen))
    SA_diff_sqr = (SA_actual - SA_sphere)**2
    Phi = SA_actual / SA_sphere
    
    plot_ellipticity_metrics(axes, job)
    
    print('Spherical deformation analysis completed.')
    
    #############################################
    ## Active-mesh positional change
    #############################################
        
    # Peason correlation of active and mesh com positions 
    '''
    1:  perfect correlation
    0:  no correlation
    -1: perfect neg correlation
    '''
    Pear_corrs = []
    for i in range(3):  # x, y, z
        corr = np.corrcoef(mesh_com_unwrapped[:, i], 
                           active_com_unwrapped[:, i])[0, 1]
        Pear_corrs.append(corr)
    
    # Convert to experimental units (um)
    disp_active_um = disp_active * len_conv_um
    disp_mesh_um = disp_mesh * len_conv_um
    active_total_distances_um = active_total_distances * len_conv_um
    mesh_total_distances_um = mesh_total_distances * len_conv_um
    active_total_distance_um = active_total_distance * len_conv_um
    mesh_total_distance_um = mesh_total_distance * len_conv_um
    
    active_linReg = {'slope':'none',
                    'intercept':'none',
                    'r_value':'none',
                    'p_value':'none',
                    'std_err':'none',
                    'r_squared':'none'}
    
    mesh_linReg = {'slope':'none',
                    'intercept':'none',
                    'r_value':'none',
                    'p_value':'none',
                    'std_err':'none',
                  'r_squared':'none'}
    
    # Linear Regression
    active_linReg['slope'], active_linReg['intercept'], active_linReg['r_value'], active_linReg['p_value'], active_linReg['std_err'] = linregress(timesteps_exp, active_total_distances_um)
    active_linReg['r_squared'] = active_linReg['r_value'] ** 2
    
    mesh_linReg['slope'], mesh_linReg['intercept'], mesh_linReg['r_value'], mesh_linReg['p_value'], mesh_linReg['std_err'] = linregress(timesteps_exp, mesh_total_distances_um)
    mesh_linReg['r_squared'] = mesh_linReg['r_value'] ** 2
    
    plot_distances(timesteps, mesh_total_distances_um, active_total_distances_um, step_to_sec, active_linReg, mesh_linReg, disp_corr, job)
    plot_displacement(timesteps, disp_active_um, disp_mesh_um, step_to_sec, disp_corr, job)
    
    # FFT of inner profuct signals
    mesh_disp_fft = fft(disp_mesh_um)
    active_disp_fft = fft(disp_active_um)
    
    ## Save to analysis
    mesh_disp_fft_data  = extract_FFT_data(mesh_disp_fft, freq)
    active_disp_fft_data  = extract_FFT_data(active_disp_fft, freq)
    
    plot_displacement_FTT(timesteps, freq, mesh_disp_fft, active_disp_fft, step_to_sec, fft_xlim_max, job)
    
    #############################################
    ## Save results to json file
    #############################################
    
    all_data = {
        "jobid": job.id,
    }
    statepoints = job.sp
    all_data.update(statepoints)
    
    analysis_data = {
        "mesh_len_ratio_xz": float(ratio_xz),
        "mesh_len_ratio_yz": float(ratio_yz),
        "mesh_len_ratio_xy": float(ratio_xy),
        "eccentricity":      float(eccen),
        "spherical_volume":  float(V_sphere), #um^3
        "ellipsoidal_volume": float(V_actual* len_conv_um**3), #um^3
        "volume_square_difference": float(V_diff_sqr), #um^6
        "spherical_SA":         float(SA_sphere), #um^2
        "ellipsoidal_volume":   float(SA_actual), #um^3 (?)
        "SA_square_difference": float(SA_diff_sqr), #um^4
        "sphericity":           float(Phi),
        "pos_correlation_x":    float(Pear_corrs[0]),
        "pos_correlation_y":    float(Pear_corrs[1]),
        "pos_correlation_z":    float(Pear_corrs[2]),
        "displacement_correlation":    float(disp_corr),
        "distance_correlation":    float(dist_corr),
        "std_norm_v_dot": float(results['velocity_alignment']['std']),
        "avg_norm_v_dot": float(results['velocity_alignment']['avg']),
        "max_norm_v_dot": float(results['velocity_alignment']['max']),
        "min_norm_v_dot": float(results['velocity_alignment']['std']),
        "std_norm_v_dot_xy": float(results['velocity_alignment_xy']['std']),
        "avg_norm_v_dot_xy": float(results['velocity_alignment_xy']['avg']),
        "max_norm_v_dot_xy": float(results['velocity_alignment_xy']['max']),
        "min_norm_v_dot_xy": float(results['velocity_alignment_xy']['std']),
        "std_norm_pos_dot": float(results['position_alignment']['std']),
        "avg_norm_pos_dot": float(results['position_alignment']['avg']),
        "max_norm_pos_dot": float(results['position_alignment']['max']),
        "min_norm_pos_dot": float(results['position_alignment']['min']),
        "std_norm_pos_dot_xy": float(results['position_alignment_xy']['std']),
        "avg_norm_pos_dot_xy": float(results['position_alignment_xy']['avg']),
        "max_norm_pos_dot_xy": float(results['position_alignment_xy']['max']),
        "min_norm_pos_dot_xy": float(results['position_alignment_xy']['min']),
        "std_norm_ori_dot": float(results['orientation_alignment']['std']),
        "avg_norm_ori_dot": float(results['orientation_alignment']['avg']),
        "max_norm_ori_dot": float(results['orientation_alignment']['max']),
        "min_norm_ori_dot": float(results['orientation_alignment']['min']),
        "std_norm_ori_dot_xy": float(results['orientation_alignment_xy']['std']),
        "avg_norm_ori_dot_xy": float(results['orientation_alignment_xy']['avg']),
        "max_norm_ori_dot_xy": float(results['orientation_alignment_xy']['max']),
        "min_norm_ori_dot_xy": float(results['orientation_alignment_xy']['min']),
        "mesh_net_distance": float(disp_mesh_um[-1]), #um
        "active_net_distance": float(disp_active_um[-1]), #um
        "mesh_total_distance": float(mesh_total_distance_um), #um
        "active_total_distance": float(active_total_distance_um), #um
        "active_mesh_diff_distance": float(active_total_distance_um-mesh_total_distance_um), #um
        "slope_dist_mesh": float(mesh_linReg['slope']), #um/sec
        "slope_err_dist_mesh": float(mesh_linReg['std_err']),      #units?
        "slope_dist_active": float(active_linReg['slope']), #um/sec
        "slope_err_dist_active": float(active_linReg['std_err']),
        "slope_diff_dists": float(active_linReg['slope'] - mesh_linReg['slope']), #um/sec
        "fft_orient_phi_max": float(fft_orient_phi_data['max_amp']),
        "fft_orient_phi_freq": float(fft_orient_phi_data['max_freq']),
        "fft_orient_phi_unwrapped_max": float(fft_orient_phi_unwrapped_data['max_amp']),
        "fft_orient_phi_unwrapped_freq": float(fft_orient_phi_unwrapped_data['max_freq']),
        "fft_orient_theta_max": float(fft_orient_theta_data['max_amp']),
        "fft_orient_theta_freq": float(fft_orient_theta_data['max_freq']),
        "fft_active_com_v_max": float(fft_active_com_v_data['max_amp']),
        "fft_active_com_v_freq": float(fft_active_com_v_data['max_freq']),
        "fft_mesh_com_v_max": float(fft_mesh_com_v_data['max_amp']),
        "fft_mesh_com_v_freq": float(fft_mesh_com_v_data['max_freq']),
        "inner_v_fft_max": float(inner_v_fft_data['max_amp']),
        "inner_v_fft_freq": float(inner_v_fft_data['max_freq']),
        "inner_v_fft_xy_max": float(inner_v_fft_xy_data['max_amp']),
        "inner_v_fft_xy_freq": float(inner_v_fft_xy_data['max_freq']),
        "inner_ori_fft_max": float(inner_ori_fft_data['max_amp']),
        "inner_ori_fft_freq": float(inner_ori_fft_data['max_freq']),
        "inner_ori_fft_xy_max": float(inner_ori_fft_xy_data['max_amp']),
        "inner_ori_fft_xy_freq": float(inner_ori_fft_xy_data['max_freq']),
        "inner_pos_fft_max": float(inner_pos_fft_data['max_amp']),
        "inner_pos_fft_freq": float(inner_pos_fft_data['max_freq']),
        "inner_normpos_fft_max,": float(inner_normpos_fft_data['max_amp']),
        "inner_normpos_fft_freq": float(inner_normpos_fft_data['max_freq']),
        "inner_normpos_fft_xy_max": float(inner_normpos_fft_xy_data['max_amp']),
        "inner_normpos_fft_xy_freq": float(inner_normpos_fft_xy_data['max_freq']),
        "diff_z_fft_max": float(diff_z_fft_data['max_amp']),
        "diff_z_fft_freq": float(diff_z_fft_data['max_freq']),
        "mesh_z_fft_max": float(mesh_z_fft_data['max_amp']),
        "mesh_z_fft_freq": float(mesh_z_fft_data['max_freq']),
        "active_z_fft_max": float(active_z_fft_data['max_amp']),
        "active_z_fft_freq": float(active_z_fft_data['max_freq']),
        "mesh_disp_fft_max": float(mesh_disp_fft_data['max_amp']),
        "mesh_disp_fft_freq": float(mesh_disp_fft_data['max_freq']),
        "active_disp_fft_max": float(active_disp_fft_data['max_amp']),
        "active_disp_fft_freq": float(active_disp_fft_data['max_freq']),
        "acf_fft_max": float(acf_fft_data['max_amp']),
        "acf_fft_freq": float(acf_fft_data['max_freq'])
    }
    all_data.update(analysis_data)
    with open(job.fn('analysis_data_wFFT.json'), 'w') as f:
        json.dump(all_data, f, indent=4)
    
    with open(job.fn('signac_job_document.json'), 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print('Analysis complete.')

def Analysis_wFFT(*jobs):
    
    processes_per_directory = os.environ['ACTION_PROCESSES_PER_DIRECTORY']
    communicator = hoomd.communicator.Communicator()
    Analysis_implementation(jobs[communicator.partition], communicator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)
