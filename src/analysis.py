import argparse
import freud
import hoomd
import json
import matplotlib
import os
import signac
import h5py

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
    time_conv_us = BG.time_conv *1e-6
    len_conv_um = BG.len_conv *1e6
    step_to_sec = BG.time_conv * SP.dt
    

    #############################################
    ## Collect positions and orientations from GSD
    #############################################

    f = gsd.pygsd.GSDFile(open(job.fn('Run.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)
    initial_frame = traj[0]

    num_frames = len(traj)
    box = initial_frame.configuration.box
    box_instance = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2], xy=box[3], xz=box[4], yz=box[5])

    time_indices = np.arange(len(traj))
    timesteps = time_indices * 10000
    timesteps_exp = timesteps * step_to_sec

    # Set up for ffts:
    freq = fftfreq(len(timesteps_exp), d=(timesteps_exp[1] - timesteps_exp[0]))  # Frequency values
    fft_xlim_max = timesteps_exp[-1]/5

    # Identify type indices
    typeid_array = initial_frame.particles.typeid
    active_indices = np.where(typeid_array == 0)[0]
    n_active = len(active_indices)

    # Translational quantities (per particle)
    active_positions     = np.empty((num_frames, n_active, 3), dtype=float)
    active_images        = np.empty((num_frames, n_active, 3), dtype=float)
    active_unwrapped     = np.empty((num_frames, n_active, 3), dtype=float)
    active_velocity      = np.empty((num_frames, n_active, 3), dtype=float)
    active_orientation   = np.empty((num_frames, n_active, 4), dtype=float)

    # Per-particle displacement and distance tracking
    active_displacements = np.zeros((num_frames, n_active), dtype=float)
    active_total_distances = np.zeros((num_frames, n_active), dtype=float)

    # Per-particle rotational autocorrelation (for l = 2,4,6)
    active_racf = np.zeros((num_frames, n_active, 3), dtype=complex)

    # COM quantities
    active_com_unwrapped   = np.empty((num_frames, 3), dtype=float)
    active_com_velocity    = np.empty((num_frames, 3), dtype=float)
    active_com_v_norms     = np.empty((num_frames, 1), dtype=float)
    active_normalized_v    = np.empty((num_frames, 3), dtype=float)
    active_com_orientation = np.empty((num_frames, 4), dtype=float)
    com_racf               = np.zeros((num_frames, 3), dtype=complex)

    # Displacement tracking for COM
    disp_active_com = np.zeros(num_frames, dtype=float)
    total_distance_com = np.zeros(num_frames, dtype=float)

    previous_ori = False
    previous_com_ori = False

    cmap_blue = plt.get_cmap('Blues')
    norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

    # Initialize variables for displacement and distance calculations
    active_total_distance = 0

    disp_active = [0]  # Starting displacement is 0
    active_total_distances = [0]

    # Rotational Autocorrelation stuff
    #3 because I calculate it for l = 2,4,6 to get diff symmetry orders
    racf = np.empty((num_frames, n_active, 3), dtype=complex)
    previous_ori = 'FALSE'

    # Loop through trajectory
    for i, frame in enumerate(traj):
        pos = frame.particles.position
        img = frame.particles.image
        tid = frame.particles.typeid
        vel = frame.particles.velocity
        ori = frame.particles.orientation

        #timesteps.append(frame.configuration.step)

        # Active particles (typeid == 0)
        active_mask = (tid == 0)
        active_pos  = pos[active_mask]
        active_img  = img[active_mask]
        active_vel  = vel[active_mask]
        active_ori  = ori[active_mask]

        # Per-particle orientations, displacement and distance tracking
        active_positions[i]     = active_pos
        active_images[i]        = active_img
        active_unwrapped[i]     = box_instance.unwrap(active_pos, active_img)
        active_velocity[i]      = active_vel
        active_v_norm[i]        = np.linalg.norm(active_velocity[i])
        active_orientation[i]   = active_ori

        # Center of mass orientation, displacement, and distance tracking
        active_com_unwrapped[i] = np.mean(active_unwrapped[i], axis=0)
        active_com_velocity[i]  = np.mean(active_vel, axis=0)
        active_com_v_norms[i]   = np.linalg.norm(active_com_velocity[i])
        active_normalized_v[i]  = active_com_velocity[i] / active_com_v_norms[i]
        
        # COM orientation = mean quaternion
        if n_active > 1:
            mean_rot = Rot.from_quat(active_ori).mean()
            active_com_orientation[i] = mean_rot.as_quat()
        else:
            active_com_orientation[i] = active_ori[0]

        # Per-particle displacements
        if i > 0:
            disp_vectors = active_unwrapped[i] - active_unwrapped[0]
            active_displacements[i] = np.linalg.norm(disp_vectors, axis=1)

            step_vectors = active_unwrapped[i] - active_unwrapped[i - 1]
            step_lengths = np.linalg.norm(step_vectors, axis=1)
            active_total_distances[i] = active_total_distances[i - 1] + step_lengths

            # COM displacement
            disp_active_com[i] = np.linalg.norm(active_com_unwrapped[i] - active_com_unwrapped[0])
            step_vector_com = active_com_unwrapped[i] - active_com_unwrapped[i - 1]
            total_distance_com[i] = total_distance_com[i - 1] + np.linalg.norm(step_vector_com)
        
        
        # Rotational autocorrelation
        active_ori /= np.linalg.norm(active_ori, axis=1, keepdims=True)
        com_ori = active_com_orientation[i].reshape(1, 4)

        for j, l in enumerate([2, 4, 6]):
            rot_auto = freud.order.RotationalAutocorrelation(l)

            if previous_ori:
                rot_auto.compute(ref_orientations=prev_ori, orientations=active_ori)
                active_racf[i, :, j] = rot_auto.particle_order

            if previous_com_ori:
                rot_auto.compute(ref_orientations=prev_com_ori, orientations=com_ori)
                com_racf[i, j] = rot_auto.particle_order

        prev_ori = active_ori.copy()
        prev_com_ori = com_ori.copy()
        previous_ori = True
        previous_com_ori = True
    
    active_init_pos = active_com_unwrapped[0]
    disp_active = np.array(disp_active)
    active_total_distances = np.array(active_total_distances)

    print("Finished processing active particle COM positions, velocities, orientations, displacements, and total distances.")


    #############################################
    # Velocity and Orientation Analysis (COM)
    #############################################
    cmap_blue = plt.get_cmap('Blues')
    cmap_red = plt.get_cmap('Reds')
    norm_t = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())
    
    # Velocity-related plots
    plot_COM_velocity(
        timesteps,
        active_com_v_norms,
        speed_conv_mm,
        step_to_sec,
        job
    )

    active_v_fft = fft(active_com_v_norms)
    fft_active_com_v_data  = extract_FFT_data(active_v_fft, freq)
    plot_COM_velocity_FFT(freq, active_com_v_norms,fft_xlim_max, job)

    # Angular distributions of COM velocity and orientation
    plot_contrasted_active_angular_dist(active_normalized_v, active_com_orientation, cmap_blue, job)
    plot_v_distrib_on_S2(active_normalized_v, job)
    plot_2D_w_1D_velocoity_hist(
        active_normalized_v,
        'Distribution of Active Particle Velocity Direction',
        plt.cm.Blues,
        job
    )
   
    ##########################################################
    # Orientation Analysis (COM and per particle)
    ##########################################################

    # COM orientation analysis and plots
    fft_com_phi, fft_com_theta, fft_com_acf = analyze_orientation_series(
        active_com_orientation,
        timesteps,
        "COM",
        job
    )

    # Per-particle orientation analysis and plots
    for rod_id in [0, 1, 2, 3, 4]:
        analyze_orientation_series(
            active_particle_orientation[:, rod_id],
            timesteps,
            f"Particle_{rod_id}",
            job
    
    #############################################
    ## Plot and save x-y plane trajectories
    #############################################

    for rod_id in [0,1,2,3]:

        active_x = active_com_unwrapped[:, rod_id, 0]
        active_y = active_com_unwrapped[:, rod_id, 1]
        active_z = active_com_unwrapped[:, rod_id, 2]

        mid_color_active = cmap_blue(norm_t(timesteps[int(3*len(timesteps)/4)]))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_facecolor('white')

        sc_active = ax.scatter(active_x*len_conv_um, active_y*len_conv_um,
                            c=timesteps, cmap=cmap_blue,
                            norm=norm_t, label="Active Particle", s=9, alpha=0.7, edgecolor='none')


        cbar_com.set_label('Simulation Timestep')
        cbar_active = fig.colorbar(sc_active, orientation='vertical', fraction=0.05, pad=0.02)
        cbar_active.ax.yaxis.set_ticks([])

        legend_active = Line2D([0], [0], marker='o', color='w', markerfacecolor=mid_color_active, markersize=9, label="Active Particle    ")

        ax.legend(handles=[legend_active], loc='best')

        ax.set_xlabel(r"x position ($\mu$m)")
        ax.set_ylabel(r"y position ($\mu$m)")
        ax.set_title(f"x-y trajectory\nJob ID: {job}")

        ax.axis('equal')
        ax.grid()

        plt.show()
        fig.savefig(job.fn(f'xy_traj_particle_{rod_id}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print('XY Trajectories plotted.')

    #############################################
    ## Z-Position Analysis
    #############################################

    active_z_exp = []
    for i in range(len(active_z)):
        active_z_exp.append((active_z[i]+R+1)*len_conv_um)
    plot_z_position_data(active_z_exp, timesteps, step_to_sec, job)

    # FFT of inner profuct signals
    active_z_fft = fft(active_z_exp)
    active_z_fft_data  = extract_FFT_data(active_z_fft, freq)
    plot_z_position_FFT(freq, active_z_fft, fft_xlim_max, job)


    #############################################
    ## Active particle displacement
    #############################################

    disp_active_um = disp_active * len_conv_um
    active_total_distances_um = active_total_distances * len_conv_um
    mean_disp_um = disp_active_um.mean(axis=1)
    mean_total_distance_um = active_total_distances_um.mean(axis=1)
    std_total_distance_um = active_total_distances_um.std(axis=1)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timesteps, mean_total_distance_um, color='blue', label='Mean total distance')
    ax.fill_between(timesteps, mean_total_distance_um-std_total_distance_um,
                    mean_total_distance_um+std_total_distance_um, color='blue', alpha=0.2)
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Total Distance (um)")
    ax.grid(True)
    ax3 = ax.twiny()
    ax3.set_xlabel("Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick*step_to_sec:.2f}" for tick in ax.get_xticks()])
    fig.savefig(job.fn('total_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(timesteps_exp, mean_total_distance_um)
    r_squared = r_value**2

    # FFT of inner profuct signals
    active_disp_fft = fft(disp_active_um)

    ## Save to analysis
    active_disp_fft_data  = extract_FFT_data(active_disp_fft, freq)

    plot_displacement_FTT(timesteps, freq, active_disp_fft, step_to_sec, fft_xlim_max, job)

    #############################################
    ## Save results to json file
    #############################################

    all_data = {
        "jobid": job.id,
    }
    statepoints = job.sp
    all_data.update(statepoints)

    analysis_data = {
        "active_net_distance_um": float(mean_disp_um[-1]),
        "active_total_distance_um": float(mean_total_distance_um[-1]),
        "slope_dist_active_um_per_sec": float(slope),
        "slope_err_dist_active_um_per_sec": float(std_err),
        "r_squared_active": float(r_squared),
        "racf": racf.tolist(),
        "phi": phi.tolist(),
        "theta": theta.tolist(),
        "phi_fft": phi_fft.tolist(),
        "theta_fft": theta_fft.tolist(),
        "acf": acf.tolist(),
        "acf_fft": acf_fft.tolist()

    }
    all_data.update(analysis_data)
    with open(job.fn('analysis_data_wFFT.json'), 'w') as f:
        json.dump(all_data, f, indent=4)

    with open(job.fn('signac_job_document.json'), 'w') as f:
        json.dump(all_data, f, indent=4)

    print('Analysis complete.')


def Analysis(*jobs):
    
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
