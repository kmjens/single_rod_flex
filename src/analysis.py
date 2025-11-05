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
    ## Load GSD trajectory
    #############################################
    f = gsd.pygsd.GSDFile(open(job.fn('Run.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)
    num_frames = len(traj)
    
    initial_frame = traj[0]
    typeid_array = initial_frame.particles.typeid
    active_indices = np.where(typeid_array == 0)[0]
    n_active = len(active_indices)
    particle_indices = list(range(min(5, n_active)))  # first 5 particles for plotting
    
    box = initial_frame.configuration.box
    box_instance = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2],
                                 xy=box[3], xz=box[4], yz=box[5])
    
    timesteps = np.arange(num_frames) * 10000
    timesteps_exp = timesteps * step_to_sec
    freq = fftfreq(num_frames, d=(timesteps_exp[1] - timesteps_exp[0]))
    fft_xlim_max = timesteps_exp[-1]/5
    
    #############################################
    ## Allocate arrays
    #############################################
    # Per-particle quantities
    active_positions   = np.empty((num_frames, n_active, 3))
    active_unwrapped   = np.empty((num_frames, n_active, 3))
    active_velocity    = np.empty((num_frames, n_active, 3))
    active_orientation = np.empty((num_frames, n_active, 4))
    
    active_displacements    = np.zeros((num_frames, n_active))
    active_total_distances  = np.zeros((num_frames, n_active))
    active_racf             = np.zeros((num_frames, n_active, 3), dtype=complex)
    
    # COM quantities
    com_positions        = np.empty((num_frames, 3))
    com_velocity         = np.empty((num_frames, 3))
    com_v_norms          = np.empty(num_frames)
    com_normalized_v     = np.empty((num_frames, 3))
    com_orientation      = np.empty((num_frames, 4))
    com_racf             = np.zeros((num_frames, 3), dtype=complex)
    
    # Temporary variables
    prev_ori = None
    prev_com_ori = None
    
    #############################################
    ## Loop over trajectory frames
    #############################################
    for i, frame in enumerate(traj):
        pos = frame.particles.position[active_indices]
        img = frame.particles.image[active_indices]
        vel = frame.particles.velocity[active_indices]
        ori = frame.particles.orientation[active_indices]
        
        # Store per-particle data
        active_positions[i]   = pos
        active_unwrapped[i]   = box_instance.unwrap(pos, img)
        active_velocity[i]    = vel
        active_orientation[i] = ori
        
        # Compute COM
        com_positions[i] = np.mean(active_unwrapped[i], axis=0)
        com_velocity[i]  = np.mean(vel, axis=0)
        com_v_norms[i]   = np.linalg.norm(com_velocity[i])
        com_normalized_v[i] = com_velocity[i] / com_v_norms[i] if com_v_norms[i] > 0 else np.zeros(3)
        
        if n_active > 1:
            com_orientation[i] = Rot.from_quat(ori).mean().as_quat()
        else:
            com_orientation[i] = ori[0]
        
        # Displacements & total distances
        if i > 0:
            step_vectors = active_unwrapped[i] - active_unwrapped[i-1]
            step_lengths = np.linalg.norm(step_vectors, axis=1)
            active_total_distances[i] = active_total_distances[i-1] + step_lengths
            active_displacements[i]   = np.linalg.norm(active_unwrapped[i] - active_unwrapped[0], axis=1)
            
            com_step = com_positions[i] - com_positions[i-1]
            com_racf_step = np.linalg.norm(com_step)
        
        # Rotational autocorrelation
        ori_normed = ori / np.linalg.norm(ori, axis=1, keepdims=True)
        com_ori_reshaped = com_orientation[i].reshape(1,4)
        
        for j, l in enumerate([2,4,6]):
            rot_auto = freud.order.RotationalAutocorrelation(l)
            if prev_ori is not None:
                rot_auto.compute(ref_orientations=prev_ori, orientations=ori_normed)
                active_racf[i,:,j] = rot_auto.particle_order
            if prev_com_ori is not None:
                rot_auto.compute(ref_orientations=prev_com_ori, orientations=com_ori_reshaped)
                com_racf[i,j] = rot_auto.particle_order
        
        prev_ori = ori_normed.copy()
        prev_com_ori = com_ori_reshaped.copy()
    
    #############################################
    ## Displacements & FFTs
    #############################################
    # COM
    disp_com_um = np.linalg.norm(com_positions - com_positions[0], axis=1) * len_conv_um
    total_dist_com_um = np.zeros(num_frames)
    for i in range(1,num_frames):
        total_dist_com_um[i] = total_dist_com_um[i-1] + np.linalg.norm(com_positions[i]-com_positions[i-1]) * len_conv_um
    
    # Per-particle
    disp_particles_um = np.linalg.norm(active_unwrapped - active_unwrapped[0], axis=2) * len_conv_um
    total_dist_particles_um = np.zeros_like(disp_particles_um)
    for i in range(1,num_frames):
        total_dist_particles_um[i] = total_dist_particles_um[i-1] + np.linalg.norm(active_unwrapped[i]-active_unwrapped[i-1], axis=1) * len_conv_um
    
    #############################################
    ## Plot first 5 particles + COM
    #############################################
    for pid in particle_indices:
        fig, ax = plt.subplots()
        ax.plot(timesteps_exp, disp_particles_um[:,pid], label=f'Particle {pid} Displacement')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (um)")
        ax.grid(True)
        fig.savefig(job.fn(f'displacement_particle_{pid}.png'), dpi=150)
        plt.close()
    
    # COM plot
    fig, ax = plt.subplots()
    ax.plot(timesteps_exp, disp_com_um, label='COM Displacement', color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (um)")
    ax.grid(True)
    fig.savefig(job.fn(f'displacement_COM.png'), dpi=150)
    plt.close()
    
    #############################################
    ## FFT example (COM)
    #############################################
    fft_com = np.abs(fft(disp_com_um))[:num_frames//2]
    fft_freqs = fftfreq(num_frames, step_to_sec)[:num_frames//2]
    fig, ax = plt.subplots()
    ax.plot(fft_freqs, fft_com, label='COM FFT')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    fig.savefig(job.fn('fft_COM.png'), dpi=150)
    plt.close()
    
    # Per-particle FFT
    for pid in particle_indices:
        fft_particle = np.abs(fft(disp_particles_um[:,pid]))[:num_frames//2]
        fig, ax = plt.subplots()
        ax.plot(fft_freqs, fft_particle, label=f'Particle {pid} FFT')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        fig.savefig(job.fn(f'fft_particle_{pid}.png'), dpi=150)
        plt.close()
    
    #############################################
    ## COM and per-particle trajectory XY plots
    #############################################
    fig, ax = plt.subplots()
    ax.plot(com_positions[:,0]*len_conv_um, com_positions[:,1]*len_conv_um, label='COM', color='blue')
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.axis('equal')
    ax.grid(True)
    fig.savefig(job.fn('xy_traj_COM.png'), dpi=150)
    plt.close()
    
    for pid in particle_indices:
        fig, ax = plt.subplots()
        ax.plot(active_unwrapped[:,pid,0]*len_conv_um,
                active_unwrapped[:,pid,1]*len_conv_um,
                label=f'Particle {pid}')
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        ax.axis('equal')
        ax.grid(True)
        fig.savefig(job.fn(f'xy_traj_particle_{pid}.png'), dpi=150)
        plt.close()
    
    print("Analysis complete for COM and first 5 particles.")
    
    #############################################
    ## Save results to JSON
    #############################################

    all_data = {
        "jobid": job.id,
    }
    all_data.update(job.sp)

    # Compute average displacement & total distance over all active particles
    mean_disp_particles_um = np.mean(disp_particles_um, axis=1)        # mean over particles
    mean_total_dist_particles_um = np.mean(total_dist_particles_um, axis=1)

    analysis_data = {
        # COM quantities
        "com_net_displacement_um": float(disp_com_um[-1]),
        "com_total_distance_um": float(total_dist_com_um[-1]),

        # Average over all particles
        "mean_particle_net_displacement_um": float(mean_disp_particles_um[-1]),
        "mean_particle_total_distance_um": float(mean_total_dist_particles_um[-1]),

        # Individual first 5 particle final displacements
        "particle_net_displacement_um": disp_particles_um[-1, :5].tolist(),
        "particle_total_distance_um": total_dist_particles_um[-1, :5].tolist(),

        # You can also save RACF, orientations, FFTs
        "racf_particles": active_racf.tolist(),
        "racf_com": com_racf.tolist(),
        "com_orientation": com_orientation.tolist(),
        "particle_orientation": active_orientation.tolist()
    }

    all_data.update(analysis_data)

    with open(job.fn('analysis_data_wFFT.json'), 'w') as f:
        json.dump(all_data, f, indent=4)

    with open(job.fn('signac_job_document.json'), 'w') as f:
        json.dump(all_data, f, indent=4)

    print('Analysis complete and saved.')

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
