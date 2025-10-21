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

from utility import *
from matplotlib.lines import Line2D
from scipy.stats import linregress
from scipy.fft import fft, fftfreq
from scipy.spatial.transform import Rotation as Rot
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
    ## Open trajectory
    #############################################
    
    f = gsd.pygsd.GSDFile(open(job.fn('Run.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)
    n_frames = len(traj)
    initial_frame = traj[0]

    rod_indices = np.where(initial_frame.particles.typeid == 0)[0]
    n_rods = len(rod_indices)

    # Box for unwrapping
    box = initial_frame.configuration.box
    box_instance = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2],
                                 xy=box[3], xz=box[4], yz=box[5])

    #############################################
    # Allocate arrays
    #############################################
    
    active_com_positions = np.empty((n_frames, n_rods, 3))
    active_com_velocities = np.empty((n_frames, n_rods, 3))
    active_com_orientation = np.empty((n_frames, 4))  # mean quaternion
    normalized_velocities = np.empty((n_frames, 3))
    total_distance_per_rod = np.zeros(n_rods)
    disp_active = np.zeros((n_frames, n_rods))
    active_total_distances = np.zeros((n_frames, n_rods))

    racf = np.empty((n_frames, 3), dtype=complex)
    prev_ori_exists = False

    #############################################
    # Time arrays
    #############################################
    
    timesteps = np.arange(n_frames) * 10000
    timesteps_exp = timesteps * step_to_sec
    freq = fftfreq(len(timesteps_exp), d=(timesteps_exp[1]-timesteps_exp[0]))
    fft_xlim_max = timesteps_exp[-1]/5
    norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())
    cmap_blue = plt.get_cmap('Blues')

    #############################################
    # Loop over frames
    #############################################
    
    for i, frame in enumerate(traj):
        pos = frame.particles.position[rod_indices]
        img = frame.particles.image[rod_indices]
        vel = frame.particles.velocity[rod_indices]
        ori = frame.particles.orientation[rod_indices]

        unwrapped_pos = box_instance.unwrap(pos, img)

        active_com_positions[i] = unwrapped_pos
        active_com_velocities[i] = vel
        normalized_velocities[i] = vel.mean(axis=0) / (np.linalg.norm(vel.mean(axis=0)) + 1e-16)

        # Mean quaternion
        if len(ori) > 1:
            r = Rot.from_quat(ori)
            active_com_orientation[i] = r.mean().as_quat()
        else:
            active_com_orientation[i] = ori[0]

        # Displacement and total distance
        if i > 0:
            disp = unwrapped_pos - active_com_positions[0]
            disp_active[i] = np.linalg.norm(disp, axis=1)

            vector = unwrapped_pos - active_com_positions[i-1]
            step_dist = np.linalg.norm(vector, axis=1)
            total_distance_per_rod += step_dist
            active_total_distances[i] = total_distance_per_rod
    
        # Rotational autocorrelation
        ori_norm = ori / np.linalg.norm(ori, axis=1, keepdims=True)
        for j, l in enumerate([2,4,6]):
            rac = freud.order.RotationalAutocorrelation(l)
            if prev_ori_exists:
                rac.compute(ref_orientations=prev_ori, orientations=ori_norm)
                racf[i, j] = rac.particle_order.mean()   # <-- use mean
        prev_ori = ori_norm
        prev_ori_exists = True

    #############################################
    # Orientation spherical coordinates
    #############################################
    
    quats = active_com_orientation / np.linalg.norm(active_com_orientation, axis=1, keepdims=True)
    local_dir = np.tile([1,0,0], 1)
    directions = np.array([Rot.from_quat([q[1],q[2],q[3],q[0]]).apply(local_dir) for q in quats])
    x, y, z = directions[:,0], directions[:,1], directions[:,2]
    theta = np.arccos(z)
    phi = np.unwrap(np.arctan2(y, x))

    #############################################
    # FFTs
    #############################################
    
    phi_fft = fft(np.degrees(phi))
    theta_fft = fft(np.degrees(theta))
    acf = quaternion_acf(active_com_orientation)
    acf_fft = fft(acf)

    #############################################
    # Save per-rod trajectory to HDF5
    #############################################
    
    out_fn = job.fn("rod_data.h5")
    with h5py.File(out_fn, 'w') as f:
        f.create_dataset("timesteps", data=timesteps)
        f.create_dataset("positions", data=active_com_positions)
        f.create_dataset("velocities", data=active_com_velocities)
        f.create_dataset("orientations", data=active_com_orientation)
        f.attrs.update({
            "num_frames": n_frames,
            "num_rods": n_rods,
            "len_conv_um": len_conv_um,
            "speed_conv_mm": speed_conv_mm,
            "step_to_sec": step_to_sec,
            "description": "Per-rod trajectory data: positions, velocities, quaternions."
        })
    print(f"Saved per-rod trajectories, velocities, and orientations --> {out_fn}")

    #############################################
    # Velocity magnitudes (mean COM)
    #############################################
    
    active_com_v_norms = np.linalg.norm(active_com_velocities, axis=2).mean(axis=1)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timesteps, active_com_v_norms, color='blue', label='Active rod')
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Speed (sim units)")
    ax2 = ax.twinx()
    ax2.set_ylabel("Speed (mm/s)")
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f"{tick*speed_conv_mm:.2f}" for tick in ax.get_yticks()])
    ax3 = ax.twiny()
    ax3.set_xlabel("Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick*step_to_sec:.2f}" for tick in ax.get_xticks()])
    ax.legend(loc='best')
    ax.set_title("Average Rod C.O.M. Speed")
    fig.savefig(job.fn('velocity_magnitudes.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #############################################
    # XY Trajectories (first 4 rods)
    #############################################
    
    for i in range(min(4, n_rods)):
        rod_x = active_com_positions[:,i,0]
        rod_y = active_com_positions[:,i,1]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(rod_x*len_conv_um, rod_y*len_conv_um, color='blue', alpha=0.5, lw=1)
        sc = ax.scatter(rod_x*len_conv_um, rod_y*len_conv_um, c=timesteps,
                        cmap=cmap_blue, norm=norm, s=9, alpha=0.7, edgecolor='none')
        cbar = fig.colorbar(sc)
        cbar.set_label("Simulation Timestep")
        ax.set_xlabel("X-Position (um)")
        ax.set_ylabel("Y-Position (um)")
        ax.axis('equal'); ax.grid(True)
        fig.savefig(job.fn(f'xy_traj_particle_{i}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    #############################################
    # Z position (mean over rods)
    #############################################
    
    active_z_mean = active_com_positions[:,:,2].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(timesteps, (active_z_mean + 3/2)*len_conv_um, color='blue', alpha=0.8)
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Z-Position (um)")
    ax.grid(True)
    ax3 = ax.twiny()
    ax3.set_xlabel("Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick*step_to_sec:.2f}" for tick in ax.get_xticks()])
    fig.savefig(job.fn('d_from_floor.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #############################################
    # Displacement and total distance
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

    #############################################
    # Save JSON with new metrics
    #############################################
    
    all_data = {"jobid": job.id}; all_data.update(job.sp)
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

    with open(job.fn('analysis_active_only.json'), 'w') as f:
        json.dump(all_data, f, indent=4)

    print("Analysis complete. JSON and plots saved.")


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
