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

'''
NOTE TO SELF: Some experimental y-axes are slightly wrong due to offset/shifts in plotting.
True as of 3/21/25.
'''


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

    time_indices = np.arange(len(traj))
    timesteps = time_indices * 10000
    timesteps_exp = timesteps * step_to_sec

    cmap_blue = plt.get_cmap('Blues')
    norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

    active_com_positions = []   # (N_frames, N_active, 3)
    active_com_velocities = []  # (N_frames, N_active, 3)
    active_orientations = []    # (N_frames, N_active, 4)

    for frame in traj:
        pos = frame.particles.position
        vel = frame.particles.velocity
        types = frame.particles.typeid
        orient = frame.particles.orientation

        mask_active = (types == 0)

        active_com_positions.append(pos[mask_active])
        active_com_velocities.append(vel[mask_active])
        active_orientations.append(orient[mask_active])

    active_com_positions = np.array(active_com_positions)
    active_com_velocities = np.array(active_com_velocities)
    active_orientations = np.array(active_orientations)

    print(f"Positions successfully read from GSD: {active_com_positions.shape[1]} active rods.")

    #############################################
    ## Export per-rod trajectory to HDF5
    #############################################

    out_fn = job.fn("rod_data.h5")

    with h5py.File(out_fn, 'w') as f:
        f.create_dataset("timesteps", data=timesteps)
        f.create_dataset("positions", data=active_com_positions)
        f.create_dataset("velocities", data=active_com_velocities)
        f.create_dataset("orientations", data=active_orientations)

        f.attrs["num_frames"] = active_com_positions.shape[0]
        f.attrs["num_rods"] = active_com_positions.shape[1]
        f.attrs["len_conv_um"] = len_conv_um
        f.attrs["speed_conv_mm"] = speed_conv_mm
        f.attrs["step_to_sec"] = step_to_sec
        f.attrs["description"] = (
            "Per-rod trajectory data: positions, velocities, and quaternions "
            "for each active rod at each frame."
        )

    print(f"Saved per-rod trajectories, velocities, and orientations --> {out_fn}")
    print(f"positions shape: {active_com_positions.shape}")

    #############################################
    ## Plot and save velocity magnitudes
    #############################################

    # Mean COM speed per frame
    active_com_v_norms = np.linalg.norm(active_com_velocities, axis=2).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, active_com_v_norms, label="Active rod", color='blue')
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Speed (sim units)")

    # experimental y-axis (mm/s)
    ax2 = ax.twinx()
    ax2.set_ylabel("Speed (mm/s)")
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f"{tick * speed_conv_mm:.2f}" for tick in ax.get_yticks()])

    # experimental x-axis (seconds)
    ax3 = ax.twiny()
    ax3.set_xlabel("Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])

    ax.legend(loc='best')
    ax.set_title("Average Rod C.O.M. Speed")
    ax.grid(True)
    fig.savefig(job.fn('velocity_magnitudes.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    #############################################
    ## Plot and save x-y trajectory for a single rod
    #############################################

    for i in [0,1,2,3]:
        rod_id = i  # which rod plotting
        active_x = active_com_positions[:, rod_id, 0]
        active_y = active_com_positions[:, rod_id, 1]

        mid_color_active = cmap_blue(norm(timesteps[int(3*len(timesteps)/4)]))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_facecolor('white')

        sc_active = ax.scatter(
            active_x * len_conv_um, active_y * len_conv_um,
            c=timesteps, cmap=cmap_blue, norm=norm,
            label="Active rod", s=9, alpha=0.7, edgecolor='none'
        )
        cbar_active = fig.colorbar(sc_active, orientation='vertical', fraction=0.05, pad=0.02)
        cbar_active.set_label('Simulation Timestep')

        ax.legend(handles=[Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=mid_color_active, markersize=9,
                                  label="Active rod")], loc='best')

        ax.set_xlabel(r"X-Position ($\mu$m)")
        ax.set_ylabel(r"Y-Position ($\mu$m)")
        ax.set_title(f"Rod {rod_id} X-Y Trajectory")
        ax.axis('equal')
        ax.grid()
        fig.savefig(job.fn(f'xy_traj_particle_{i}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    print('XY Trajectories plotted.')

    ## Plot distance above flat surface
    active_z = active_com_positions[:, :, 2].mean(axis=1)  # mean z over rods

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, (active_z+3/2)*len_conv_um, label="Active rod", color='blue', alpha=0.8)
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel(r"Z-Position ($\mu$m)")

    ax.set_xlim(left=0)
    ax.set_xlim(right=max(timesteps))

    ax.legend(loc='best')
    ax.set_title(f"Distance Above Flat Surface")

    ax3 = ax.twiny()
    ax3.set_xlabel(r"Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])

    ax.grid(True)
    fig.savefig(job.fn('d_from_floor.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    #############################################
    ## Displacement and total distance travelled
    #############################################

    n_frames, n_rods, _ = active_com_positions.shape

    # Initial positions per rod
    active_init_pos = active_com_positions[0]  # shape (n_rods, 3)

    # Storage arrays
    disp_active = np.zeros((n_frames, n_rods))        # displacement from initial pos
    active_total_distances = np.zeros((n_frames, n_rods))  # cumulative distance travelled

    # Total distance accumulator per rod
    total_distance_per_rod = np.zeros(n_rods)

    # Compute per-frame displacement and cumulative distance
    for t in range(1, n_frames):
        # Displacement from initial position
        disp = active_com_positions[t] - active_init_pos
        disp_active[t] = np.linalg.norm(disp, axis=1)

        # Incremental displacement between frames
        vector = active_com_positions[t] - active_com_positions[t-1]
        step_dist = np.linalg.norm(vector, axis=1)
        total_distance_per_rod += step_dist
        active_total_distances[t] = total_distance_per_rod

    print("Displacement calculated.")

    # Convert to experimental units (um)
    disp_active_um = disp_active * len_conv_um
    active_total_distances_um = active_total_distances * len_conv_um

    # Mean distance across rods
    mean_total_distance_um = active_total_distances_um.mean(axis=1)
    std_total_distance_um = active_total_distances_um.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')

    # Plot mean ± std
    ax.plot(timesteps, mean_total_distance_um, color='blue', label='Active rods (mean)')
    ax.fill_between(timesteps,
                    mean_total_distance_um - std_total_distance_um,
                    mean_total_distance_um + std_total_distance_um,
                    color='blue', alpha=0.2, label='±1 std')

    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel(r"Total Distance Travelled ($\mu$m)")

    # Experimental time axis
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])

    ax.legend(loc='best')
    ax.set_title("Total Distance Travelled by Active Rods")
    ax.grid(True)

    fig.savefig(job.fn('total_dist.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Total distance plot saved.")
    
    # Linear regression
    slope_active, intercept_active, r_value_active, p_value_active, std_err_active = linregress(
        timesteps_exp, mean_total_distance_um
    )
    r_squared_active = r_value_active ** 2

    # Plot with slope annotation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')

    ax.plot(timesteps, mean_total_distance_um, label="Active rods (mean)", color='blue', alpha=0.8)
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=0, top=max(mean_total_distance_um))

    # Annotate slope, R^2, and std error
    ax.text(
        0.05, 0.95,
        f"Slope (Active rods): {slope_active:.2e}" + r" ($\mu$m/sec)" +
        f"\nR²: {r_squared_active:.2f}\nStd Err: {std_err_active:.2e}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        color='blue'
    )

    # Twin x-axis for experimental time
    ax3 = ax.twiny()
    ax3.set_xlabel("Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])

    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel(r"Total Distance Travelled ($\mu$m)")
    ax.legend(loc='best')
    ax.set_title("Total Distance Travelled During Simulation (Linear Fit)")
    ax.grid(True)

    fig.savefig(job.fn('total_dist_Wslopes.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Total distance with regression slope plotted.")
    
    # Use mean displacement across all rods
    mean_disp_um = disp_active_um.mean(axis=1)
    std_disp_um = disp_active_um.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')

    # Plot mean +/- std
    ax.plot(timesteps, mean_disp_um, color='blue', label='Active rods (mean)')
    ax.fill_between(timesteps,
                    mean_disp_um - std_disp_um,
                    mean_disp_um + std_disp_um,
                    color='blue', alpha=0.2, label='±1 std')

    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel(r"Displacement from Starting Position ($\mu$m)")
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=0, top=max(mean_disp_um))

    # Twin x-axis for experimental time
    ax3 = ax.twiny()
    ax3.set_xlabel("Time (sec)")
    ax3.set_xticks(ax.get_xticks())
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])

    ax.legend(loc='best')
    ax.set_title("Center of Mass Displacement During Simulation")
    ax.grid(True)

    fig.savefig(job.fn('displacements.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print('Displacement plotted.')
        
    all_data = {
        "jobid": job.id,
    }
    statepoints = job.sp
    all_data.update(statepoints)

    analysis_data = {
        "active_net_distance_um": float(mean_disp_um[-1]),  # mean net displacement at final frame
        "active_total_distance_um": float(active_total_distances_um.mean(axis=1)[-1]),  # mean total distance
        "slope_dist_active_um_per_sec": float(slope_active),  # linear regression slope
        "slope_err_dist_active_um_per_sec": float(std_err_active),  # regression std err
        "r_squared_active": float(r_squared_active),
    }

    all_data.update(analysis_data)

    # Save both files
    for fn in ['analysis_data.json', 'signac_job_document.json']:
        with open(job.fn(fn), 'w') as f:
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
