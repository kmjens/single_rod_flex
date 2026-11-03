import argparse
import os

import hoomd
import signac
import matplotlib
matplotlib.use('Agg')  # no display on the cluster

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Circle

from utility import *


## Color Maps
# GUV in lab frame: jet (blue = beginning, red = end)
CMAP_GUV = plt.get_cmap('jet')
# Rod in GUV frame: grey (black = beginning, white = end) ending at 0.85 so final point still visible
CMAP_ROD = ListedColormap(plt.get_cmap('gray')(np.linspace(0.0, 0.85, 256)))


def plot_xy_traj(xy, t, cmap, t_label, len_label, title, out_file,
                 guv_radius=None):
    """Scatter an x-y trajectory colored by time and save it.

    xy         : (n_frames, 3) or (n_frames, 2) positions
    t          : (n_frames,) time used for coloring
    guv_radius : if given, draw the GUV outline (circle centered on origin)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('white')

    sc = ax.scatter(xy[:, 0], xy[:, 1],
                    c=t, cmap=cmap, norm=Normalize(vmin=t.min(), vmax=t.max()),
                    s=9, alpha=0.7, edgecolor='none')

    if guv_radius is not None:
        ax.add_patch(Circle((0, 0), guv_radius, fill=False,
                            ls='--', lw=1, color='tab:blue', label='GUV (mean radius)'))
        ax.plot(0, 0, '+', color='tab:blue', ms=8)
        ax.legend(loc='upper right')

    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label(t_label)

    ax.set_xlabel(f"X-Position ({len_label})")
    ax.set_ylabel(f"Y-Position ({len_label})")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)

    fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # free memory when looping over many jobs


def PlotTraj_implementation(job, communicator):
    SP = JobParser(job)
    data = np.load(job.fn('extracted_timestep_data.npz'))

    timesteps = data['timesteps']
    time_exp = data['time_exp']

    # Mean GUV radius: distance of mesh particles from mesh COM, averaged over all frames
    mesh_rel = data['all_mesh_positions'] - data['mesh_com_positions'][:, None, :]
    guv_radius = float(np.mean(np.linalg.norm(mesh_rel, axis=-1)))
    guv_radius_exp = guv_radius * float(data['len_conv_um'])

    #############################################
    ## Plot GUV in Lab Frame
    #############################################
    # Shift so the trajectory starts at the origin
    mesh_com = data['mesh_com_positions'] - data['mesh_com_positions'][0]
    mesh_com_exp = data['mesh_com_positions_exp'] - data['mesh_com_positions_exp'][0]

    ## Plot in sim units
    plot_xy_traj(mesh_com, timesteps, CMAP_GUV,
                 t_label='Simulation Timestep',
                 len_label='sim units',
                 title='GUV Center of Mass Trajectory (Lab Frame)',
                 out_file=job.fn('GUV_traj_lab_frame_sim.png'))

    ## Plot in exp units
    plot_xy_traj(mesh_com_exp, time_exp, CMAP_GUV,
                 t_label='Time (s)',
                 len_label=r'$\mu$m',
                 title='GUV Center of Mass Trajectory (Lab Frame)',
                 out_file=job.fn('GUV_traj_lab_frame_exp.png'))

    #############################################
    ## Plot Rod in GUV frame
    #############################################
    # Rod COM relative to mesh COM (GUV center at origin)
    rod_rel = data['active_rel_mesh_positions']
    rod_rel_exp = data['active_rel_mesh_positions_exp']

    ## Plot in sim units
    plot_xy_traj(rod_rel, timesteps, CMAP_ROD,
                 t_label='Simulation Timestep',
                 len_label='sim units',
                 title='Rod Trajectory (GUV Frame)',
                 out_file=job.fn('Rod_traj_GUV_frame_sim.png'),
                 guv_radius=guv_radius)

    ## Plot in exp units
    plot_xy_traj(rod_rel_exp, time_exp, CMAP_ROD,
                 t_label='Time (s)',
                 len_label=r'$\mu$m',
                 title='Rod Trajectory (GUV Frame)',
                 out_file=job.fn('Rod_traj_GUV_frame_exp.png'),
                 guv_radius=guv_radius_exp)

    print(f"Trajectory plots saved for job {job.id}")


def PlotTraj(*jobs):
    communicator = hoomd.communicator.Communicator()
    PlotTraj_implementation(jobs[communicator.partition], communicator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)