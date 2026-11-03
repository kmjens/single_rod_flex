import argparse
import os

import hoomd
import signac
import matplotlib
matplotlib.use('Agg')  # no display on the cluster

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utility import *

# Mesh particles used to define the local membrane plane at the rod tip.
# 3 = triangle of the 3 closest particles (original notebook method).
# ~10 gives a much less noisy normal; see notes.
N_NEIGHBORS = 3


def rod_axis_from_quaternion(q):
    """Rotate the body-frame x axis by quaternions q = (w, x, y, z), shape (n, 4).

    The rod's constituent beads lie along body x (-1.5 ... +1.5), so this is the
    unit vector from the rod COM toward the front bead.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    v = np.stack([1 - 2 * (y * y + z * z),
                  2 * (x * y + w * z),
                  2 * (x * z - w * y)], axis=1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def local_mesh_normals(front_bead, mesh_positions, n_neighbors=3):
    """Local membrane normal at the front bead, from its closest mesh particles.

    Fits a plane (SVD) through the n_neighbors mesh particles closest to the
    front bead. With n_neighbors=3 this is exactly the triangle normal used in
    the notebook; larger values average over membrane noise.

    front_bead     : (n_frames, 3)
    mesh_positions : (n_frames, n_mesh, 3)
    Returns normals (n_frames, 3) oriented toward the front bead, and the
    indices of the closest mesh particles (n_frames, n_neighbors).
    """
    n_frames = len(front_bead)
    normals = np.full((n_frames, 3), np.nan)
    closest = np.zeros((n_frames, n_neighbors), dtype=np.int64)

    for t in range(n_frames):
        mesh = mesh_positions[t].astype(np.float64)
        bead = front_bead[t].astype(np.float64)

        d2 = np.sum((mesh - bead) ** 2, axis=1)
        idx = np.argpartition(d2, n_neighbors)[:n_neighbors]
        idx = idx[np.argsort(d2[idx])]
        closest[t] = idx

        pts = mesh[idx]
        center = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - center)
        if sv[1] == 0:          # degenerate (collinear) points
            continue
        n = vt[-1]              # direction of least spread = plane normal

        # orient normal toward the front bead
        if np.dot(n, bead - center) < 0:
            n = -n
        normals[t] = n

    return normals, closest


def plot_angle_vs_time(t, angle, t_label, out_file):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, angle, lw=0.8, color='k')
    ax.set_xlabel(t_label)
    ax.set_ylabel(r'Rod angle to membrane ($^\circ$)')
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.grid(alpha=0.3)
    fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_angle_dist(angle, out_file):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(angle[np.isfinite(angle)], bins=30, range=(0, 90), density=True,
            alpha=0.5, edgecolor='black', label='Simulation')
    ax.set_xlabel(r'Rod angle to membrane ($^\circ$)')
    ax.set_ylabel('Probability density')
    ax.set_xlim(0, 90)
    ax.legend()
    fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def AngleAnalysis_implementation(job, communicator):
    #############################################
    ## Collect Data
    #############################################
    npz_file = job.fn('extracted_timestep_data.npz')
    with np.load(npz_file) as z:
        data = {k: z[k] for k in z.files}   # load fully so the file can be rewritten

    timesteps = data['timesteps']     # simulation timesteps
    time_exp = data['time_exp']       # seconds

    all_mesh_positions = data['all_mesh_positions']      # (n_frames, n_mesh, 3)
    all_const_positions = data['all_const_positions']    # (n_frames, n_const, 3)
    active_orientations = data['active_orientations']    # (n_frames, n_rods, 4)

    #############################################
    ## Rod orientation and local membrane normal
    #############################################
    # Front bead = last constituent (body-frame +x end); single rod
    front_bead_idx = all_const_positions.shape[1] - 1
    front_bead = all_const_positions[:, front_bead_idx, :]

    rod_axis = rod_axis_from_quaternion(active_orientations[:, 0, :].astype(np.float64))

    mesh_normals, closest_mesh_indices = local_mesh_normals(front_bead, all_mesh_positions,
                                                             n_neighbors=N_NEIGHBORS)

    # Angle between rod axis and the local membrane plane:
    # 0 deg = rod lies flat along the membrane, 90 deg = rod points straight into it.
    # (= 90 - angle between rod and membrane normal, as in the notebook)
    cos_to_normal = np.clip(np.abs(np.sum(rod_axis * mesh_normals, axis=1)), 0.0, 1.0)
    angle_deg = 90.0 - np.degrees(np.arccos(cos_to_normal))

    finite = np.isfinite(angle_deg)
    angle_mu, angle_sigma = norm.fit(angle_deg[finite])

    #############################################
    ## Save into extracted_timestep_data.npz
    #############################################
    data.update(
        rod_axis=rod_axis.astype(np.float32),
        mesh_normals=mesh_normals.astype(np.float32),
        closest_mesh_indices=closest_mesh_indices,
        angle_deg=angle_deg,
        angle_mu=angle_mu,
        angle_sigma=angle_sigma,
        angle_n_neighbors=N_NEIGHBORS,
    )
    # write to a temp file first so a crash can't corrupt the original
    tmp_file = job.fn('extracted_timestep_data.tmp.npz')
    np.savez_compressed(tmp_file, **data)
    os.replace(tmp_file, npz_file)

    #############################################
    ## Plot
    #############################################
    ## Sim units
    plot_angle_vs_time(timesteps, angle_deg, 'Simulation Timestep',
                       job.fn('rod_membrane_angle_sim.png'))

    ## Exp units
    plot_angle_vs_time(time_exp, angle_deg, 'Time (s)',
                       job.fn('rod_membrane_angle_exp.png'))

    ## Distribution (same in either unit system)
    plot_angle_dist(angle_deg, job.fn('rod_membrane_angle_dist.png'))

    print(f"Angle analysis saved for job {job.id}: "
          f"mean = {angle_mu:.1f} deg, std = {angle_sigma:.1f} deg, "
          f"{np.sum(~finite)} degenerate frames")


def AngleAnalysis(*jobs):
    communicator = hoomd.communicator.Communicator()
    AngleAnalysis_implementation(jobs[communicator.partition], communicator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)
