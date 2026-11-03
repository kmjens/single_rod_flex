import signac
import argparse
import freud
import hoomd
import json
import matplotlib
import os
import signac

import gsd
import gsd.hoomd
import gsd.pygsd

import numpy as np
import matplotlib.pyplot as plt

from utility import *
from matplotlib.lines import Line2D
from scipy.stats import linregress

# Sim -> experimental unit conversions (same values as BuoyancyAndGravity in utility.py)
MASS_CONV   = 7.863e-14       # kg per sim mass unit (approx mass of one rod)
LEN_CONV    = 3e-6            # m per sim length unit (3 um, approx rod width)
ENERGY_CONV = 2.0709735e-20   # J per sim energy unit (5 kT)
TIME_CONV   = LEN_CONV * np.sqrt(MASS_CONV / ENERGY_CONV)   # s per sim time unit (~0.005846)
LEN_CONV_UM = LEN_CONV * 1e6  # um per sim length unit
STEPS_PER_FRAME = 10000       # GSD write period (timesteps per frame)

def compute_msd_crg(mesh_com, active_com):
    """Mesh-COM MSD and rod-GUV displacement coupling C_RG for every frame lag.

    For lag k (k >= 1), using all time origins:
        dG = mesh_com[k:] - mesh_com[:-k]
        dR = active_com[k:] - active_com[:-k]
        MSD(k)  = < |dG|^2 >
        C_RG(k) = sum(dG . dR) / sqrt( sum|dG|^2 * sum|dR|^2 )
    Lag 0 is left as NaN, matching the notebook.
    """
    # float64 so small displacements are not lost to float32 rounding
    G = np.asarray(mesh_com, dtype=np.float64)
    R = np.asarray(active_com, dtype=np.float64)
    n_frames = len(G)

    msd = np.full(n_frames, np.nan)
    crg = np.full(n_frames, np.nan)
    n_pairs = np.zeros(n_frames, dtype=np.int64)

    for lag in range(1, n_frames):
        dG = G[lag:] - G[:-lag]
        dR = R[lag:] - R[:-lag]
        sum_GG = np.sum(dG * dG)
        sum_RR = np.sum(dR * dR)

        n_pairs[lag] = len(dG)
        msd[lag] = sum_GG / len(dG)
        den = np.sqrt(sum_GG * sum_RR)
        if den > 0:
            crg[lag] = np.sum(dG * dR) / den

    return msd, crg, n_pairs

def CollectData_implementation(job, communicator):
    #############################################
    ## Unit Conversions
    #############################################
    SP = JobParser(job)

    f = gsd.pygsd.GSDFile(open(job.fn('Run.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)

    time_indices = np.arange(len(traj))

    timesteps = time_indices * STEPS_PER_FRAME           # simulation timesteps
    time_exp = timesteps * SP.dt * TIME_CONV             # seconds
    L = np.float32(LEN_CONV_UM)                          # um per sim length
    V = np.float32(LEN_CONV_UM / TIME_CONV)              # (um/s) per (sim length / sim time)

    # all positions of every particle over time
    all_mesh_positions = []
    all_active_positions = []
    all_const_positions = []

    # positions of coms over time
    mesh_com_positions   = []
    mesh_com_velocities  = []
    active_com_positions = []
    active_orientations  = []
    active_com_velocities = []

    for frame in traj:

        positions = frame.particles.position
        velocities = frame.particles.velocity
        typeids = frame.particles.typeid
        orientations = frame.particles.orientation
        images = frame.particles.image

        # Box dimensions
        box = frame.configuration.box
        Lx = box[0]
        Ly = box[1]
        Lz = box[2]

        # Unwrap all particle positions
        unwrapped_positions = positions + images * np.array([Lx, Ly, Lz])

        # Mesh particles
        mask = (typeids == 0)
        mesh_positions = unwrapped_positions[mask]
        all_mesh_positions.append(mesh_positions)
        mesh_com_positions.append(np.average(mesh_positions, axis=0))

        mesh_velocities = velocities[mask]
        mesh_com_velocities.append(np.average(mesh_velocities, axis=0))


        # Active rods
        mask = (typeids == 1)
        active_positions = unwrapped_positions[mask]
        all_active_positions.append(active_positions)
        active_com_positions.append(np.average(active_positions, axis=0)) #works for multiple rods
        active_velocities = velocities[mask]
        active_com_velocities.append(np.average(active_velocities, axis=0))

        active_orientations.append(orientations[mask]) #only works for 1 active rod right now


        # Rod constituents
        mask = (typeids == 2)
        const_positions = unwrapped_positions[mask]
        all_const_positions.append(const_positions)

    print("Positions taken successfully from GSD.")

    # Convert remaining lists to NumPy arrays
    all_mesh_positions = np.asarray(all_mesh_positions, dtype=np.float32)
    all_active_positions = np.asarray(all_active_positions, dtype=np.float32)
    all_const_positions = np.asarray(all_const_positions, dtype=np.float32)

    all_const_positions = np.asarray(all_const_positions, dtype=np.float32)

    mesh_com_positions = np.asarray(mesh_com_positions, dtype=np.float32)
    mesh_com_velocities = np.asarray(mesh_com_velocities, dtype=np.float32)

    active_com_positions = np.asarray(active_com_positions, dtype=np.float32)
    active_orientations = np.asarray(active_orientations, dtype=np.float32)
    active_com_velocities = np.asarray(active_com_velocities, dtype=np.float32)

    #############################################
    ## Displacements
    #############################################
    # COM displacement from the first frame
    mesh_com_displacements   = mesh_com_positions - mesh_com_positions[0]
    active_com_displacements = active_com_positions - active_com_positions[0]
    # rod COM relative to mesh COM (rod position in the GUV frame)
    active_rel_mesh_positions = active_com_positions - mesh_com_positions

    #############################################
    ## MSD and C_RG
    #############################################
    msd, crg, n_pairs = compute_msd_crg(mesh_com_positions, active_com_positions)

    lag_frames = np.arange(len(timesteps))
    lag_time = timesteps - timesteps[0]                  # lag in simulation timesteps
    lag_time_exp = lag_time * SP.dt * TIME_CONV          # lag in seconds
 
    ## Save in one compressed file
    # Sim units: lengths in sim length, velocities in sim length / sim time, time in timesteps.
    # *_exp: lengths in um, velocities in um/s, time in s, MSD in um^2. C_RG is dimensionless.
    output_file = job.fn('extracted_timestep_data.npz')

    np.savez_compressed(
        output_file,
        # time
        timesteps=timesteps,
        time_exp=time_exp,

        # per-particle positions
        all_mesh_positions=all_mesh_positions,
        all_active_positions=all_active_positions,
        all_const_positions=all_const_positions,
        all_mesh_positions_exp=all_mesh_positions * L,
        all_active_positions_exp=all_active_positions * L,
        all_const_positions_exp=all_const_positions * L,

        # COM positions
        mesh_com_positions=mesh_com_positions,
        active_com_positions=active_com_positions,
        active_rel_mesh_positions=active_rel_mesh_positions,
        mesh_com_positions_exp=mesh_com_positions * L,
        active_com_positions_exp=active_com_positions * L,
        active_rel_mesh_positions_exp=active_rel_mesh_positions * L,

        # COM displacements from frame 0
        mesh_com_displacements=mesh_com_displacements,
        active_com_displacements=active_com_displacements,
        mesh_com_displacements_exp=mesh_com_displacements * L,
        active_com_displacements_exp=active_com_displacements * L,

        # COM velocities
        mesh_com_velocities=mesh_com_velocities,
        active_com_velocities=active_com_velocities,
        mesh_com_velocities_exp=mesh_com_velocities * V,
        active_com_velocities_exp=active_com_velocities * V,

        # orientation (unitless quaternions)
        active_orientations=active_orientations,

        # MSD and C_RG vs lag (index = lag in frames; lag 0 is NaN)
        lag_frames=lag_frames,
        lag_time=lag_time,
        lag_time_exp=lag_time_exp,
        MSD=msd,
        MSD_exp=msd * LEN_CONV_UM**2,
        CRG=crg,
        n_pairs=n_pairs,

        # conversion factors used
        len_conv_um=LEN_CONV_UM,
        time_conv_s=TIME_CONV,
        dt=SP.dt,
        steps_per_frame=STEPS_PER_FRAME,
    )

    print(f"Extracted data (timesteps, MSD, C_RG, exp units) saved to: {output_file}")

def CollectData(*jobs):
    processes_per_directory = os.environ['ACTION_PROCESSES_PER_DIRECTORY']
    communicator = hoomd.communicator.Communicator()
    CollectData_implementation(jobs[communicator.partition], communicator)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)