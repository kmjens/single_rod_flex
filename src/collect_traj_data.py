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

def CollectData_implementation(job, communicator):
    #############################################
    ## Unit Conversions
    #############################################
    SP = JobParser(job)

    f = gsd.pygsd.GSDFile(open(job.fn('Run.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)

    time_indices = np.arange(len(traj))

    timesteps = time_indices*10000

    cmap_red = plt.get_cmap('Reds')
    cmap_blue = plt.get_cmap('Blues')
    norm = plt.Normalize(vmin=timesteps.min(), vmax=timesteps.max())

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

    # Save everything in one compressed file
    output_file = job.fn('extracted_timestep_data.npz')

    np.savez_compressed(
        output_file,
        timesteps=timesteps,
        all_mesh_positions=all_mesh_positions,
        all_active_positions=all_active_positions,
        all_const_positions=all_const_positions,
        mesh_com_positions=mesh_com_positions,
        mesh_com_velocities=mesh_com_velocities,
        active_com_positions=active_com_positions,
        active_orientations=active_orientations,
        active_com_velocities=active_com_velocities
    )

    print(f"Extracted data saved to: {output_file}")

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
