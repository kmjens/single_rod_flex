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


def AngleAnalysis_implementation(job, communicator):
    # Collect Data: 
    data = np.load(job.fn('extracted_timestep_data.npz'))

    timesteps = data['timesteps']

    all_mesh_positions = data['all_mesh_positions']
    all_active_positions = data['all_active_positions']
    all_const_positions = data['all_const_positions']

    mesh_com_positions = data['mesh_com_positions']
    mesh_com_velocities = data['mesh_com_velocities']

    active_com_positions = data['active_com_positions']
    active_orientations = data['active_orientations']
    active_com_velocities = data['active_com_velocities']

    ##Calc angle to mesh normal:
    f = gsd.pygsd.GSDFile(open(sim_folder[motion]+'initial_wRigid.gsd', 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)


    for frame in traj:
        positions = frame.particles.position
        typeids = frame.particles.typeid
        mask = (typeids == 2)
        const_positions = positions[mask]

    print(const_positions)

    num_const_beads = len(const_positions)
    front_bead_idx = num_const_beads - 1
    front_const_pos = const_positions[front_bead_idx]

    print(front_const_pos)

    num_const_beads = len(all_const_positions[0])
    num_mesh_particles = len(all_mesh_positions[0])
    num_frames = len(timesteps)

    #Shift COM
    shifted_mesh_pos = []
    shifted_const_pos = []

    for t in list(np.arange(num_frames)):
        rod_com_pos = all_active_positions[t][0]

        #constituent particles
        const_pos = all_const_positions[t][front_bead_idx]
        const_pos = const_pos - rod_com_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    project = signac.get_project()
    jobs = [project.open_job(id=directory) for directory in args.directories]
    globals()[args.action](*jobs)
