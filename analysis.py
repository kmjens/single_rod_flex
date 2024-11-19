import argparse
import freud
import hoomd
import matplotlib
import signac

import gsd
import gsd.hoomd
import gsd.pygsd

import numpy as np
import matplotlib.pyplot as plt

def Analysis_implementation(job, communicator):

    f = gsd.pygsd.GSDFile(open(job.fn('active.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)
    
    # get color maps
    time_indices = np.arange(len(traj))
    cmap_red = plt.get_cmap('Reds')
    cmap_blue = plt.get_cmap('Blues')
    norm = plt.Normalize(vmin=time_indices.min(), vmax=time_indices.max())

    com_positions = []
    active_positions = []
    
    for frame in traj:
        positions = frame.particles.position
        typeids = frame.particles.typeid
        
        mask = (typeids == 0)
        filtered_positions = positions[mask]
        com_positions.append(np.average(filtered_positions, axis=0))

        mask = (typeids == 1)
        filtered_positions = positions[mask]
        active_positions.append(np.average(filtered_positions, axis=0))

    com_positions = np.array(com_positions)
    com_x = com_positions[:, 0]
    com_y = com_positions[:, 1]

    active_positions = np.array(active_positions)
    active_x = active_positions[:, 0]
    active_y = active_positions[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('white')
    
    sc_com = ax.scatter(com_x, com_y, 
                        c=time_indices, cmap=cmap_red, 
                        norm=norm, label="Mesh Center of Mass", s=7)
    sc_active = ax.scatter(active_x, active_y, 
                           c=time_indices, cmap=cmap_blue, 
                           norm=norm, label="Active Particle", s=7)
    cbar_com = fig.colorbar(sc_com, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
    cbar_com.set_label('Simulation Time')
    cbar_active = fig.colorbar(sc_active, orientation='vertical', fraction=0.05, pad=0.02)
    cbar_active.ax.yaxis.set_ticks([])

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(f"x-y trajectory\nJob ID: {job}")
    ax.legend(loc='best')
    ax.axis('equal')
    ax.grid()
    fig.savefig(job.fn('xy_traj.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

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
