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

def Analysis_implementation(job, communicator):
    
    #############################################
    ## Collect positions
    #############################################
   
    #f = gsd.pygsd.GSDFile(open(job.fn('active.gsd'), 'rb'))
    f = gsd.pygsd.GSDFile(open(job.fn('Setup.gsd'), 'rb'))
    traj = gsd.hoomd.HOOMDTrajectory(f)
    
    time_indices = np.arange(len(traj))
    cmap_red = plt.get_cmap('Reds')
    cmap_blue = plt.get_cmap('Blues')
    norm = plt.Normalize(vmin=time_indices.min(), vmax=time_indices.max())

    com_positions = []
    active_positions = []
    mesh_positions = []
    
    for frame in traj:
        positions = frame.particles.position
        typeids = frame.particles.typeid
        
        mask = (typeids == 0)
        mesh_position = positions[mask]
        mesh_positions.append(mesh_position)
        com_positions.append(np.average(mesh_position, axis=0))

        mask = (typeids == 1)
        filtered_positions = positions[mask]
        active_positions.append(np.average(filtered_positions, axis=0))

    print("Positions taken from GSD.")

    com_positions = np.array(com_positions)
    active_positions = np.array(active_positions)
   

    #############################################
    ## Plot and save x-y plane trajectories
    #############################################
    
    com_x = com_positions[:, 0]
    com_y = com_positions[:, 1]
    active_x = active_positions[:, 0]
    active_y = active_positions[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('white')

    # Plot and add color bars
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

    print('Trajectories plotted.')

    #############################################
    ## Spherical deformation
    #############################################
    
    final_pos = mesh_positions[-1][:]
    axes = [(max(final_pos[:, 0]) - min(final_pos[:, 0])),
            (max(final_pos[:, 1]) - min(final_pos[:, 1])),
            (max(final_pos[:, 2]) - min(final_pos[:, 2]))]
    
    # Mesh length, width, height aspect ratios:
    ratio_xz = axes[0] / axes[2]
    ratio_yz = axes[1] / axes[2]
    ratio_xy = axes[0] / axes[1]

    # Polar eccentricity: 
    '''
    e = 0 is perfect sphere.
    e --> 1 with infinite deformation
    Here, we assume symmetry along x-y plane and take average.
    We assume z axis is shorter than x-y because of gravity.
    Else, we set e to NaN.
    '''
    max_ax = (axes[0] + axes[1])/2 # assume xy symmetry
    min_ax = axes[2] # set to axis_z
    e = 1 - (min_ax * min_ax) / (max_ax * max_ax)

    if e < 0:
        e = np.nan  # Set to NaN
    else:
        e = np.sqrt(e)

    # Volume deviations:
    R_avg = (axes[0] * axes[1] * axes[2])**(1/3)
    V_sphere = (4/3) * np.pi * R_avg**3
    V_actual = (4/3) * np.pi * axes[0] * axes[1] * axes[2]
    V_diff_sqr = (V_actual - V_sphere)**2

    # Sphericity: (assume oblate spheriod)
    '''
    Phi = 1 for perfect sphere
    Phi > 1 as deformations increase.
    Here, we assume symmetry along the x-y plane.
    We asume z-axis is shorted than x-y because of gravity. (ie. oblate)
    Else, we set Phi to NaN
    '''
    SA_sphere = (4 * np.pi)**(1/3) * ((3 * V_actual)**(2/3))
    SA_actual = 2 * np.pi * max_ax**2 * (1 + (min_ax/(max_ax * e)) * np.arcsin(e))
    SA_diff_sqr = (SA_actual - SA_sphere)**2
    Phi = SA_actual / SA_sphere

    print('Spherical deformation analysis completed.')

    #############################################
    ## Active-mesh positional change
    #############################################
    
    # Peason correlation of active and mesh com positions 
    '''
    1:  perfect correlation
    0:  no correlation
    -1: perfect neg correlation
    '''
    Pear_corrs = []
    for i in range(3):  # We have three dimensions: x, y, z
        corr = np.corrcoef(com_positions[:, i], active_positions[:, i])[0, 1]
        Pear_corrs.append(corr)

    # Pearson correlation of MSDs
    active_i_pos = active_positions[0]
    com_i_pos = com_positions[0]

    msd_active = []
    msd_com = []

    for t in range(1, len(active_positions)):
        disp = active_positions[t] - active_i_pos
        msd_active.append(np.sum(disp**2))
        
        disp = com_positions[t] - com_i_pos
        msd_com.append(np.sum(disp**2))

    MSD_corr = np.corrcoef(msd_active, msd_com)[0, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(msd_active, label="MSD of active rod", color='red')
    ax.plot(msd_com, label="MSD of mesh COM", color='blue')
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("MSD")
    ax.legend(loc='best')
    ax.set_title(f"MSDs of Rod and Mesh COM\nCorrelation: {MSD_corr:.2f}")
    ax.grid(True)
    fig.savefig(job.fn('MSDs.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print('Correlations calculated and plotted')

    #############################################
    ## Save results to json file
    #############################################

    all_data = {
        "jobid": job.id,
    }
    statepoints = job.sp
    all_data.update(statepoints)
    analysis_data = {
        "mesh_len_ratio_xz": float(ratio_xz),
        "mesh_len_ratio_yz": float(ratio_yz),
        "mesh_len_ratio_xy": float(ratio_xy),
        "eccentricity":      float(e),
        "spherical_volume":  float(V_sphere),
        "ellipsoidal_volume": float(V_actual),
        "volume_square_difference": float(V_diff_sqr),
        "spherical_SA":         float(SA_sphere),
        "ellipsoidal_volume":   float(SA_actual),
        "SA_square_difference": float(SA_diff_sqr),
        "sphericity":           float(Phi),
        "pos_correlation_x":    float(Pear_corrs[0]),
        "pos_correlation_y":    float(Pear_corrs[1]),
        "pos_correlation_z":    float(Pear_corrs[2]),
        "MSD_correlation":      float(MSD_corr)
    }
    all_data.update(analysis_data)
    with open(job.fn('analysis_data.json'), 'w') as f:
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
