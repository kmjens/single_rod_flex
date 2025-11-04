import argparse
import freud
import glob
import hoomd
import json
import matplotlib
import os
import signac
import plotly

import gsd
import gsd.hoomd
import gsd.pygsd

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utility import *
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


def plot_rotational_autocorrelation_function(timesteps, racf, job):
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    #Note: only taking the real part
    plt.suptitle("Rotational Autocorrelation Function of Active Particle", fontsize=18)
    # -- Top row: l=2 --
    axs[0].plot(timesteps[1:], racf[:, 0][1:].real, linestyle='-', color='blue')
    
    axs[0].set_title('Magnitude |RACF| (l=2)')
    axs[0].set_ylabel('Magnitude')
    axs[0].set_ylim([0,1])
    axs[0,].grid(True)
    
    #axs[0, 1].plot(timesteps[1:], np.angle(acf[:, 0][1:]), color='purple')
    #axs[0, 1].set_title('Phase ∠RACF (l=2)')
    #axs[0, 1].set_ylabel('Phase [radians]')
    #axs[0, 1].set_xlabel('Timesteps')
    #axs[0, 1].set_ylim([-2*np.pi,2*np.pi])
    #axs[0, 1].grid(True)
    
    # -- Middle row: l=4 --
    axs[1].plot(timesteps[1:], racf[:, 1][1:].real, linestyle='-',color='blue')
    axs[1].set_title('Magnitude |RACF| (l=4)')
    axs[1].set_ylabel('Magnitude')
    
    axs[1].grid(True)
    axs[1].set_ylim([0,1])
    
    #axs[1, 1].plot(timesteps[1:], np.angle(acf[:, 1][1:]), color='purple')
    #axs[1, 1].set_title('Phase ∠RACF (l=4)')
    #axs[1, 1].set_xlabel('Timesteps')
    #axs[1, 1].set_ylim([-2*np.pi,2*np.pi])
    #axs[1, 1].grid(True)
    
    # -- Bottom row: l=6 --
    axs[2].plot(timesteps[1:], racf[:, 2][1:].real, linestyle='-', color='blue')
    axs[2].set_title('Magnitude |RACF| (l=6)')
    axs[2].set_ylabel('Magnitude')
    axs[2].set_xlabel('Simulation Timestep')
    axs[2].grid(True)
    axs[2].set_ylim([0,1])
    
    #axs[2, 1].plot(timesteps[1:], np.angle(acf[:, 2][1:]), color='purple')
    #axs[2, 1].set_title('Phase ∠RACF (l=6)')
    #axs[2, 1].set_xlabel('Timesteps')
    #axs[2, 1].set_ylim([-2*np.pi,2*np.pi])
    #axs[2, 1].grid(True)
    
    # Layout tidy
    plt.tight_layout()
    fig.savefig(job.fn('Rot_autocorr.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_orientation_path_on_S2(timesteps,x,y,z,job):
    sphere_x, sphere_y, sphere_z = get_unit_sphere()
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.15,
        showscale=False,
        colorscale='Greys',
        hoverinfo='skip',
    ))
    
    # Add orientation path with color mapping
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=5, color=timesteps, colorscale='Blues', colorbar=dict(title="Timestep")),
        line=dict(color='grey', width=2),
        name='Orientation Path',
        text=[f'Timestep: {t}' for t in timesteps],
        hoverinfo='text',
    ))
    
    fig.update_layout(
        title="Active Particle Orientation Path Projected on S2",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig.show()
    fig.write_image(job.fn("orient_path_3D.png"), scale=2)
    fig.write_html(job.fn("orient_path_3D.html"))

def plot_orientation_2D_versus_time(timesteps_exp, phi, phi_unwrapped, theta, job):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].set_title('Active Particle Orientation Direction')
    ax[0].plot(timesteps_exp, phi, label='Azimuth (phi)', color='purple')
    ax[0].legend()
    
    
    ax[1].plot(timesteps_exp, phi_unwrapped, label='Unwrapped Azimuth (phi)', color='purple')
    ax[1].legend()
    
    ax[2].plot(timesteps_exp, theta, label='Inclination (theta)', color='orange')
    ax[2].set_ylabel('Inclination [Rad]')
    ax[2].set_xlabel('Time (sec)')
    ax[2].legend()
    
    plt.tight_layout()
    
    fig.savefig(job.fn('orient_path_2D_in_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_FFT_of_orientation_data(freq, fft_xlim_max, phi_fft, phi_unwrapped_fft, theta_fft, job):
    # Only take the positive half of the spectrum
    pos_mask = freq > 0
    
    # Plot the FFT magnitude spectrum
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].set_title('FFT of Active Particle Orientation Direction')
    ax[0].plot(freq[pos_mask], np.abs(phi_fft[pos_mask]), label='Azimuth (phi) FFT', color='purple')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlim([0,fft_xlim_max])
    ax[0].legend()
    
    ax[1].plot(freq[pos_mask], np.abs(phi_unwrapped_fft[pos_mask]), label='Unwrapped Azimuth (phi) FFT', color='purple')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlim([0,fft_xlim_max])
    ax[1].legend()
    
    ax[2].plot(freq[pos_mask], np.abs(theta_fft[pos_mask]), label='Inclination (theta) FFT', color='orange')
    ax[2].set_ylabel('Amplitude')
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_xlim([0,fft_xlim_max])
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
    fig.savefig(job.fn('FFT_orientation_data.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_autocorrelation_function(timesteps_exp, acf, job):
    fig, ax = plt.subplots(figsize=(6, 8))
    # Plot the autocorrelation function alone
    ax.plot(timesteps_exp, acf, color ='k')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function of Active Rod Orientation')
    fig.savefig(job.fn('active_rod_orient_autocorr.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_autocorrelation_function_FFT(timesteps_exp, acf, acf_fft, freq, fft_xlim_max, job):
    # FFT of signals
    pos_mask = freq > 0 #Only positives
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].plot(timesteps_exp, acf, color ='k')
    ax[0].set_xlabel('Time (sec)')
    ax[0].set_ylabel('Autocorrelation')
    ax[0].set_xlim([0,timesteps_exp[-1]])
    ax[0].set_title('Rotational Autocorrelation Function of Active Rod Orientation')
    
    ax[1].plot(freq[pos_mask], np.abs(acf_fft[pos_mask]), color ='orange')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_xlim([0,fft_xlim_max])
    ax[1].set_title('FFT of Autocorrelation Function')
    
    plt.tight_layout()
    fig.savefig(job.fn('FFT_active_rod_orient_autocorr.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_contrasted_active_angular_dist(active_normalized_v, directions, cmap, job):
    nbins = 50

    # Normalize on xy plane
    vectors_xy = active_normalized_v[:, :2]
    vectors_xy /= np.linalg.norm(vectors_xy, axis=1, keepdims=True)
    angles = np.mod(np.arctan2(vectors_xy[:, 1], vectors_xy[:, 0]), 2 * np.pi)

    bin_edges = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, _ = np.histogram(angles, bins=bin_edges)
    probabilities = counts / counts.sum()

    # spherical coords
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    theta = np.arccos(directions[:, 2] / np.linalg.norm(directions, axis=1))

    fig, ax = plt.subplots(figsize=(8, 6))  # <-- fixed here
    hist, xedges, yedges = np.histogram2d(phi, theta, bins=100, range=[[-np.pi, np.pi], [0, np.pi]])
    hist /= hist.sum()

    c = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap, shading='auto')
    fig.colorbar(c, ax=ax, label='Normalized Density')
    ax.set_xlabel('Azimuthal Angle (phi)')
    ax.set_ylabel('Polar Angle (theta)')
    ax.set_title('Probability Distribution of Active Particle Orientation Angle\n(Normalized for Extra Contrast)')
    
    fig.savefig(job.fn('extra_contrast_2D_orient_dist.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_orientation_on_S2_for_active(phi,theta,job):
    
    # Compute the 2D histogram (distribution of points in spherical coordinates)
    phi_edges = np.linspace(-np.pi, np.pi, 30)
    theta_edges = np.linspace(0, np.pi, 15)
    hist, phi_edges, theta_edges = np.histogram2d(phi, theta, bins=[phi_edges, theta_edges])
    hist = hist / np.sum(hist)
    
    # Calculate the solid angle for each bin
    dphi = phi_edges[1] - phi_edges[0]
    dtheta = theta_edges[1] - theta_edges[0]
    solid_angle = np.sin(theta_edges[:-1]) * dtheta * dphi
    
    solid_angle = np.outer(np.ones(len(phi_edges) - 1), solid_angle)
    solid_angle_distribution = hist * solid_angle
    phi_grid, theta_grid = np.meshgrid(phi_edges[:-1], theta_edges[:-1])
    
    x_grid = np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = np.cos(theta_grid)
    
    colorscale = matplotlib_to_plotly(plt.cm.Blues)
    
    fig = go.Figure(data=[go.Surface(
        x=x_grid, 
        y=y_grid, 
        z=z_grid,
        surfacecolor=solid_angle_distribution.T,  # Color by solid angle distribution
        colorscale=colorscale,
        cmin=0, cmax=np.max(solid_angle_distribution),
        colorbar=dict(title="Solid Angle Density")
    )])
    
    
    fig.update_layout(
        title='Solid Angle Distribution on Sphere',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()
    fig.write_image(job.fn('extra_contrast_3D_orient_dist.png'), scale=2)
    fig.write_html(job.fn("extra_contrast_3D_orient_dist.html"))

def plot_1D_orientation_hists_for_active(phi,theta,job):
    bins = 30
    phi_range = [-np.pi, np.pi]
    theta_range = [0, np.pi]
    cmap_blue = plt.cm.Blues
    cmap_red = plt.cm.Reds
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    
    # Compute 2D histogram
    hist2d, xedges, yedges = np.histogram2d(phi, theta, bins=bins, range=[phi_range, theta_range])
    hist2d_normalized = hist2d / np.sum(hist2d)
    phi_hist = np.sum(hist2d_normalized, axis=1)   # length: bins
    theta_hist = np.sum(hist2d_normalized, axis=0) # length: bins
    
    phi_centers = 0.5 * (xedges[:-1] + xedges[1:])       # shape: (bins,)
    theta_centers = 0.5 * (yedges[:-1] + yedges[1:])     # shape: (bins,)
    
    phi_centers_norm = (phi_centers - phi_range[0]) / (phi_range[1] - phi_range[0])
    theta_centers_norm = (theta_centers - theta_range[0]) / (theta_range[1] - theta_range[0])
    
    # Normalized bin width
    phi_bin_width = 1 / bins
    theta_bin_width = 1 / bins
    phi_colors = cmap_red(norm(phi_hist))
    theta_colors = cmap_red(norm(theta_hist))
    
    fig, (ax_phi, ax_theta) = plt.subplots(
        1, 2, figsize=(12, 4),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [2, 1]}
    )
    
    # Phi
    ax_phi.bar(
        phi_centers_norm, phi_hist,
        width=phi_bin_width, align='center',
        color=phi_colors, edgecolor='none'
    )
    ax_phi.set_ylim([0, 1])
    ax_phi.set_xlim([0, 1])
    ax_phi.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_phi.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax_phi.set_xlabel('Azimuthal Angle (phi)')
    ax_phi.set_ylabel('Normalized Density')
    ax_phi.set_title('Phi Distribution')
    
    # Theta
    ax_theta.bar(
        theta_centers_norm, theta_hist,
        width=theta_bin_width, align='center',
        color=theta_colors, edgecolor='none'
    )
    ax_theta.set_ylim([0, 1])
    ax_theta.set_xlim([0, 1])
    ax_theta.set_xticks([0, 0.5, 1.0])
    ax_theta.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax_theta.set_xlabel('Polar Angle (theta)')
    ax_theta.set_ylabel('Normalized Density')
    ax_theta.set_title('Theta Distribution')
    
    fig.suptitle('Distribution of Active Particle Orientation Direction', fontsize=16, y=1.1)
    fig.savefig(job.fn('active_orient_1D_historgrams.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_2D_w_1D_velocoity_hist(orientations, suptitle, cmap, job):
    
    norms = np.linalg.norm(orientations, axis=1)
    normalized_orientations = orientations / norms[:, None]
    
    # Convert to spherical coordinates
    phi = np.arctan2(normalized_orientations[:, 1], normalized_orientations[:, 0])
    theta = np.arccos(normalized_orientations[:, 2])
    
    phi_edges = np.linspace(-np.pi, np.pi, 30)
    theta_edges = np.linspace(0, np.pi, 15)
    hist, phi_edges, theta_edges = np.histogram2d(phi, theta, bins=[phi_edges, theta_edges])
    hist_normalized = hist / np.sum(hist)
    
    phi_hist = np.sum(hist_normalized, axis=1)
    theta_hist = np.sum(hist_normalized, axis=0)
    
    # Bin centers
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    
    fig, ax_main = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax_main)
    ax_xhist = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax_main)
    ax_yhist = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax_main)
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    
    # Main 2D histogram plot
    mesh = ax_main.pcolormesh(
        phi_edges, theta_edges, hist_normalized.T,
        cmap=cmap, shading='auto', norm=norm
    )
    ax_main.set_xlabel('Azimuthal Angle (phi)')
    ax_main.set_ylabel('Polar Angle (theta)')
    ax_main.set_aspect('equal')
    fig.suptitle(suptitle, fontsize=16, y=0.8)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.88, 0.11, 0.03, 0.66])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label('Normalized Density')
    cbar.set_ticks(np.linspace(0, 1.0, 6))
    
    # 1D histograms with color-mapped bars
    phi_colors = cmap(norm(phi_hist))
    theta_colors = cmap(norm(theta_hist))
    
    ax_xhist.bar(phi_centers, phi_hist, width=np.diff(phi_edges), align='center', color=phi_colors, edgecolor='none')
    ax_xhist.axis('off')
    
    ax_yhist.barh(theta_centers, theta_hist, height=np.diff(theta_edges), align='center', color=theta_colors, edgecolor='none')
    ax_yhist.axis('off')
    
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)
    
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9)
    
    fig.savefig(job.fn('active_v_dirxn_2D_w_1D_histogram.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


    ### only 1D:
    phi_hist = np.sum(hist_normalized, axis=1)
    theta_hist = np.sum(hist_normalized, axis=0)
    
    phi_range = [-np.pi, np.pi]
    theta_range = [0, np.pi]
    
    # Normalize bin centers to [0, 1]
    phi_centers_norm = (phi_centers - phi_range[0]) / (phi_range[1] - phi_range[0])
    theta_centers_norm = (theta_centers - theta_range[0]) / (theta_range[1] - theta_range[0])
    
    phi_bin_width = 1 / len(phi_hist)
    theta_bin_width = 1 / len(theta_hist)
    phi_colors = cmap(norm(phi_hist))
    theta_colors = cmap(norm(theta_hist))
    
    fig, (ax_phi, ax_theta) = plt.subplots(
        1, 2, figsize=(12, 4),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [2, 1]}  # Aspect ratio for phi=2π, theta=π
    )
    
    # Phi histogram (azimuthal angle)
    ax_phi.bar(
        phi_centers_norm, phi_hist,
        width=phi_bin_width, align='center',
        color=phi_colors, edgecolor='none'
    )
    ax_phi.set_ylim([0, 1])
    ax_phi.set_xlim([0, 1])
    ax_phi.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_phi.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax_phi.set_xlabel('Azimuthal Angle (phi)')
    ax_phi.set_ylabel('Normalized Density')
    ax_phi.set_title('Phi Distribution')
    
    # Theta histogram (polar angle)
    ax_theta.bar(
        theta_centers_norm, theta_hist,
        width=theta_bin_width, align='center',
        color=theta_colors, edgecolor='none'
    )
    ax_theta.set_ylim([0, 1])
    ax_theta.set_xlim([0, 1])
    ax_theta.set_xticks([0, 0.5, 1.0])
    ax_theta.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax_theta.set_xlabel('Polar Angle (theta)')
    ax_theta.set_ylabel('Normalized Density')
    ax_theta.set_title('Theta Distribution')
    
    # Overall title
    fig.suptitle(suptitle, fontsize=16, y=1.1)
    
    # Show the plot
    fig.savefig(job.fn('active_v_dirxn_1D_histogram.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_v_distrib_on_S2(active_normalized_v, job):
    # First data: Active Particle Velocity
    x_grid1, y_grid1, z_grid1, solid_angle_dist1 = plot_solid_angle_dist(active_normalized_v)
    colorscale_blue = matplotlib_to_plotly(plt.get_cmap('Blues'))
    
    # Create subplot with two 3D scenes side by side
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scene'}]],
        subplot_titles=("Active Particle Velocity")
    )
    
    # First surface plot
    fig.add_trace(go.Surface(
        x=x_grid1, y=y_grid1, z=z_grid1,
        surfacecolor=solid_angle_dist1.T,
        colorscale=colorscale_blue,
        cmin=0, cmax=max(np.max(solid_angle_dist1), np.max(solid_angle_dist2)),
        colorbar=dict(title="Solid Angle Density"),
        showscale=True
    ), row=1, col=1)
    
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000,
        title_text="Solid Angle Distributions",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='cube'
        )
    )
    
    fig.show()
    fig.write_image(job.fn("plot_BOTH_v_distribs_on_S2.png"), scale=2)
    fig.write_html(job.fn("plot_BOTH_v_distribs_on_S2.html"))

def plot_COM_velocity(timesteps, active_com_v_norms, speed_conv_mm, step_to_sec, job):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, active_com_v_norms, label="Active rod", color='blue',alpha=.8)
    
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(timesteps))
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=np.max([np.max(active_com_v_norms),np.max(mesh_com_v_norms)]))
    
    # second axes for exp units
    ax2 = ax.twinx()
    ax2.set_ylabel("Speed (mm/s)")
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f"{tick * speed_conv_mm:.2f}" for tick in ax.get_yticks()])  # Convert the tick labels
    
    ax3 = ax.twiny()
    ax3.set_xlabel(r"Time (sec)")  
    ax3.set_xticks(ax.get_xticks())  
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()]) 
    
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Speed (sim units)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # zip preserves first occurrence
    ax.legend(unique.values(), unique.keys(), loc='best')
    
    ax.set_title(f"Magnitudes of Rod and Mesh C.O.M. Velocity")
    ax.grid(True)

    fig.savefig(job.fn('velocity_mags_BOTH_w_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_COM_velocity_FFT(freq, active_v_fft, fft_xlim_max, job):
    
    # Only take the positive half of the spectrum
    pos_mask = freq > 0
    
    # Plot the FFT magnitude spectrum
    fig, ax = plt.subplots(figsize=(6, 8), sharex=True)
    ax.plot(freq[pos_mask], np.abs(active_v_fft[pos_mask]), label='FFT of active velocity', color = 'blue', alpha=0.8,linestyle='--')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim([0,fft_xlim_max])
    ax.legend()
    plt.title('FFT of Magnitudes of Rod C.O.M. Velocity')
    plt.tight_layout()

    fig.savefig(job.fn('FFT_velocity_mags_w_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_alignment_metrics(AM, timesteps, step_to_sec, job):

    ## NORMALIZED V
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- First subplot: Full velocity inner product ---
    ax = axes[0]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.norm_v_dots, color='black', alpha=0.9)
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Inner Product of Normalized Velocity Vectors")
    ax.set_title("Inner Product of Active and Mesh Velocity Vectors")
    
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=-1, top=1)
    ax.grid(True)
    
    # Secondary x-axis (time)
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    # --- Second subplot: XY-projected velocity inner product ---
    ax = axes[1]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.norm_v_dots_xy, color='grey')
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Inner Product of Normalized Velocity Vectors")
    ax.set_ylabel("Inner Product of Normalized Velocity Vectors")
    ax.set_title("For xy-projected vectors")
    
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=-1, top=1)
    ax.grid(True)
    
    # Secondary x-axis (time)
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    fig.tight_layout()
    fig.savefig(job.fn('v_dots_norm.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    ## ORIENTATION
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # --- First subplot: Orientation vs mesh velocity (3D) ---
    ax = axes[0]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.norm_ori_dots, color='black')
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Inner Product of Normalized Active Particle Orientation and Mesh Velocity")
    ax.set_title("Inner Product of Active Force and Mesh Orientation")
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=-1, top=1)
    ax.grid(True)
    
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    # --- Second subplot: Orientation vs mesh velocity (XY projection) ---
    ax = axes[1]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.norm_ori_dots_xy, color='grey')
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Inner Product of Normalized Active Particle Orientation and Mesh Velocity")
    ax.set_title("For xy-projected vectors")
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=-1, top=1)
    ax.grid(True)
    
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    fig.tight_layout()
    
    fig.savefig(job.fn('ori_norm.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    
    ## DISPLACEMENT
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # --- Plot 1: UN-normalized displacement inner product ---
    ax = axes[0]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.pos_dots, color='green')
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel("Inner Product of (UN-normalized) Displacement Vectors")
    ax.set_title("Inner Product of UN-Normalized Active and Mesh Displacement Vectors")
    ax.set_xlim(left=0, right=max(timesteps))
    ax.grid(True)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    # --- Plot 2: Normalized displacement inner product (3D) ---
    ax = axes[1]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.norm_pos_dots, color='black')
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Inner Product of Normalized Displacement Vectors")
    ax.set_title("Inner Product of Active and Mesh Displacement Vectors")
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=-1, top=1)
    ax.grid(True)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    # --- Plot 3: Normalized displacement inner product (XY projection) ---
    ax = axes[2]
    ax.set_facecolor('white')
    ax.plot(timesteps, AM.norm_pos_dots_xy, color='grey')
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel("Inner Product of Normalized Displacement Vectors (xy-projections)")
    ax.set_title("Inner Product of Active and Mesh Displacement Vectors (xy-projections)")
    ax.set_xlim(left=0, right=max(timesteps))
    ax.set_ylim(bottom=-1, top=1)
    ax.grid(True)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlabel("Time (sec)")
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])
    
    fig.tight_layout()
    
    # Save as PNG
    fig.savefig(job.fn('pos_dots.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_alignment_metrics_FFT(AM, freq, timesteps, step_to_sec, fft_xlim_max, job):
    
    # Only take the positive half of the spectrum
    pos_mask = freq > 0
    
    # Plot the FFT magnitude spectrum
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    ax[0].set_title('FFT of Inner Product of Active Force and Mesh Orientation')
    ax[0].plot(freq[pos_mask], np.abs(AM.inner_v_fft_xy[pos_mask]), label='XY-projection vector', alpha=0.8, color='grey',linestyle='-')
    ax[0].plot(freq[pos_mask], np.abs(AM.inner_v_fft[pos_mask]), label='3-vector', color = 'k',alpha=0.8, )
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_xlim([0,fft_xlim_max])
    ax[0].legend()
    
    ax[1].set_title('FFT of Innter Product of Active Particle Orientation and Mesh Velocity')
    ax[1].plot(freq[pos_mask], np.abs(AM.inner_ori_fft_xy[pos_mask]), label='XY-projection vector', alpha=0.8, color='grey',linestyle='-')
    ax[1].plot(freq[pos_mask], np.abs(AM.inner_ori_fft[pos_mask]), label='3-vector', color = 'k',alpha=0.8, )
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_xlim([0,fft_xlim_max])
    ax[1].legend()
    
    ax[2].set_title('FFT of Inner Product of Normalized Displacement Vectors')
    ax[2].plot(freq[pos_mask], np.abs(AM.inner_pos_fft[pos_mask]), label='unnormalized 3-vector', alpha=0.8, color = 'green', linestyle='--')
    ax[2].plot(freq[pos_mask], np.abs(AM.inner_normpos_fft[pos_mask]), label='XY-projection vector', alpha=0.8, color='black',linestyle='-')
    ax[2].plot(freq[pos_mask], np.abs(AM.inner_normpos_fft_xy[pos_mask]), label='3-vector', alpha=0.8, color = 'grey')
    ax[2].set_ylabel('Amplitude')
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_xlim([0,fft_xlim_max])
    ax[2].legend()
    
    plt.tight_layout()
    fig.savefig(job.fn('FFT_alignment_metrics.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_z_position_data(active_z_exp, timesteps, step_to_sec, job):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, active_z_exp, label="Active rod", color='blue', alpha=0.8)
    ax.set_xlabel("Simulation Timestep")
    ax.set_ylabel(r"Z-Position ($\mu$m)")
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(timesteps))
    ax.legend(loc='best')
    ax.set_title(f"Distance Above Flat Surface")
    ax3 = ax.twiny()
    ax3.set_xlabel(r"Time (sec)")  # Label for the second x-axis
    ax3.set_xticks(ax.get_xticks())  # Set the ticks of the second x-axis to match the first one
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])  # Convert the tick labels
    ax.grid(True)
    fig.savefig(job.fn('d_from_floor.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_z_position_FFT(freq, active_z_fft, fft_xlim_max, job):
    # Only take the positive half of the spectrum
    pos_mask = freq > 0
    
    # Plot the FFT magnitude spectrum
    fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
    ax.set_title('Z-position relative to floor')
    ax.plot(freq[pos_mask], np.abs(active_z_fft[pos_mask]), label='active z-position', alpha=0.8, color='blue',linestyle='--')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0,fft_xlim_max])
    ax.legend()
    plt.tight_layout()
    fig.savefig(job.fn('FFT_z_pos_data.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_ellipticity_metrics(axes, job):
    # 3D Ellipsoid Plot
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # Parametric Equations
    x = axes[0] * np.outer(np.cos(u), np.sin(v))
    y = axes[1] * np.outer(np.sin(u), np.sin(v))
    z = axes[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_surface(x, y, z, color='grey', alpha=0.3)
    
    # XY-plane ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    x_xy = axes[0] * np.cos(theta)
    y_xy = axes[1] * np.sin(theta)
    ax1.plot(x_xy, y_xy, np.zeros_like(x_xy), color='r', linestyle='-', 
             label="XY-plane ellipse")
    
    # YZ-plane ellipse
    y_yz = axes[1] * np.cos(theta)
    z_yz = axes[2] * np.sin(theta)
    ax1.plot(np.zeros_like(y_yz), y_yz, z_yz, color='g', linestyle='--', label="YZ-plane ellipse")
    
    # XZ-plane ellipse
    x_xz = axes[0] * np.cos(theta)
    z_xz = axes[2] * np.sin(theta)
    ax1.plot(x_xz, np.zeros_like(x_xz), z_xz, color='b', linestyle=':', label="XZ-plane ellipse")
    
    ax1.set_box_aspect([axes[0], axes[1], axes[2]])
    ax1.set_title('3D Ellipsoid with Traces')
    
    fig1.savefig(job.fn('ellipsoid_3D_wProjections.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    
    ## 2D Projections of Ellipses
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)
    
    ## Make sure longer axes is on x coordinate
    if axes[0] > axes[1]: 
        ellipse_xy = Ellipse((0, 0), width=2*axes[0], height=2*axes[1], fill=False, color='r', linestyle='-', label="XY-plane")
    else:
        ellipse_xy = Ellipse((0, 0), width=2*axes[1], height=2*axes[0], fill=False, color='r', linestyle='-', label="XY-plane")
    
    if axes[1] > axes[2]: 
        ellipse_yz = Ellipse((0, 0), width=2*axes[1], height=2*axes[2], fill=False, color='g', linestyle='--', label="YZ-plane")
    else: 
        ellipse_yz = Ellipse((0, 0), width=2*axes[2], height=2*axes[1], fill=False, color='g', linestyle='--', label="YZ-plane")
    
    if axes[0] > axes[2]:
        ellipse_xz = Ellipse((0, 0), width=2*axes[0], height=2*axes[2], fill=False, color='b', linestyle=':', label="XZ-plane")
    else:
        ellipse_xz = Ellipse((0, 0), width=2*axes[2], height=2*axes[0], fill=False, color='b', linestyle=':', label="XZ-plane")
    
    
    ax2.add_artist(ellipse_xy)
    ax2.add_artist(ellipse_yz)
    ax2.add_artist(ellipse_xz)
    
    ax2.set_xlim(-max(axes), max(axes))
    ax2.set_ylim(-max(axes), max(axes))
    ax2.set_ylabel('Minor Axis')
    ax2.set_xlabel('Major Axis')
    ax2.set_aspect('equal')
    
    ax2.set_title('2D Projections of Ellipses')
    ax2.legend()
    ax2.grid(True)
    
    fig2.savefig(job.fn('ellipse_2D_projections.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_distances(timesteps, active_total_distances_um, step_to_sec, active_linReg, job):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, active_total_distances_um, label="Active rod", color='blue', alpha = 0.8)
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(timesteps))
    ax.set_ylabel(r"Total Distance Travelled ($\mu$m)")
    ax.legend(loc='best')
    ax.set_title(f"Total Distance Travelled During Simulation")#\nCorrelation: {disp_corr:.2f}")
    
    ax3 = ax.twiny()
    ax3.set_xlabel(r"Time (sec)")  # Label for the second x-axis
    ax3.set_xticks(ax.get_xticks())  # Set the ticks of the second x-axis to match the first one
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])  # Convert the tick labels
    
    
    ax.grid(True)
    fig.savefig(job.fn('total_dist.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print('Distance Travelled plotted.')  


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, active_total_distances_um, label="Active rod", color='blue', alpha=0.8)
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(timesteps))
    
    # Annotating slope and R2 values
    ax.text(0.05, 0.95, f"Slope (Active rod): {active_linReg['slope']:.2e}"+r" ($\mu$m/sec)"+f"\nR² (Active rod): {active_linReg['r_squared']:.2f}\nStd Err (Active rod): {active_linReg['std_err']:.2e}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top', color='blue')
    
    ax3 = ax.twiny()
    ax3.set_xlabel(r"Time (sec)")  # Label for the second x-axis
    ax3.set_xticks(ax.get_xticks())  # Set the ticks of the second x-axis to match the first one
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])  # Convert the tick labels
    
    
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel(r"Total Distance Travelled ($\mu$m)")
    ax.legend(loc='best')
    ax.set_title(f"Total Distance Travelled During Simulation") #\nCorrelation: {dist_corr:.2f}")
    ax.grid(True)
    
    fig.savefig(job.fn('total_dist_Wslopes.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print('Distance plotted w slopes.')


def plot_displacement(timesteps, disp_active_um, step_to_sec, disp_corr, job):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    ax.plot(timesteps, disp_active_um, label="Active rod", color='blue', alpha = 0.8)
    ax.set_xlabel("Simulation Timesteps")
    ax.set_ylabel(r"Displacement from Starting Position ($\mu$m)")
    ax.legend(loc='best')
    ax.set_title(f"Center of Mass Displacement During Simulation \n Correlation: {disp_corr:.3f}")
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(timesteps))
    
    ax3 = ax.twiny()
    ax3.set_xlabel(r"Time (sec)")  # Label for the second x-axis
    ax3.set_xticks(ax.get_xticks())  # Set the ticks of the second x-axis to match the first one
    ax3.set_xticklabels([f"{tick * step_to_sec:.2f}" for tick in ax.get_xticks()])  # Convert the tick labels
    
    fig.savefig(job.fn('displacements.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print('Displacement plotted.')  


def plot_displacement_FTT(timesteps, freq, active_disp_fft, step_to_sec, fft_xlim_max, job):
    # Only take the positive half of the spectrum
    pos_mask = freq > 0
    
    # Plot the FFT magnitude spectrum
    fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
    
    ax.set_title('COM positional displacement')
    ax.plot(freq[pos_mask], np.abs(active_disp_fft[pos_mask]), label='active', alpha=0.8, color='blue',linestyle='--')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([0,fft_xlim_max])
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(job.fn('FFT_displacements.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Displacement FFTs plotted.')  
