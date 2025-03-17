# Various-Forces
#Perturbations of Positively Charged Dust Grains of D68 ringlet of Saturn’s D ring by Gravitational and non-gravitational forces.
# TRAJECTORIES PLOTS
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 01:38:57 2025

@author: sugnu
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# Constants and functions
μ       = 379312077e8            # m^3/s^2 
R       = 60268e3                # metre
g_10    = 21141e-9
Ω       = 1.74e-4                # rad/s
j_2     = 1.629071e-2            # J2 perturbation coefficient
# charge by mass ratio
ρ       = 1e3                    # 1gm/cm^3 Headmen 2021= 10^3 kg/m^3
#b = 1e-6                        # = 1e-3 micro
V       = 10                     # Volt
ε       = 8.85e-12               # Farad/metre
μ_0     = 4*np.pi*1e-7

# Initial parameters
r0      = 67628000               # 1.11*R #67628e3  # Initial radial distance
z_kep   = (μ/(r0**3))**0.5

# Initial conditions for the plasma ions
r_i     = 67628000              #1.11 * R 
x_i     = 0.0
θ_i     = np.pi /2
y_i     = 0.0
z_i     = 1.74e-4

# Initial conditions of the charged particle
x0      = 0
θ0      = np.pi /2
y0      = 0.0
z_0     = z_kep  - 1.74e-4

n_i     = 1e6                #[Hadid 2018:100/cubic-cm, Wahlund 2017]

w_relative = np.sqrt((x_i - x0)**2 + (r_i*y_i - r0*y0)**2 + (r_i*z_i*np.sin(θ_i) - r0*z_0*np.sin(θ0))**2)

#Burns
L_sun           = 3.828e26      # Solar luminosity in watts##S_o = 1360 # W/m^2 1.36e6 ergs cm^-2 sec^-1/ 1.36×10^6 × 10^-7J × 10^4 m^-2 × 1s^-1
c               = 3e8           # Speed of light in m/s
Q_pr            = 1.0           # 1 # radiation pressure coefficient ( for dielectric grains -~1 for 1µm and = 0.3 for 0.11µm radii grains, (Burns et al. 1979), 

#alpha_sw       = 0.35          # Solar wind drag factor
r_sun_to_dust   = 1.433e11      #9.58 AU- 1.11*R   1.433e12     # 9.5 AU, 1AU =149600000000

#v_rel          = 3e4           # Relative velocity of the particle in m/s (assumed 30 km/s)

alpha           = 26.5*(pi/180) # Solar latitude 20°, [Cravens et al. 2018 ], Hadid 2018 26.6 deg
ϕ_s             = 0             # Solar longitude (e.g., 45 degrees)

def event_func_r(t, p):
    return p[0] - 61268e3 # Radiation belt Kollman et al. 2018
event_func_r.terminal   = True
event_func_r.direction  = -1

def event_func_outer(t, p):
    return p[0] - 9198e4#7466e4 C
event_func_outer.terminal   = True
event_func_outer.direction  = +1

from matplotlib import cm
import matplotlib.colors as mcolors

#b_values  = np.array([1e-9,5e-9,9e-9,1e-8,3e-8,7e-8])
#b_values  = np.array([8e-8,1e-7,5e-7,9e-7,1e-6,10e-6]) # Hit after some movement
b_values  = np.array([2e-8,5e-8,9e-8,2e-7,3e-7,4e-7])
#b_values  = np.array([5e-7,9e-7,1e-6,5e-6,9e-6,10e-6])

#colors = plt.cm.rainbow_r(np.linspace(0.1, 0.9, len(b_values)))
#colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(b_values)))
#colors= np.array(['cyan','deepskyblue','dodgerblue','blue'])
colors= np.array(['#C7EA46','#00FF00','#90EE90','#9ACD32','olive','green'])#'#AAFF32','darkgreen''#76FF7B',
#colors= np.array(['thistle','plum','violet','mediumorchid','mediumorchid','darkorchid'])
#colors= np.array(['firebrick','chocolate','lightcoral','rosybrown','tan','burlywood'])#'tomato'])
#colors= np.array(['cyan','mediumturquoise','lightskyblue','deepskyblue','dodgerblue','blue'])

trajectories = {}
b_values_end_times = []

# Iterate over different values of b
for i, b_value in enumerate(b_values):
    # Update parameters based on the current value of b
    m = (4/3) * pi * (b_value**3) * ρ  # Particle mass (assuming spherical particle)
    q = 4 * pi * ε * b_value * V       # Particle charge
    β = q /m                           # Charge-to-mass ratio

    a_plasma = np.pi *b_value**2 *n_i* w_relative  # Plasma drag acceleration

    def odes(t, p):
        # Extracting variables from the state vector
        r, x, θ, y, ϕ, z = p

        # Magnetic field components in a dipole field approximation
        B_θ = μ_0 * (R/r)**3 * g_10 * np.sin(θ)  # θ-component of magnetic field
        B_r = μ_0 * (R/r)**3 * g_10 * np.cos(θ)  # r-component of magnetic field

        a_rp = (L_sun * b_value**2* Q_pr) / (4 * m * c *r_sun_to_dust**2)
        
        a_r = -a_rp  * (np.sin(θ) * np.cos(alpha) * np.cos(ϕ - ϕ_s)+ np.cos(θ) * np.sin(alpha))
        a_θ = -a_rp * (np.cos(θ) * np.cos(alpha) * np.cos(ϕ - ϕ_s) - np.sin(θ) * np.sin(alpha))
        a_ϕ =  a_rp  * np.cos(alpha) * np.sin(ϕ - ϕ_s)

       # Avoid division by zero in spherical coordinates
        sin_θ_safe = np.sin(θ) + 1e-15
        
        # Time derivatives
        drdt = x  # Radial velocity
        dxdt = (r * (y**2 + (z + Ω)**2 * np.sin(θ)**2  # Centrifugal force term
                - β * z * np.sin(θ)* B_θ)              # Lorentz force (θ-component)
                - a_plasma * x                         # Plasma drag (radial component)
                + a_r
                - (μ /r**2) * (1 - (3/2) * j_2 * ((R /r)**2) *(3 *np.cos(θ)**2 - 1)))# Radiation pressure (radial)

        dθdt = y  # Colatitudinal velocity
        dydt = ((-2 * x * y                                # Coriolis force
                + r * (z + Ω)**2 * np.sin(θ) * np.cos(θ)  # Centrifugal force
                + β * r * z *np.sin(θ)* B_r) / r - a_plasma * y + a_θ /r 
                + (3 * μ / r**3) *j_2 *((R / r)**2) * np.sin(θ) * np.cos(θ))

        dϕdt = z  # Azimuthal velocity
        dzdt = (-2 * (z + Ω) * (x * sin_θ_safe + r * y * np.cos(θ))  # Coriolis force
                + β * (x * B_θ - r * y *B_r)) / (r *sin_θ_safe) - a_plasma *z + a_ϕ/ (r *sin_θ_safe) 

        # Return derivatives as a numpy array
        return np.array([drdt, dxdt, dθdt, dydt, dϕdt, dzdt])

    # time window
    t_span = (0, 864000)
    #t = np.linspace(t_span[0],t_span[1],3600)

    dt_desired = 5 # Desired time step in seconds
    num_points = int((t_span[1] - t_span[0]) / dt_desired) + 1
    t = np.linspace(t_span[0], t_span[1], num_points)
    
    # initial conditions
    p0 = np.array([r0, x0, 90.0*(pi/180), 0, 0.0*(pi/180),z_0])

    # Solve IVP
    sol = solve_ivp(odes, t_span, p0, t_eval=t, events=[event_func_r, event_func_outer], method="BDF", rtol=1e-9, atol=1e-11)
    #sol = solve_ivp(odes, t_span, p0, t_eval=t, events=event_func_r, method="BDF", rtol=1e-9, atol=1e-11)

    # Extract final values and end time
    final_values = sol.y[:, -1]
    end_time = sol.t[-1]

    labels = [" r", "dr", "θ ","dθ ", "φ ", "dφ"]
  # Print or use final values for the next part of integration
    print(f" b = {b_value:.1e} m:")
    print("Final Values:")
    for label, value in zip(labels, final_values):
        print(f"  {label}: {value:.4e}")
    print("End Time:", f"{end_time:.2f} s")
    
        # Check if any events occurred
    if sol.status == 0 and len(sol.t_events[0]) > 0:
        for i, event_time in enumerate(sol.t_events[0]):
            event_idx = np.where(t >= event_time)[0][0]
            t_event = sol.t_events[0][i]
            r_event = sol.y_events[0][i, 0]
            print(f"Event {i + 1} occurred at time {t_event:.2f} seconds.Radial distance:{r_event:.2f}")
        else:
            event_idx = len(t) - 1  # Plot until the end if no event occurred
            print("Integration completed without reaching the specified event condition.")
    
    print("Integration Status:", sol.message)
    print("Number of Events Detected:", len(sol.t_events[0]))
   
    # Calculate total rotation angle in radians 
    total_rotation = sol.y[4, -1] - sol.y[4, 0] 

    # Convert total rotation angle to degrees 
    total_rotation_degrees = total_rotation * (180/pi) 

    # Calculate the number of complete rounds 
    complete_rounds = int(total_rotation_degrees //360) 

    # Print the results
    print(f"b ={b_value} completed {total_rotation_degrees:.2f} degrees",
          f", which is {complete_rounds} complete rounds.")

    # Convert azimuthal angle to degrees
    phi_degrees = sol.y[4]*(180/pi)  

    # Initialize variables
    rotation_times = []
    current_round = 1  # Start from the first round
    target_angle = 360 * current_round  # First complete round target

    # Find times when each complete round is reached
    for i in range(1, len(phi_degrees)):
        if phi_degrees[i-1] < target_angle <= phi_degrees[i]:  
            rotation_times.append(sol.t[i])  # Store the time
            print(f"b= {b_value} completed round {current_round} at t = {sol.t[i]:.2f} seconds")
            current_round += 1  # Move to the next round
            target_angle = 360 * current_round  # Update the next target

    # Print minimum values of sol.y for each component
    min_values = np.min(sol.y, axis=1)
    min_indices = np.argmin(sol.y, axis=1)
    min_times = sol.t[min_indices]

    print(f"b = {b_value}:")
    print("Min Values of sol.y:", min_values)
    print("Min Time Points:", min_times)
    
    # Print minimum values of sol.y for each component
    max_values = np.max(sol.y, axis=1)
    max_indices = np.argmax(sol.y, axis=1)
    max_times = sol.t[max_indices]

    print(f"b = {b_value}:")
    print("Max Values of sol.y:", max_values)
    print("Max Time Points:", max_times)
   
    trajectories[b_value] = sol
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                        VISUALISATION
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit if not already
from collections import OrderedDict  # For legend handling
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.ticker import ScalarFormatter, AutoLocator, FuncFormatter

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                               Plot all trajectories in polar plot
# ---------------------------------------------------------------------------------------------
fig_polar, axs_polar = plt.subplots(
    subplot_kw={'projection': 'polar'}, dpi=300)
axs_polar.set_facecolor('black')

# Planet representation
planet = plt.Circle(
    (0, 0), R, transform=axs_polar.transData._b, color='#EE9A49', alpha=0.8)
axs_polar.add_patch(planet)
for idx, (b_value, sol) in enumerate(trajectories.items()):
    r_sol = sol.y[0, :]
    phi_sol = sol.y[4, :]
    axs_polar.plot(phi_sol, r_sol, linewidth=1.5, linestyle='solid', alpha=1, color=colors[idx],
                   label=f'{b_value:.1e} m', zorder=10)
    axs_polar.plot(phi_sol[-1], r_sol[-1], 'o',
                   color=colors[idx], markersize=5, markeredgecolor='white',zorder=12)

# Configure polar plot
#axs_polar.set_rlim(0, 7.93e7)
axs_polar.set_rticks([R, 61268e3, 676e5, 7156e4, 7327e4, 7466e4, 9198e4])
axs_polar.set_yticklabels([])
axs_polar.set_rlabel_position(30)
axs_polar.grid(color='white', linestyle='dotted', linewidth=0.7, alpha=1)
axs_polar.tick_params(axis='x', labelsize=13, pad=3)

# Add legend
axs_polar.legend(loc='upper left', bbox_to_anchor=(1.2, 1), borderaxespad=0.,
                 title="Legend", fontsize=10, title_fontsize=10, ncol=1)
# ..................................................................................

# Define the radii for the concentric circles (rings) ,labels and text angles
ring_info = [(60968e3, "ATM", 0, -1e6),(676e5, "D68", 0, -2500),(7156e4, "D72", 2, -1e2),
             (7327e4, "D73", 54, -1e2),(7466e4, "C", 70, -1e3),(9198e4,"B", 55, -1e4 )]

# Loop through each item and add labels with specific position adjustments
for radius, label, angle_deg, radius_offset in ring_info:
    angle_rad = np.radians(angle_deg)  # Convert angle to radians
    adjusted_radius = radius + radius_offset  # Apply radial offset

    # Customize alignment and position
    axs_polar.text(
        angle_rad, adjusted_radius, label,
        ha='right', va='top', color='white', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='none', alpha=1))

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                             3D cartesian
# ----------------------------------------------------------------------------------
from matplotlib.ticker import FuncFormatter 
# Create the first 3D figure 
fig_3d_spherical = plt.figure(dpi=330) 
ax_3d_spherical = fig_3d_spherical.add_subplot(111, projection="3d") 

#Pane color (black) 
pane_color = (0.0, 0.0, 0.0, 1.0) 
ax_3d_spherical.xaxis.set_pane_color(pane_color) 
ax_3d_spherical.yaxis.set_pane_color(pane_color) 
ax_3d_spherical.zaxis.set_pane_color(pane_color) 

# Set labels for spherical coordinates 
ax_3d_spherical.set_xlabel('X', fontsize=10, labelpad=3) 
ax_3d_spherical.set_ylabel('Y', fontsize=10, labelpad=3) 
ax_3d_spherical.set_zlabel('Z', fontsize=10, labelpad=1)
 
# Customize tick colors, font size, and tick font size 
ax_3d_spherical.tick_params(axis='x', labelsize=10, pad=1) 
ax_3d_spherical.tick_params(axis='y', labelsize=10, pad=1) 
ax_3d_spherical.tick_params(axis='z', labelsize=10, pad=0.3) 

# Customize grid line colors and size 
ax_3d_spherical.grid(True, color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
#ax_3d_spherical.set_zlim(-11e3, 3e3) 

# Apply the ScalarFormatter to the z-axis 
z_formatter = ScalarFormatter(useMathText=True) 
z_formatter.set_powerlimits((0, 0)) 
ax_3d_spherical.zaxis.set_major_formatter(z_formatter) 
# Set the z-axis tick labels to two significant figures with one decimal place 
ax_3d_spherical.zaxis.set_tick_params(labelsize=10, pad=3)

# --------------------------------------------------------------------------------
# Plot trajectories in spherical polar coordinates
for i, (b_value, sol) in enumerate(trajectories.items()):
    color = colors[i]  # Assign color based on the index
    # zorder based on the custom mapping
    # z_order = b_zorders.get(b_value, 1)

    # 3D Trajectories around the sphere
    # Convert spherical coordinates to Cartesian coordinates
    r_sol     = sol.y[0, :]
    theta_sol = sol.y[2, :]
    phi_sol   = sol.y[4, :]
    x_traj    = r_sol * np.sin(theta_sol) *np.cos(phi_sol)
    y_traj    = r_sol * np.sin(theta_sol) *np.sin(phi_sol)
    z_traj    = r_sol * np.cos(theta_sol)

    # Plot the trajectories in Cartesian coordinates (x, y, z)
    ax_3d_spherical.plot3D(x_traj, y_traj, z_traj, label=f'{b_value:.1e}',
                           linewidth=2, linestyle='solid', color=color, alpha=1, zorder=7)  # ,zorder=z_order)

    ax_3d_spherical.plot3D(x_traj[-1], y_traj[-1], z_traj[-1], marker='o',
                           color=color, markeredgecolor='white', markersize=5, zorder=7)      # ,zorder=z_order)

## ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                       CARTESIAN with centre XY plane
# ------------------------------------------------------------------------------

# Create the main figure and axis
fig_cart, ax_cart = plt.subplots(dpi=360)
ax_cart.set_facecolor('black')

# Plot the trajectories (Ensure you have defined 'trajectories' and 'colors')
for idx, (b_value, sol) in enumerate(trajectories.items()):
     r_sol   = sol.y[0, :]
     phi_sol = sol.y[4, :]
     x_sol   = r_sol * np.cos(phi_sol)
     y_sol   = r_sol * np.sin(phi_sol)
     
     ax_cart.plot(x_sol, y_sol, linewidth=4, linestyle='solid', label=f'{b_value:.1e} m', color=colors[idx])
     ax_cart.plot(x_sol[-1], y_sol[-1], marker='o', color=colors[idx], markeredgecolor='white', markersize=8,zorder=12)

# Add filled circle at the center
R = 60268000  # Radius in meters
circle = plt.Circle((0, 0), R, color='#EE9A49', alpha=1, zorder=7)
ax_cart.add_artist(circle)

# Concentric circles and their labels with individual offsets
ring_info = [(61268e3,"ATM", np.radians(2), -1e4),(676e5,  "D68", np.radians(359.9), -500),
             (7156e4, "D72", np.radians(15), -800),(7327e4, "D73", np.radians(25),  -800),
             (7466e4, "C", np.radians(30), -800),(9198e4,"B",  np.radians(41), -1e3)]

# Add concentric circles and labels
for radius, label, angle, offset in ring_info:
    # Create a circle
    circle = plt.Circle((0, 0), radius, color='dimgray', linestyle='solid', linewidth=2, alpha=0.8, fill=False)
    ax_cart.add_artist(circle)
# Customize plot appearance
ax_cart.set_xlabel("X Distance (m)", fontsize=12)
ax_cart.set_ylabel("Y Distance (m)", fontsize=12)
ax_cart.tick_params(axis='x', labelsize=12)
ax_cart.tick_params(axis='y', labelsize=12)
#ax_cart.set_xlim(-1.5 * R, 1.5 * R)
#ax_cart.set_ylim(-1.5 * R, 1.5 * R)
ax_cart.grid(color='gray', linestyle='--', linewidth=0.6, alpha=1, zorder=1)

plt.tight_layout()
plt.show()
#::::::::::::::::::::::::::::::::::::::
# TIMVE VARIATIONS OF POSITIONAL VARIABLES AND SPEED COMPONENTS


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                        VISUALISATION
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                                   Time vs Radial vs Phi vs Theta speeds
# -------------------------------------------------------------------------------------------

# Create the main figure and axis
fig, ax_r = plt.subplots(figsize=(12, 8), dpi=270)
ax_r.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)

# Create a secondary x-axis for time
#ax_time = ax_r.twiny()

# Create a secondary y-axis for colatitude
ax_theta = ax_r.twinx()

# Plot on the main axis for each b_value in `trajectories`
for i, (b_value, sol) in enumerate(trajectories.items()):
    # z_order = b_zorders.get(b_value, 2)

    r_sol = sol.y[0, :]
    phi_sol = sol.y[4, :] *(180 /np.pi)  # Azimuthal angle in degrees
    # Convert to degrees for better readability
    theta_sol = sol.y[2, :] *(180 /np.pi)

    color = colors[i]  # Assign color based on the index
    
   
    # Plot radial distance vs azimuthal angle on the main axis
    ax_r.plot(phi_sol, r_sol, linewidth=2, linestyle='solid', alpha=1,
              color=color, label=f'{b_value:.1e}m')  # ,zorder=z_order)
    ax_r.plot(phi_sol[-1], r_sol[-1], 'o', color=color,
              markeredgecolor='white', markersize=5,zorder=10)  # ,zorder=z_order)

    ax_theta.plot(phi_sol, theta_sol, linewidth=2, linestyle='dotted',
                  alpha=1, color=color, label=f'{b_value:.1e}m')   # ,zorder=z_order)
    ax_theta.plot(phi_sol[-1], theta_sol[-1], 'D', color=color,
                  markeredgecolor='black', markersize=5,zorder=10)  # ,zorder=z_order)

# Set labels for the axes
ax_r.set_xlabel('Azimuth (deg)', fontsize=17)
ax_r.set_ylabel('Radial Distance (m)', fontsize=17)
ax_theta.set_ylabel('Colatitude (deg)',fontsize=17)
ax_theta.invert_yaxis()

#ax_time.set_xlabel('Time (s)', fontsize=18)
#ax_time.invert_xaxis()

# Custom formatter to display ticks in exponential form with a single scale at the corner
scalar_formatter = ScalarFormatter(useMathText=True)
scalar_formatter.set_powerlimits((0, 0))
scalar_formatter.set_scientific(True)
scalar_formatter.set_useOffset (False)

# Apply formatter to the axes
ax_r.xaxis.set_major_formatter(scalar_formatter)
#ax_time.xaxis.set_major_formatter(scalar_formatter)

# Customize tick parameters
ax_r.tick_params(axis='y', labelsize=17)
ax_r.tick_params(axis='x', labelsize=17)
ax_theta.tick_params(axis='y',labelsize=17)
#ax_time.tick_params(axis='x', labelsize=18)

# Customize grid lines
ax_theta.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)
#ax_time.grid(color='grey', linestyle='dotted', linewidth=0.6, alpha=1)

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# TIME vs RADIAL SPEED vs COLATITUDE SPEED

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Create the figure
fig1 = plt.figure(figsize=(6, 5), dpi=270)

# Time axis (only vertical grid lines)
ax1_time = fig1.add_subplot(111)
ax1_time.set_xlabel("Time (s)", fontsize=16)
ax1_time.tick_params(axis='x', labelsize=16, colors="black")
ax1_time.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax1_time.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)  # Only vertical grid
ax1_time.grid(False, which='major', axis='y')  # Explicitly disable horizontal grid

# Radial speed on right y-axis
ax1_radial = ax1_time.twinx()
ax1_radial.set_ylabel("Radial Speed (m/s)", fontsize=16, color="black")
ax1_radial.tick_params(axis='y', labelsize=16, colors="black")
ax1_radial.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1_radial.spines['left'].set_visible(False)
ax1_radial.grid(True, linestyle='solid', linewidth=0.5, color='silver')

# Colatitude speed on left y-axis
ax1_colatitude = ax1_time.twinx()
ax1_colatitude.spines['left'].set_position(('outward', 0))
ax1_colatitude.spines['left'].set_color('black')
ax1_colatitude.set_ylabel("Colatitudinal Speed (deg/s)", fontsize=16, color="black")
ax1_colatitude.tick_params(axis='y', labelsize=16, colors="black")
ax1_colatitude.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1_colatitude.spines['right'].set_visible(False)
ax1_colatitude.yaxis.set_label_position('left')
ax1_colatitude.yaxis.set_ticks_position('left')
ax1_colatitude.grid(True, linestyle='dashed', linewidth=0.5, color='black')

# Remove any residual y-ticks from the time axis
ax1_time.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# Plot data
for i, b_value in enumerate(b_values):
    sol = trajectories[b_value]
    time = sol.t
    color = colors[i]
    
    d_r = sol.y[1, :]# Radial speed
    d_theta = sol.y[3, :] * (180 / np.pi)  # Colatitude speed in degrees/s

    ax1_radial.plot(time, d_r, color=color, linewidth=2, linestyle='solid', label=f'{b_value:.1e} m',alpha=1)
    ax1_radial.plot(time[-1], d_r[-1], 'o', color=color, markeredgecolor='black', markersize=5, zorder=10)
    
    ax1_colatitude.plot(time, d_theta, color=color, linewidth=2, linestyle='dotted', label=f'{b_value:.1e} m',alpha=1)
    ax1_colatitude.plot(time[-1], d_theta[-1], 'D', color=color, markeredgecolor='black', markersize=5, zorder=10)

# Add legends
#ax1_colatitude.legend(title='Colatitudinal', fontsize=10, bbox_to_anchor=(1.35, 1.0))
#ax1_radial.legend(title='Radial', fontsize=10, bbox_to_anchor=(1.15, 1.0))

# Show the plot
plt.show()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TIME vs RADIAL SPEED vs AZIMUTHAL SPEED
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Create the figure
fig2 = plt.figure(figsize=(6, 5), dpi=270)

# Time axis (only vertical grid lines)
ax2_time = fig2.add_subplot(111)
ax2_time.set_xlabel("Time (s)", fontsize=16)
ax2_time.tick_params(axis='x', labelsize=16, colors="black")
ax2_time.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax2_time.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)  # Only vertical grid
ax2_time.grid(False, which='major', axis='y')  # Explicitly disable horizontal grid

# Radial speed on right y-axis
ax2_radial = ax2_time.twinx()
ax2_radial.set_ylabel("Radial Speed (m/s)", fontsize=16, color="black")
ax2_radial.tick_params(axis='y', labelsize=16, colors="black")
ax2_radial.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2_radial.spines['left'].set_visible(False)
ax2_radial.grid(True, linestyle='solid', linewidth=0.5, color='silver')

# Azimuthal speed on left y-axis
ax2_azimuthal = ax2_time.twinx()
ax2_azimuthal.spines['left'].set_position(('outward', 0))
ax2_azimuthal.spines['left'].set_color('black')
ax2_azimuthal.set_ylabel("Azimuthal Speed (deg/s)", fontsize=16, color="black")
ax2_azimuthal.tick_params(axis='y', labelsize=16, colors="black")
ax2_azimuthal.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax2_azimuthal.spines['right'].set_visible(False)
ax2_azimuthal.yaxis.set_label_position('left')
ax2_azimuthal.yaxis.set_ticks_position('left')
ax2_azimuthal.grid(True, linestyle='dashed', linewidth=0.5, color='black')

# Remove any residual y-ticks from the time axis
ax2_time.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# Plot data
for i, b_value in enumerate(b_values):
    sol = trajectories[b_value]
    time = sol.t
    color = colors[i]
    
    d_r = sol.y[1, :]# Radial speed
    d_phi = sol.y[5, :] * (180 / np.pi)# Azimuthal speed in degrees/s

    ax2_radial.plot(time, d_r, color=color, linewidth=5, linestyle='solid', label=f'{b_value:.1e} m',alpha=1)
    ax2_radial.plot(time[-1], d_r[-1], 'o', color=color, markeredgecolor='black', markersize=8, zorder=10)
    
    ax2_azimuthal.plot(time, d_phi, color=color, linewidth=5, linestyle=(5,(10,3)), label=f'{b_value:.1e} m',alpha=1)
    ax2_azimuthal.plot(time[-1], d_phi[-1], 'h', color=color, markeredgecolor='black', markersize=8, zorder=10)

# Add legends
#ax2_azimuthal.legend(title='Azimuthal', fontsize=10, bbox_to_anchor=(1.35, 1.0))
#ax2_radial.legend(title='Radial', fontsize=10, bbox_to_anchor=(1.15, 1.0))

# Show the plot
plt.show()
