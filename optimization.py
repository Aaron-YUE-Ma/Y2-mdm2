import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
M_earth = 5.972e24  # Earth's mass, kg
R_earth = 6371000  # Earth's radius, m

rho_0 = 1.225  # Sea-level air density, kg/m^3
Cd = 0.75  # Drag coefficient
A = 10.75  # Rocket's cross-sectional area, m^2
m0 = 22200.0  # Initial rocket mass, kg
mair = 4.81e-26 #the average mass of air per unit
fuel_mass = 41000.0  # Fuel mass, kg
v_exhaust = 3100.0  # Exhaust velocity, m/s
T = 288.15  # Temperature, K
k = 1.38e-23  # Boltzmann constant, J/K

# Function to calculate air density as a function of altitude
def air_density(h):
    return rho_0 * np.exp(-mair * 9.81 * h / (k * T))

# Simulation parameters
dt = 0.05  # Time step, s
time_total = 500.0  # Total simulation time, s

# Thrust and activation height ranges for the 3D plot
thrust_values = np.linspace(300000, 600000, 50)  # Thrust values to test, N
activation_heights = np.linspace(0, 10000, 50)  # Activation heights to test, m

# Store results for 3D plot
thrust_data = []
activation_height_data = []
landing_velocity_data = []

# Simulation for each thrust and activation height combination
for F_thrust in thrust_values:
    for h_activation in activation_heights:
        # Initial conditions
        y = R_earth + 100000.0  # Initial altitude, m
        v = 0.0  # Initial velocity, m/s
        m = m0  # Initial mass, kg
        fuel_consumed = 0.0  # Initial fuel consumption

        # Simulation loop
        for t in np.arange(0, time_total, dt):
            # Calculate gravity at the current altitude
            g = G * M_earth / y**2

            # Calculate air drag
            drag = 0.5 * air_density(y - R_earth) * A * Cd * v**2
            drag *= np.sign(v)  # Ensure drag is opposite to velocity direction

            # Calculate thrust: apply only when below activation altitude and fuel remains
            if y - R_earth <= h_activation and fuel_consumed < fuel_mass:
                thrust = F_thrust
                dm_dt = F_thrust / v_exhaust  # Fuel consumption rate
            else:
                thrust = 0
                dm_dt = 0

            # Update velocity and altitude
            dv = (thrust - drag - m * g) / m * dt
            v += dv
            dy = v * dt
            y += dy

            # Update rocket mass due to fuel consumption
            dm = dm_dt * dt
            m -= dm
            fuel_consumed += dm

            # End simulation if rocket reaches the ground
            if y <= R_earth:
                break

        # Store the results
        thrust_data.append(F_thrust)
        activation_height_data.append(h_activation)
        landing_velocity_data.append(abs(v))  # Use absolute value for landing speed

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot data
ax.plot_trisurf(thrust_data, activation_height_data, landing_velocity_data, cmap='viridis')
ax.set_xlabel("Thrust (N)")
ax.set_ylabel("Activation Height (m)")
ax.set_zlabel("Landing Velocity (m/s)")
ax.set_title("Landing Velocity as a Function of Thrust and Activation Height")

plt.show()

