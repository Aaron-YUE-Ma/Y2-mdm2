import numpy as np
import matplotlib.pyplot as plt

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
F_thrust = 600000.0  # Thrust force, N
v_exhaust = 3000.0  # Exhaust velocity, m/s
T = 288.15  # Temperature, K
k = 1.38e-23  # Boltzmann constant, J/K
h_activation = 1500  # Altitude to activate thrust, m

# Function to calculate air density as a function of altitude
def air_density(h):
    return rho_0 * np.exp(-mair * 9.81 * h / (k * T))

# Simulation time step and total time
dt = 0.1  # Time step, s
time_total = 500.0  # Total simulation time, s

# Initial conditions
y = R_earth + 100000.0  # Initial altitude, m
v = 0.0  # Initial velocity, m/s
m = m0  # Initial mass, kg
fuel_consumed = 0.0  # Initial fuel consumption

# Data storage
altitude_data = []
velocity_data = []
mass_data = []

# Main simulation loop
for t in np.arange(0, time_total, dt):
    altitude_data.append(y - R_earth)
    velocity_data.append(v)
    mass_data.append(m)

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

# Plot velocity vs. altitude
plt.figure(figsize=(10, 5))
plt.plot(altitude_data, velocity_data)
plt.xlabel("Altitude (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Rocket Descent Simulation")
plt.gca().invert_xaxis()  # Invert x-axis from high to low altitude
plt.show()

# Plot mass vs. altitude
plt.figure(figsize=(10, 5))
plt.plot(altitude_data, mass_data)
plt.xlabel("Altitude (m)")
plt.ylabel("Mass (kg)")
plt.title("Rocket Mass vs. Altitude")
plt.gca().invert_xaxis()  # Invert x-axis from high to low altitude
plt.show()
