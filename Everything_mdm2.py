import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
M_earth = 5.972e24  # Earth's mass, kg
R_earth = 6371000  # Earth's radius, m

rho_0 = 1.225  # Sea-level air density, kg/m^3
A = 10.75  # Rocket's cross-sectional area, m^2
mair = 4.81e-26  # The average mass of air per unit
T = 288.15  # Temperature, K
k = 1.38e-23  # Boltzmann constant, J/K
m0 = 75000.0  # Initial rocket mass, kg
fuel_mass = 0.3 * (75000 - 20569)  # Fuel mass, kg
m0 = 20569 + fuel_mass
v_exhaust = 2500.0  # Exhaust velocity, m/s

# Function to calculate air density as a function of altitude
def air_density(h):
    return rho_0 * np.exp(-mair * 9.81 * h / (k * T))

# Simulation parameters
dt = 0.05  # Time step, s
time_total = 500.0  # Total simulation time, s

# Initial conditions
initial_altitude = R_earth + 1120 + 101205.792  # Initial altitude, m
target_v = 840.4352  # Target velocity, m/s

# Range of drag coefficients
Cd_values = np.linspace(0.1, 6, 100)
final_velocities = []

# Simulation loop for each drag coefficient
for Cd in Cd_values:
    y = initial_altitude
    v = 0.0  # Initial velocity, m/s
    m = m0  # Initial mass, kg
    for t in np.arange(0, time_total, dt):
        g = G * M_earth / y**2
        drag = 0.5 * air_density(y - R_earth) * A * Cd * v**2
        drag *= np.sign(v)
        dv = (0 - drag - m * g) / m * dt
        v += dv
        dy = v * dt
        y += dy
        if y - R_earth - 1120 <= 64175.3352:
            break
    final_velocities.append(v)

final_velocities = [value * -1 for value in final_velocities]
# Find the drag coefficient that produces the velocity closest to the target
closest_index_ff = np.argmin(np.abs(np.array(final_velocities) - target_v))
closest_Cd_ff = Cd_values[closest_index_ff]
closest_v_ff = final_velocities[closest_index_ff]

# Simulation parameters
dt = 0.05  # Time step, s
time_total = 500.0  # Total simulation time, s

# Initial conditions
initial_altitude = R_earth + 1120 + 64175.3352  # Initial altitude, m
target_v = 283.8704  # Target velocity, m/s

# Range of drag coefficients
Cd_values = np.linspace(0.1, 6, 100)
final_velocities = []

# Simulation loop for each drag coefficient
for Cd in Cd_values:
    y = initial_altitude
    v = 840.4352  # Initial velocity, m/s
    m = m0  # Initial mass, kg
    for t in np.arange(0, time_total, dt):
        g = G * M_earth / y**2
        drag = 0.5 * air_density(y - R_earth) * A * Cd * v**2
        drag *= np.sign(v)
        dv = (0 - drag - m * g) / m * dt
        v += dv
        dy = v * dt
        y += dy
        if y - R_earth - 1120 <= 4861.2552:
            break
    final_velocities.append(v)


final_velocities = [value * -1 for value in final_velocities]
# Find the drag coefficient that produces the velocity closest to the target
closest_index_wf = np.argmin(np.abs(np.array(final_velocities) - target_v))
closest_Cd_wf = Cd_values[closest_index_wf]
closest_v_wf = final_velocities[closest_index_wf]

# Simulation parameters
dt = 0.05  # Time step, s
time_total = 500.0  # Total simulation time, s

# Initial conditions
initial_altitude = R_earth + 1120 + 4861.2552  # Initial altitude, m
target_v = 164.06368  # Target velocity, m/s

# Range of drag coefficients
Cd_values = np.linspace(0.1, 6, 100)
final_velocities = []

# Simulation loop for each drag coefficient
for Cd in Cd_values:
    y = initial_altitude
    v = 283.8704  # Initial velocity, m/s
    m = m0  # Initial mass, kg
    for t in np.arange(0, time_total, dt):
        g = G * M_earth / y**2
        drag = 0.5 * air_density(y - R_earth) * A * Cd * v**2
        drag *= np.sign(v)
        dv = (0 - drag - m * g) / m * dt
        v += dv
        dy = v * dt
        y += dy
        if y - R_earth - 1120 <= 876.3:
            break
    final_velocities.append(v)


final_velocities = [value * -1 for value in final_velocities]
# Find the drag coefficient that produces the velocity closest to the target
closest_index_db = np.argmin(np.abs(np.array(final_velocities) - target_v))
closest_Cd_db = Cd_values[closest_index_db]
closest_v_db = final_velocities[closest_index_db]



# Function to calculate air density as a function of altitude
def air_density(h):
    return rho_0 * np.exp(-mair * 9.81 * h / (k * T))

# Simulation parameters
dt = 0.05  # Time step, s
time_total = 500.0  # Total simulation time, s

# Thrust and activation height ranges for the 3D plot
thrust_values = np.linspace(80000, 500000, 30)  # Thrust values to test, N
activation_heights = np.linspace(0, 4861.2552, 30)  # Activation heights to test, m

# Store results for 3D plot
thrust_data = []
activation_height_data = []
landing_velocity_data = []

# Simulation for each thrust and activation height combination
for F_thrust in thrust_values:
    #print(F_thrust)
    for h_activation in activation_heights:
        # Initial conditions
        y = R_earth + 1120 + 101205.792  # Initial altitude, m
        v = 0.0  # Initial velocity, m/s
        m = m0  # Initial mass, kg
        fuel_consumed = 0.0  # Initial fuel consumption

        # Simulation loop
        for t in np.arange(0, time_total, dt):
            # Calculate gravity at the current altitude
            g = G * M_earth / y**2
            if R_earth + 1120 + 4861.2552 < y <= R_earth + 1120 + 64175.3352:
                Cd = closest_Cd_wf
            elif R_earth + 1120 + 0 <= y <= R_earth + 1120 + 4861.2552:
                Cd = closest_Cd_db
            else:
                Cd = 0.75

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

min_index = np.argmin(landing_velocity_data)

min_velocity = landing_velocity_data[min_index]
min_thrust = thrust_data[min_index]
min_activation_height = activation_height_data[min_index]

print(f"Minimum Landing Velocity: {min_velocity} m/s")
print(f"Corresponding Thrust: {min_thrust} N")
print(f"Corresponding Activation Height: {min_activation_height} m")

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot data
ax.plot_trisurf(thrust_data, activation_height_data, landing_velocity_data, cmap='viridis')
ax.set_xlabel("Thrust (N)")
ax.set_ylabel("Activation Height (m)")
ax.set_zlabel("Landing Velocity (m/s)")
ax.set_title("Landing Velocity as a Function of Thrust and Activation Height")


# Function to calculate air density as a function of altitude
def air_density(h):
    return rho_0 * np.exp(-mair * 9.81 * h / (k * T))

# Simulation parameters
dt = 0.05  # Time step, s
time_total = 500.0  # Total simulation time, s

# Thrust and activation height ranges for the 3D plot
thrust_values = np.linspace(80000, 500000, 20)  # Thrust values to test, N
activation_heights = np.linspace(0, 4861.2552, 20)  # Activation heights to test, m

# Store results for heatmap
landing_velocity_matrix = np.zeros((len(thrust_values), len(activation_heights)))

# Simulation for each thrust and activation height combination
for i, F_thrust in enumerate(thrust_values):
    for j, h_activation in enumerate(activation_heights):
        # Initial conditions
        y = R_earth + 1120 + 101205.792  # Initial altitude, m
        v = 0.0  # Initial velocity, m/s
        m = m0  # Initial mass, kg
        fuel_consumed = 0.0  # Initial fuel consumption

        # Simulation loop
        for t in np.arange(0, time_total, dt):
            # Calculate gravity at the current altitude
            g = G * M_earth / y**2
            if R_earth + 1120 + 4861.2552 < y <= R_earth + 1120 + 64175.3352:
                Cd = closest_Cd_wf
            elif R_earth + 1120 + 0 <= y <= R_earth + 1120 + 4861.2552:
                Cd = closest_Cd_db
            else:
                Cd = 0.75

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

        # Store the final landing velocity
        landing_velocity_matrix[i, j] = abs(v)  # Use absolute value for landing speed

# Create heatmap
plt.figure(figsize=(10, 7))
plt.imshow(landing_velocity_matrix, origin='lower', aspect='auto', cmap='viridis', 
           extent=[activation_heights.min(), activation_heights.max(), thrust_values.min(), thrust_values.max()])
plt.colorbar(label='Landing Velocity (m/s)')
plt.xlabel('Activation Height (m)')
plt.ylabel('Thrust (N)')
plt.title('Confusion Matrix of Landing Velocity as a Function of Thrust and Activation Height')
plt.show()
