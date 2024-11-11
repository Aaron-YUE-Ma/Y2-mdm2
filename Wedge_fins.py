import numpy as np
import matplotlib.pyplot as plt

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
m0 = 75000.0  # Initial rocket mass, kg
fuel_mass = 75000 - 20569  # Fuel mass, kg
m0 = 20569 + 0.3 * fuel_mass

# Function to calculate air density as a function of altitude
def air_density(h):
    return rho_0 * np.exp(-mair * 9.81 * h / (k * T))

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
closest_index = np.argmin(np.abs(np.array(final_velocities) - target_v))
closest_Cd = Cd_values[closest_index]
closest_v = final_velocities[closest_index]

print(f"Drag coefficient (Cd) producing velocity closest to {target_v}: {closest_Cd}")
print(f"Final velocity for this Cd: {closest_v}")

# Plot final velocity against drag coefficient
plt.plot(Cd_values, final_velocities, label='Final Velocity')
plt.axhline(y=target_v, color='r', linestyle='--', label='Target Velocity')
plt.axvline(x=closest_Cd, color='g', linestyle='--', label='Closest Cd')
plt.xlabel('Drag Coefficient (Cd)')
plt.ylabel('Final Velocity (v) [m/s]')
plt.title('Final Velocity vs. Drag Coefficient')
plt.legend()
plt.grid(True)
plt.show()
