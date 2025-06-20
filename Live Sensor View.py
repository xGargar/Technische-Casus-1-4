import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import time
import csv
import os

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# === Settings ===
SIMULATION_MODE = False  # Set to False to use real sensors
SERIAL_PORT = 'COM3'
BAUDRATE = 115200
N_SENSORS = 8
BUFFER_SIZE = 100  # Max frames to keep in buffer
SENSITIVITY= 1
QUIVER_SENSITIVITY = SENSITIVITY * 10
SIM_AMPLITUDE = 25
RADIUS = 7  # cm

# === Sensor Geometry (circular layout on XY plane) ===
angle_steps = np.linspace(0, 2 * np.pi, N_SENSORS, endpoint=False)
center = np.array([10, 10])
sensor_xy = np.column_stack((center[0] + RADIUS * np.cos(angle_steps),
                             center[1] + RADIUS * np.sin(angle_steps)))
sensor_z = np.zeros(sensor_xy.shape[0])
sensor_positions = np.column_stack((sensor_xy, sensor_z))

# === Sensor Axes (Z axis radially outward) ===
sensor_z_dirs = sensor_xy - center
sensor_z_dirs = np.column_stack((sensor_z_dirs, np.zeros(sensor_z_dirs.shape[0])))
sensor_z_dirs = sensor_z_dirs / np.linalg.norm(sensor_z_dirs, axis=1)[:, np.newaxis]

# X axis: +90 degrees rotation in XY plane from Z axis
sensor_x_dirs = np.column_stack((-sensor_z_dirs[:,1], sensor_z_dirs[:,0], np.zeros(sensor_z_dirs.shape[0])))

# Y axis: out of XY plane (up)
sensor_y_dirs = np.tile([0, 0, 1], (sensor_z_dirs.shape[0], 1))

arrow_length = 1.5  # Adjust for visibility

# === Serial Port Setup ===
if not SIMULATION_MODE:
    import serial
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
else:
    ser = None

# === Data Buffer ===
timestamps = []
data_buffer = []

# === Grid Setup ===
nx, ny, nz = 50, 50, 10
x = np.linspace(0, 20, nx)
y = np.linspace(0, 20, ny)
z = np.linspace(-5, 5, nz)
grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
slice_z_idx = nz // 2
slice_y_idx = ny // 2
slice_x_idx = nx // 2

skip = 3  # For quiver plot density

# === CSV Logging Functions ===
def init_csv_logger(filename, n_sensors):
    header = ['timestamp']
    for i in range(1, n_sensors + 1):
        header += [f'sensor{i}_bx', f'sensor{i}_by', f'sensor{i}_bz']
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_csv_logger(filename, timestamp, sensor_vals):
    row = [timestamp]
    for vals in sensor_vals:
        row.extend(vals)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

csv_filename = time.strftime(r"Recorded Sensor Data\live_sensor_log_%Y%m%d_%H%M%S.csv")
init_csv_logger(csv_filename, N_SENSORS)

# === Read 8 XYZ vectors from Serial or Dummy Data ===
def read_xyz_values():
    if SIMULATION_MODE:
        t = time.time()
        sensor_vals = np.zeros((N_SENSORS, 3))
        for i in range(N_SENSORS):
            sensor_vals[i, 0] = SIM_AMPLITUDE * np.sin(t + i)
            sensor_vals[i, 1] = SIM_AMPLITUDE * np.cos(t + i)
            sensor_vals[i, 2] = SIM_AMPLITUDE * np.sin(t + i / 2)
        return sensor_vals
    xyz = []
    start = time.time()
    while len(xyz) < N_SENSORS and (time.time() - start) < 0.5:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if "X=" in line and "Y=" in line and "Z=" in line:
                try:
                    x = float(line.split("X=")[1].split("uT")[0].split(",")[0].strip())
                    y = float(line.split("Y=")[1].split("uT")[0].split(",")[0].strip())
                    z = float(line.split("Z=")[1].split("uT")[0].strip())
                    xyz.append([x, y, z])
                except:
                    continue
    while len(xyz) < N_SENSORS:
        xyz.append([0, 0, 0])
    return np.array(xyz)

# === RBF Interpolation ===
def interpolate_field(sensor_pos, sensor_vals):
    rbf = Rbf(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2], sensor_vals/1000,
              function='multiquadric', smooth=0.5, epsilon=2)
    return rbf(grid_x, grid_y, grid_z)

# === Plot Setup ===
fig = plt.figure(figsize=(22, 6))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.5])
ax_xy = fig.add_subplot(gs[0])
ax_xz = fig.add_subplot(gs[1])
ax_yz = fig.add_subplot(gs[2])
ax3d = fig.add_subplot(gs[3], projection='3d')
axs = [ax_xy, ax_xz, ax_yz]
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9, wspace=0.25)

# Prepare grid for quiver plots
X_xy, Y_xy = np.meshgrid(x[::skip], y[::skip], indexing='ij')
X_xz, Z_xz = np.meshgrid(x[::skip], z[::skip], indexing='ij')
Y_yz, Z_yz = np.meshgrid(y[::skip], z[::skip], indexing='ij')

# Initial dummy data for heatmaps and quivers
dummy_Bmag = np.zeros((nx, ny, nz))
dummy_Bx = np.zeros((nx, ny, nz))
dummy_By = np.zeros((nx, ny, nz))
dummy_Bz = np.zeros((nx, ny, nz))

# XY slice heatmap and quiver
heatmap_xy = ax_xy.imshow(dummy_Bmag[:, :, slice_z_idx].T, origin='lower',
                           extent=(x[0], x[-1], y[0], y[-1]), cmap='inferno', alpha=0.8, vmin=0, vmax=SENSITIVITY)
Q_xy = ax_xy.quiver(X_xy, Y_xy, dummy_Bx[::skip, ::skip, slice_z_idx], dummy_By[::skip, ::skip, slice_z_idx],
                     scale=QUIVER_SENSITIVITY, color='cyan')
ax_xy.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='black', s=40)
for pos, x_dir, y_dir, z_dir in zip(sensor_positions, sensor_x_dirs, sensor_y_dirs, sensor_z_dirs):
    # X axis (red)
    ax_xy.arrow(pos[0], pos[1], x_dir[0]*arrow_length, x_dir[1]*arrow_length,
                head_width=0.4, head_length=0.6, fc='red', ec='red', length_includes_head=True)
    # Y axis (green)
    ax_xy.arrow(pos[0], pos[1], y_dir[0]*arrow_length, y_dir[1]*arrow_length,
                head_width=0.4, head_length=0.6, fc='green', ec='green', length_includes_head=True)
    # Z axis (blue)
    ax_xy.arrow(pos[0], pos[1], z_dir[0]*arrow_length, z_dir[1]*arrow_length,
                head_width=0.4, head_length=0.6, fc='blue', ec='blue', length_includes_head=True)
ax_xy.set_title('XY Slice')
ax_xy.set_xlabel('X (cm)')
ax_xy.set_ylabel('Y (cm)')
fig.colorbar(heatmap_xy, ax=ax_xy).set_label('|B| magnitude (mT)')

# XZ slice heatmap and quiver
heatmap_xz = ax_xz.imshow(dummy_Bmag[:, slice_y_idx, :].T, origin='lower',
                           extent=(x[0], x[-1], z[0], z[-1]), cmap='inferno', alpha=0.8, vmin=0, vmax=SENSITIVITY)
Q_xz = ax_xz.quiver(X_xz, Z_xz, dummy_Bx[::skip, slice_y_idx, ::skip], dummy_Bz[::skip, slice_y_idx, ::skip],
                     scale=QUIVER_SENSITIVITY, color='cyan')
ax_xz.scatter(sensor_positions[:, 0], sensor_positions[:, 2], c='black', s=40)
for pos, x_dir, y_dir, z_dir in zip(sensor_positions, sensor_x_dirs, sensor_y_dirs, sensor_z_dirs):
    # X axis (red)
    ax_xz.arrow(pos[0], pos[2], x_dir[0]*arrow_length, x_dir[2]*arrow_length,
                head_width=0.4, head_length=0.6, fc='red', ec='red', length_includes_head=True)
    # Y axis (green)
    ax_xz.arrow(pos[0], pos[2], y_dir[0]*arrow_length, y_dir[2]*arrow_length,
                head_width=0.4, head_length=0.6, fc='green', ec='green', length_includes_head=True)
    # Z axis (blue)
    ax_xz.arrow(pos[0], pos[2], z_dir[0]*arrow_length, z_dir[2]*arrow_length,
                head_width=0.4, head_length=0.6, fc='blue', ec='blue', length_includes_head=True)

ax_xz.set_title('XZ Slice')
ax_xz.set_xlabel('X (cm)')
ax_xz.set_ylabel('Z (cm)')
fig.colorbar(heatmap_xz, ax=ax_xz).set_label('|B| magnitude (mT)')

# YZ slice heatmap and quiver
heatmap_yz = ax_yz.imshow(dummy_Bmag[slice_x_idx, :, :].T, origin='lower',
                           extent=(y[0], y[-1], z[0], z[-1]), cmap='inferno', alpha=0.8, vmin=0, vmax=SENSITIVITY)
Q_yz = ax_yz.quiver(Y_yz, Z_yz, dummy_By[slice_x_idx, ::skip, ::skip], dummy_Bz[slice_x_idx, ::skip, ::skip],
                     scale=QUIVER_SENSITIVITY, color='cyan')
ax_yz.scatter(sensor_positions[:, 1], sensor_positions[:, 2], c='black', s=40)
for pos, x_dir, y_dir, z_dir in zip(sensor_positions, sensor_x_dirs, sensor_y_dirs, sensor_z_dirs):
    # X axis (red)
    ax_yz.arrow(pos[1], pos[2], x_dir[1]*arrow_length, x_dir[2]*arrow_length,
                head_width=0.4, head_length=0.6, fc='red', ec='red', length_includes_head=True)
    # Y axis (green)
    ax_yz.arrow(pos[1], pos[2], y_dir[1]*arrow_length, y_dir[2]*arrow_length,
                head_width=0.4, head_length=0.6, fc='green', ec='green', length_includes_head=True)
    # Z axis (blue)
    ax_yz.arrow(pos[1], pos[2], z_dir[1]*arrow_length, z_dir[2]*arrow_length,
                head_width=0.4, head_length=0.6, fc='blue', ec='blue', length_includes_head=True)

ax_yz.set_title('YZ Slice')
ax_yz.set_xlabel('Y (cm)')
ax_yz.set_ylabel('Z (cm)')
fig.colorbar(heatmap_yz, ax=ax_yz).set_label('|B| magnitude (mT)')

# === Animation Update ===
def update(frame):
    sensor_vals = read_xyz_values()
    timestamp = time.time()

    # Save sensor data to CSV
    append_csv_logger(csv_filename, timestamp, sensor_vals)

    # Store in buffer
    timestamps.append(timestamp)
    data_buffer.append(sensor_vals)
    if len(data_buffer) > BUFFER_SIZE:
        data_buffer.pop(0)
        timestamps.pop(0)

    Bx = interpolate_field(sensor_positions, sensor_vals[:, 0])
    By = interpolate_field(sensor_positions, sensor_vals[:, 1])
    Bz = interpolate_field(sensor_positions, sensor_vals[:, 2])
    Bmag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    heatmap_xy.set_data(Bmag[:, :, slice_z_idx].T)
    Q_xy.set_UVC(Bx[::skip, ::skip, slice_z_idx], By[::skip, ::skip, slice_z_idx])

    heatmap_xz.set_data(Bmag[:, slice_y_idx, :].T)
    Q_xz.set_UVC(Bx[::skip, slice_y_idx, ::skip], Bz[::skip, slice_y_idx, ::skip])

    heatmap_yz.set_data(Bmag[slice_x_idx, :, :].T)
    Q_yz.set_UVC(By[slice_x_idx, ::skip, ::skip], Bz[slice_x_idx, ::skip, ::skip])

    # 3D Surface Plot (|B| modulus)
    ax3d.clear()
    surf = ax3d.plot_surface(
        grid_x[:, :, slice_z_idx], grid_y[:, :, slice_z_idx], Bmag[:, :, slice_z_idx],
        cmap='inferno', vmin=0, vmax=SENSITIVITY, edgecolor='none', alpha=0.9
    )
    ax3d.scatter(sensor_positions[:, 0], sensor_positions[:, 1], np.linalg.norm(sensor_vals, axis=1), color='red', s=50, label='Sensors')
    ax3d.set_xlabel('X (cm)')
    ax3d.set_ylabel('Y (cm)')
    ax3d.set_zlabel('|B|')
    ax3d.set_title('Magnitude|B| (mT))')
    ax3d.set_zlim(0, SENSITIVITY)
    ax3d.legend(loc='upper left')

    return heatmap_xy, Q_xy, heatmap_xz, Q_xz, heatmap_yz, Q_yz, surf

ani = FuncAnimation(fig, update, interval=500)
plt.show()

if ser is not None:
    ser.close()