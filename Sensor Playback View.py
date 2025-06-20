import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv
from scipy.interpolate import Rbf
from tkinter import Tk, filedialog

#Settings
SENSITIVITY = 1
QUIVER_SENSITIVITY = SENSITIVITY * 10
SIM_AMPLITUDE = 25
N_SENSORS = 8  # Number of sensors
RADIUS = 7  # cm

# Sensor positions
angle_steps = np.linspace(0, 2 * np.pi, N_SENSORS, endpoint=False)
center = np.array([10, 10])
sensor_xy = np.column_stack((center[0] + RADIUS * np.cos(angle_steps),
                             center[1] + RADIUS * np.sin(angle_steps)))
sensor_z = np.zeros(sensor_xy.shape[0])  # All sensors on z=0 plane
sensor_positions = np.column_stack((sensor_xy, sensor_z))

# Sensor directions
sensor_z_dirs = sensor_xy - center
sensor_z_dirs = np.column_stack((sensor_z_dirs, np.zeros(sensor_z_dirs.shape[0])))
sensor_z_dirs = sensor_z_dirs / np.linalg.norm(sensor_z_dirs, axis=1)[:, np.newaxis]
sensor_y_dirs = np.tile([0, 0, 1], (sensor_z_dirs.shape[0], 1))
sensor_x_dirs = np.cross(sensor_y_dirs, sensor_z_dirs)
arrow_length = 1.5  

# Read CSV file
def load_csv_data(filename):
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        timestamps = []
        data = []
        for row in reader:
            timestamps.append(float(row['timestamp']))
            bx = np.array([float(row[f'sensor{i}_bx']) for i in range(1, 9)])
            by = np.array([float(row[f'sensor{i}_by']) for i in range(1, 9)])
            bz = np.array([float(row[f'sensor{i}_bz']) for i in range(1, 9)])
            data.append(np.stack([bx, by, bz], axis=1))
    return np.array(timestamps), np.array(data)

# Interpolate 
def interpolate_3d_field(sensor_pos, sensor_vals, grid_x, grid_y, grid_z):
    rbf = Rbf(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2], sensor_vals/1000,
              function='multiquadric', smooth=0.5, epsilon=2)
    return rbf(grid_x, grid_y, grid_z)

# Choose file
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not filename:
        print("No file selected.")
        exit()

    timestamps, data = load_csv_data(filename)

    # Graph setup
    nx, ny, nz = 50, 50, 10
    x = np.linspace(0, 20, nx)
    y = np.linspace(0, 20, ny)
    z = np.linspace(-5, 5, nz)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    time_idx = 0
    slice_z_idx = nz // 2
    slice_y_idx = ny // 2
    slice_x_idx = nx // 2

    sensor_B = data[time_idx]
    Bx_grid = interpolate_3d_field(sensor_positions, sensor_B[:, 0], grid_x, grid_y, grid_z)
    By_grid = interpolate_3d_field(sensor_positions, sensor_B[:, 1], grid_x, grid_y, grid_z)
    Bz_grid = interpolate_3d_field(sensor_positions, sensor_B[:, 2], grid_x, grid_y, grid_z)
    Bmag_grid = np.sqrt(Bx_grid**2 + By_grid**2 + Bz_grid**2)

    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2])
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    ax3d = fig.add_subplot(gs[3], projection='3d')
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9, wspace=0.3)
    skip = 3

    # XY slice
    heatmap_xy = axs[0].imshow(Bmag_grid[:, :, slice_z_idx].T, origin='lower',
                               extent=(x[0], x[-1], y[0], y[-1]), cmap='inferno', alpha=0.8, vmin=0, vmax=SENSITIVITY)
    Q_xy = axs[0].quiver(grid_x[::skip, ::skip, slice_z_idx],
                         grid_y[::skip, ::skip, slice_z_idx],
                         Bx_grid[::skip, ::skip, slice_z_idx],
                         By_grid[::skip, ::skip, slice_z_idx],
                         scale=QUIVER_SENSITIVITY, color='cyan')
    # Draw sensor axes
    for pos, x_dir, y_dir, z_dir in zip(sensor_positions, sensor_x_dirs, sensor_y_dirs, sensor_z_dirs):
        axs[0].arrow(pos[0], pos[1], x_dir[0]*arrow_length, x_dir[1]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='red', ec='red', length_includes_head=True)
        axs[0].arrow(pos[0], pos[1], y_dir[0]*arrow_length, y_dir[1]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='green', ec='green', length_includes_head=True)
        axs[0].arrow(pos[0], pos[1], z_dir[0]*arrow_length, z_dir[1]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='blue', ec='blue', length_includes_head=True)
    axs[0].scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='black', s=40, label='Sensors')
    axs[0].set_title(f'XY slice at Z={z[slice_z_idx]:.2f} cm, Time={timestamps[time_idx]:.2f}')
    axs[0].set_xlabel('X (cm)')
    axs[0].set_ylabel('Y (cm)')
    axs[0].legend()
    fig.colorbar(heatmap_xy, ax=axs[0]).set_label('|B| magnitude (mT)')

    # XZ slice
    heatmap_xz = axs[1].imshow(Bmag_grid[:, slice_y_idx, :].T, origin='lower',
                               extent=(x[0], x[-1], z[0], z[-1]), cmap='inferno', alpha=0.8, vmin=0, vmax=SENSITIVITY)
    Q_xz = axs[1].quiver(grid_x[::skip, slice_y_idx, ::skip],
                         grid_z[::skip, slice_y_idx, ::skip],
                         Bx_grid[::skip, slice_y_idx, ::skip],
                         Bz_grid[::skip, slice_y_idx, ::skip],
                         scale=QUIVER_SENSITIVITY, color='cyan')
    for pos, x_dir, y_dir, z_dir in zip(sensor_positions, sensor_x_dirs, sensor_y_dirs, sensor_z_dirs):
        axs[1].arrow(pos[0], pos[2], x_dir[0]*arrow_length, x_dir[2]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='red', ec='red', length_includes_head=True)
        axs[1].arrow(pos[0], pos[2], y_dir[0]*arrow_length, y_dir[2]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='green', ec='green', length_includes_head=True)
        axs[1].arrow(pos[0], pos[2], z_dir[0]*arrow_length, z_dir[2]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='blue', ec='blue', length_includes_head=True)
    axs[1].scatter(sensor_positions[:, 0], sensor_positions[:, 2], c='black', s=40)
    axs[1].set_title(f'XZ slice at Y={y[slice_y_idx]:.2f} cm, Time={timestamps[time_idx]:.2f}')
    axs[1].set_xlabel('X (cm)')
    axs[1].set_ylabel('Z (cm)')
    axs[1].legend()
    fig.colorbar(heatmap_xz, ax=axs[1]).set_label('|B| magnitude (mT)')

    # YZ slice
    heatmap_yz = axs[2].imshow(Bmag_grid[slice_x_idx, :, :].T, origin='lower',
                               extent=(y[0], y[-1], z[0], z[-1]), cmap='inferno', alpha=0.8, vmin=0, vmax=SENSITIVITY)
    Q_yz = axs[2].quiver(grid_y[slice_x_idx, ::skip, ::skip],
                         grid_z[slice_x_idx, ::skip, ::skip],
                         By_grid[slice_x_idx, ::skip, ::skip],
                         Bz_grid[slice_x_idx, ::skip, ::skip],
                         scale=QUIVER_SENSITIVITY, color='cyan')
    for pos, x_dir, y_dir, z_dir in zip(sensor_positions, sensor_x_dirs, sensor_y_dirs, sensor_z_dirs):
        axs[2].arrow(pos[1], pos[2], x_dir[1]*arrow_length, x_dir[2]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='red', ec='red', length_includes_head=True)
        axs[2].arrow(pos[1], pos[2], y_dir[1]*arrow_length, y_dir[2]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='green', ec='green', length_includes_head=True)
        axs[2].arrow(pos[1], pos[2], z_dir[1]*arrow_length, z_dir[2]*arrow_length,
                     head_width=0.4, head_length=0.6, fc='blue', ec='blue', length_includes_head=True)
    axs[2].scatter(sensor_positions[:, 1], sensor_positions[:, 2], c='black', s=40)
    axs[2].set_title(f'YZ slice at X={x[slice_x_idx]:.2f} cm, Time={timestamps[time_idx]:.2f}')
    axs[2].set_xlabel('Y (cm)')
    axs[2].set_ylabel('Z (cm)')
    axs[2].legend()
    fig.colorbar(heatmap_yz, ax=axs[2]).set_label('|B| magnitude (mT)')


    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_time = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=axcolor)
    ax_slice_z = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
    ax_slice_y = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
    ax_slice_x = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)

    time_slider = Slider(ax_time, 'Time', 0, len(timestamps)-1, valinit=time_idx, valstep=1)
    slice_z_slider = Slider(ax_slice_z, 'Z Slice', 0, nz-1, valinit=slice_z_idx, valstep=1)
    slice_y_slider = Slider(ax_slice_y, 'Y Slice', 0, ny-1, valinit=slice_y_idx, valstep=1)
    slice_x_slider = Slider(ax_slice_x, 'X Slice', 0, nx-1, valinit=slice_x_idx, valstep=1)

    #Update graphs
    def update(val):
        idx_z = int(slice_z_slider.val)
        idx_y = int(slice_y_slider.val)
        idx_x = int(slice_x_slider.val)
        t_idx = int(time_slider.val)

        sensor_B = data[t_idx]
        Bx = interpolate_3d_field(sensor_positions, sensor_B[:, 0], grid_x, grid_y, grid_z)
        By = interpolate_3d_field(sensor_positions, sensor_B[:, 1], grid_x, grid_y, grid_z)
        Bz = interpolate_3d_field(sensor_positions, sensor_B[:, 2], grid_x, grid_y, grid_z)
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

        heatmap_xy.set_data(Bmag[:, :, idx_z].T)
        Q_xy.set_UVC(Bx[::skip, ::skip, idx_z], By[::skip, ::skip, idx_z])
        axs[0].set_title(f'XY slice at Z={z[idx_z]:.2f} cm, Time={timestamps[t_idx]:.2f}')

        heatmap_xz.set_data(Bmag[:, idx_y, :].T)
        Q_xz.set_UVC(Bx[::skip, idx_y, ::skip], Bz[::skip, idx_y, ::skip])
        axs[1].set_title(f'XZ slice at Y={y[idx_y]:.2f} cm, Time={timestamps[t_idx]:.2f}')

        heatmap_yz.set_data(Bmag[idx_x, :, :].T)
        Q_yz.set_UVC(By[idx_x, ::skip, ::skip], Bz[idx_x, ::skip, ::skip])
        axs[2].set_title(f'YZ slice at X={x[idx_x]:.2f} cm, Time={timestamps[t_idx]:.2f}')

        # 3D graph
        ax3d.clear()
        surf = ax3d.plot_surface(
            grid_x[:, :, idx_z], grid_y[:, :, idx_z], Bmag[:, :, idx_z],
            cmap='inferno', vmin=0, vmax=SENSITIVITY, edgecolor='none', alpha=0.9
        )
        ax3d.scatter(sensor_positions[:, 0], sensor_positions[:, 1], np.linalg.norm(sensor_B, axis=1), color='red', s=50, label='Sensors')
        ax3d.set_xlabel('X (cm)')
        ax3d.set_ylabel('Y (cm)')
        ax3d.set_zlabel('|B|')
        ax3d.set_title(f'3D |B| Surface at Z={z[idx_z]:.2f} cm, Time={timestamps[t_idx]:.2f}')
        ax3d.set_zlim(0, SENSITIVITY)
        ax3d.legend(loc='upper left')

        fig.canvas.draw_idle()

    # Call update function when sliders change
    time_slider.on_changed(update)
    slice_z_slider.on_changed(update)
    slice_y_slider.on_changed(update)
    slice_x_slider.on_changed(update)

    plt.show()
