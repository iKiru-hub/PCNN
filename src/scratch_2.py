import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_steps = 5000  # More steps for smoother trajectory
step_size = 0.01  # Smaller step size
box_size = 1.0

# Generate a smoothed random walk with periodic boundary conditions
def generate_smooth_trajectory(n_steps, step_size, box_size):
    # Initialize raw trajectory
    x_raw = np.zeros(n_steps)
    y_raw = np.zeros(n_steps)
    
    # Start at center position
    x_raw[0] = 0.5
    y_raw[0] = 0.5
    
    # Generate correlated random walk for smoothness
    theta = np.random.uniform(0, 2*np.pi)  # Initial direction
    persistence = 0.95  # Higher directional persistence for smoother path
    
    for i in range(1, n_steps):
        # Update direction with some persistence
        theta = theta + (1-persistence) * np.random.uniform(-np.pi, np.pi)
        
        # Step in current direction
        dx = step_size * np.cos(theta)
        dy = step_size * np.sin(theta)
        
        # Update position
        x_raw[i] = x_raw[i-1] + dx
        y_raw[i] = y_raw[i-1] + dy
        
        # Apply periodic boundary conditions
        x_raw[i] = x_raw[i] % box_size
        y_raw[i] = y_raw[i] % box_size
    
    return x_raw, y_raw

# Generate the raw trajectory
x_raw, y_raw = generate_smooth_trajectory(n_steps, step_size, box_size)

# Function to identify segments where trajectory doesn't cross boundaries
def segment_trajectory(x, y, box_size, threshold=0.5*step_size):
    segments_x = []
    segments_y = []
    current_segment_x = [x[0]]
    current_segment_y = [y[0]]
    
    for i in range(1, len(x)):
        # Check if we crossed a boundary
        dx = abs(x[i] - x[i-1])
        dy = abs(y[i] - y[i-1])
        
        if dx > threshold or dy > threshold:
            # Boundary crossed, end current segment
            segments_x.append(np.array(current_segment_x))
            segments_y.append(np.array(current_segment_y))
            # Start new segment
            current_segment_x = [x[i]]
            current_segment_y = [y[i]]
        else:
            # Continue current segment
            current_segment_x.append(x[i])
            current_segment_y.append(y[i])
    
    # Add the last segment
    if current_segment_x:
        segments_x.append(np.array(current_segment_x))
        segments_y.append(np.array(current_segment_y))
    
    return segments_x, segments_y

# Segment the trajectory for 2D plot
segments_x, segments_y = segment_trajectory(x_raw, y_raw, box_size)

# Map 2D coordinates to torus
def map_to_torus(x, y, R, r):
    """Map 2D coordinates to a torus in 3D
    R: major radius of the torus
    r: minor radius of the torus
    """
    # Convert 2D coordinates to angles
    theta = 2 * np.pi * x / box_size  # Around the major circle
    phi = 2 * np.pi * y / box_size    # Around the minor circle
    
    # Parametric equations for a torus
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    
    return X, Y, Z

# Torus parameters
R = 1.0  # Major radius
r = 0.3  # Minor radius

# Map the complete trajectory to the torus (no segmentation needed)
X_torus, Y_torus, Z_torus = map_to_torus(x_raw, y_raw, R, r)

# Create the plot
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Periodic Boundary Conditions: 2D Square to Torus Mapping', y=0.98, fontweight='bold')

# 2D plot (square with periodic boundaries)
ax1 = fig.add_subplot(121)

# Plot each segment with the same color
for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
    ax1.plot(seg_x, seg_y, '-', color='royalblue', alpha=0.8, linewidth=1.2)

# Draw the box boundaries
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
ax1.axhline(y=box_size, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
ax1.axvline(x=box_size, color='k', linestyle='-', alpha=0.5, linewidth=0.8)

# Mark start and end points
ax1.scatter(x_raw[0], y_raw[0], color='green', s=30, label='Start', zorder=5, edgecolor='black', linewidth=0.5)
ax1.scatter(x_raw[-1], y_raw[-1], color='red', s=30, label='End', zorder=5, edgecolor='black', linewidth=0.5)

ax1.set_xlim(0, box_size)
ax1.set_ylim(0, box_size)
ax1.set_aspect('equal')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_title('2D Trajectory with Periodic Boundaries', pad=10)
ax1.legend(loc='upper right', frameon=True, framealpha=0.9)

# 3D plot (torus)
ax2 = fig.add_subplot(122, projection='3d')

# Plot the full trajectory on the torus as a single continuous line
ax2.plot(X_torus, Y_torus, Z_torus, '-', color='royalblue', alpha=0.8, linewidth=1.2)

# Create a semi-transparent torus surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
u_grid, v_grid = np.meshgrid(u, v)

x_torus = (R + r * np.cos(v_grid)) * np.cos(u_grid)
y_torus = (R + r * np.cos(v_grid)) * np.sin(u_grid)
z_torus = r * np.sin(v_grid)

# Create a more transparent surface
torus_surface = ax2.plot_surface(x_torus, y_torus, z_torus, color='gray', alpha=0.15, 
                                 edgecolor='lightgray', linewidth=0.1)

# Mark start and end points on torus
X_start, Y_start, Z_start = map_to_torus(x_raw[0], y_raw[0], R, r)
X_end, Y_end, Z_end = map_to_torus(x_raw[-1], y_raw[-1], R, r)
ax2.scatter(X_start, Y_start, Z_start, color='green', s=30, label='Start', zorder=5, edgecolor='black', linewidth=0.5)
ax2.scatter(X_end, Y_end, Z_end, color='red', s=30, label='End', zorder=5, edgecolor='black', linewidth=0.5)

ax2.set_title('Trajectory Mapped onto a Torus', pad=10)
ax2.legend(loc='upper right')

# Remove axis numbers but keep axis labels 
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_xlabel(r'$X$', labelpad=5)
ax2.set_ylabel(r'$Y$', labelpad=5)
ax2.set_zlabel(r'$Z$', labelpad=5)

# Set the 3D plot aspect ratio and adjust view
ax2.set_box_aspect([1,1,1])
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Uncomment to save the figure
# plt.savefig('torus_mapping.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('torus_mapping.png', dpi=300, bbox_inches='tight')

plt.show()
