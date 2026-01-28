import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
N_FRAMES = 500
WINDOW_SIZE = 64
TRUE_TURBULENCE = 0.8
MEAN_SHEAR = 0.2
PARTICLE_SIGMA = 1.5
np.random.seed(42)

# --- GENERATION FUNCTIONS (Same as before) ---
def generate_particles(size, n_particles=200, sigma=1.5):
    im = np.zeros((size, size))
    xy = np.random.rand(n_particles, 2) * size
    for x, y in xy:
        xi, yi = int(x), int(y)
        if 0 <= xi < size and 0 <= yi < size:
            im[yi, xi] += 1
    return nd.gaussian_filter(im, sigma)

def correlate_fft(im1, im2):
    f1 = np.fft.fft2(im1)
    f2 = np.fft.fft2(im2)
    return np.fft.fftshift(np.fft.ifft2(f1 * np.conj(f2)).real)

# --- ACCUMULATORS ---
sum_corr = np.zeros((WINDOW_SIZE, WINDOW_SIZE))
sum_im1 = np.zeros((WINDOW_SIZE, WINDOW_SIZE))
sum_im2 = np.zeros((WINDOW_SIZE, WINDOW_SIZE))

print("Simulating flow...")

# --- SIMULATION ---
for i in range(N_FRAMES):
    # 1. Generate Frame A
    im1 = generate_particles(WINDOW_SIZE)
    
    # 2. Generate Frame B (Sheared + Turbulenct)
    im2 = np.zeros_like(im1)
    y_coords = np.arange(WINDOW_SIZE) - WINDOW_SIZE/2
    
    # Physics: Mean Shear + Turbulence
    u_mean = y_coords * MEAN_SHEAR
    u_prime = np.random.normal(0, TRUE_TURBULENCE)
    
    # Deform image (Shift Back to remove mean)
    # This simulates the "Predictor" step having done its job
    total_shift = u_mean + u_prime
    # If we shift by -u_mean, we are left with just u_prime (turbulence)
    # BUT, the "Mean Image" will capture the structure of the particles because
    # the deformation aligned them.
    
    # Let's simulate the "Deformed Image" directly
    # We shift im1 by JUST u_prime (simulating that u_mean was perfectly removed)
    for y in range(WINDOW_SIZE):
        im2[y, :] = nd.shift(im1[y, :], u_prime, mode='wrap')

    # Accumulate
    sum_corr += correlate_fft(im1, im2)
    sum_im1 += im1
    sum_im2 += im2

# --- CALCULATE TERMS ---
# Term 1: The Total Average Correlation <AB>
Total_Corr = sum_corr / N_FRAMES

# Term 2: The Background <A><B>
Mean_A = sum_im1 / N_FRAMES
Mean_B = sum_im2 / N_FRAMES
Background_Term = correlate_fft(Mean_A, Mean_B)

# Term 3: The Result
Result = Total_Corr - Background_Term

# --- VISUALIZATION ---
fig = plt.figure(figsize=(18, 6))

# Helper to plot surface
X, Y = np.meshgrid(np.arange(WINDOW_SIZE), np.arange(WINDOW_SIZE))
def plot_surf(ax, data, title, color):
    # Normalize for viewability
    Z = data 
    surf = ax.plot_surface(X, Y, Z, cmap=color, edgecolor='none', alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_zlabel('Correlation Energy')
    # Focus view
    mid = WINDOW_SIZE//2
    ax.set_xlim(mid-15, mid+15)
    ax.set_ylim(mid-15, mid+15)

# PLOT 1: TOTAL CORRELATION <AB>
ax1 = fig.add_subplot(131, projection='3d')
plot_surf(ax1, Total_Corr, "1. Total Correlation <AB>\n(Signal + Background)", 'viridis')

# PLOT 2: BACKGROUND TERM <A><B>
ax2 = fig.add_subplot(132, projection='3d')
plot_surf(ax2, Background_Term, "2. Background Term <A><B>\n(What we subtract)", 'plasma')

# PLOT 3: RESULT
ax3 = fig.add_subplot(133, projection='3d')
plot_surf(ax3, Result, "3. Final Result\n(Diff)", 'coolwarm')

plt.tight_layout()
plt.show()