import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Define helper functions equivalent to Rectangle, projrec, and shlo (to be implemented)
def rectangle(r, cx, cy, angle, resolution):
    """
    Create a binary image of a rectangle (placeholder).
    """
    grid = np.linspace(-1, 1, 2 * resolution + 1)
    X, Y = np.meshgrid(grid, grid)
    rect = ((abs(X - cx) <= r) & (abs(Y - cy) <= r)).astype(float)
    return rotate(rect, angle, reshape=False)

def projrec(r, cx, cy, angle, resolution, q, p):
    """
    Compute projections of the rectangle (placeholder).
    """
    theta = np.linspace(0, np.pi, p)
    projections = np.zeros((p, 2 * q + 1))
    # Add the logic for Radon transform and projections
    return projections

def shlo(q):
    """
    Generate filter for backprojection (placeholder).
    """
    return np.hamming(2 * q + 1)

# Parameters
q = 60  # Line integrals per direction
p = 180  # Number of directions
q_recim = 150  # Sampling resolution
cx = np.sqrt(2) / 3  # Center of the square
r = 2 / 3 - cx  # Half-length of the square sides
phi = 0  # Rotation angle of the square

# Generate original image and projection data
orig = rectangle(r, cx, cx, phi, q_recim)
data = projrec(r, cx, cx, phi, q_recim, q, p)

# Apply mask
mask = np.zeros_like(data)
mask[:, 20:101] = 1
data *= mask

# Apply filter
fil = shlo(q)
dim = 2 * q + 1
w = np.array([fil[i:dim + i] for i in range(dim)])

v = np.dot(data, w.T)
dim_recim = 2 * q_recim + 1
recim = np.zeros((dim_recim, dim_recim))

# Reconstruction via backprojection
theta = np.linspace(0, np.pi, p)
co, si = np.cos(theta), np.sin(theta)
X = np.linspace(-1, 1, dim_recim)
skal = np.pi / (p * q)

for j in range(p):
    Xcos = np.outer(co, X)
    Ysin = np.outer(si, X)
    
    smat = q * (Ysin + Xcos.T)
    kmat = np.floor(smat).astype(int) + q + 1
    tmat = smat - np.floor(smat)
    
    for ix in range(dim_recim):
        yshift = int(np.sqrt((ix - 1) * (dim_recim - ix)))
        for iy in range(q_recim + 1 - yshift, q_recim + 1 + yshift):
            zz = (1 - tmat[:, iy]) * v[kmat[:, iy] - 1] + tmat[:, iy] * v[kmat[:, iy]]
            recim[iy, dim_recim - ix - 1] = skal * np.sum(zz[:j])

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

X = np.linspace(-1, 1, dim_recim)
ax[0].imshow(orig, extent=(-1, 1, -1, 1), cmap='turbo', vmin=-0.5, vmax=3)
ax[0].set_title('Original ROI')
ax[0].axis('off')

ax[1].imshow(recim.T, extent=(-1, 1, -1, 1), cmap='turbo', vmin=-0.5, vmax=3)
ax[1].set_title('Reconstructed ROI')
ax[1].axis('off')

plt.show()