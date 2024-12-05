from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

# Modified Shepp-Logan Phantom erzeugen
N = 256  # Bildgröße N x N
phantom = shepp_logan_phantom()

# Phantom auf gewünschte Größe anpassen
phantom_resized = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)
print(phantom.shape)
print(phantom_resized.shape)

# Phantom anzeigen
plt.imshow(phantom_resized, cmap='gray')
plt.title(f'Modified Shepp-Logan Phantom ({N}x{N})')
plt.colorbar()
plt.show()

# Optional: Matrix speichern
np.save('/mnt/data/shepp_logan_phantom.npy', phantom_resized)