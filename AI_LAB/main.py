import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.manifold import LocallyLinearEmbedding, Isomap

# Generate Swiss Roll dataset
n_samples = 1500
X, color = datasets.make_swiss_roll(n_samples, noise=0.1)

# Apply LLE for dimensionality reduction (2D)
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)

# Apply Isomap for dimensionality reduction (2D)
isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)

# Plotting the original Swiss Roll and the results of LLE and Isomap
fig = plt.figure(figsize=(18, 6))

# Original 3D Swiss Roll
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral, s=10)
ax1.set_title('Original Swiss Roll (3D)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# LLE result
ax2 = fig.add_subplot(132)
ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
ax2.set_title('LLE Result (2D)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Isomap result
ax3 = fig.add_subplot(133)
ax3.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap=plt.cm.Spectral, s=10)
ax3.set_title('Isomap Result (2D)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

plt.tight_layout()
plt.show()
