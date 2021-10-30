import matplotlib.pyplot as plt
import numpy as np


n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = np.sin(-x*y)

v1 = (0.0598, 0.0279, 0.1131)
v2 = (0.0570, 0.0274, 0.1115)

# x1 = np.array([0.0598, 0.0570, 0.0590])
# y1 = np.array([0.0279, 0.0274, 0.0267])
# z1 = np.array([0.1131, 0.1115, 0.1095])

# x2 = np.array([0.0570, 0.0573, 0.0530])
# y2 = np.array([0.0274, 0.0282, 0.0279])
# z2 = np.array([0.1115, 0.1139, 0.1132])

box1 = [(0.004502, 3.084070), (-0.476621, 0.278756), (-0.381270, 1.983040)]
box2 = [(0.013106, 3.084070), (-0.476621, 0.279363), (-0.381964, 1.962760)]

xs = [box1[0][i] for j in range(4) for i in range(2)]
ys = [box1[1][j] for j in range(2) for i in range(4)]
zs = [box1[2][i] for i in range(2) for j in range(4)]

ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(xs, ys, zs, linewidth=0.2, antialiased=True, color='r')

xs = [box2[0][i] for j in range(4) for i in range(2)]
ys = [box2[1][j] for j in range(2) for i in range(4)]
zs = [box2[2][i] for i in range(2) for j in range(4)]

ax.plot_trisurf(xs, ys, zs, linewidth=0.2, antialiased=True, color='b')

plt.show()

