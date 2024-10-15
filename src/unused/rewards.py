import numpy as np
import matplotlib.pyplot as plt



# make a xy mesh as an array of points
granularity = 100
X = np.linspace(0, 1, granularity)
Y = np.linspace(0, 1, granularity)
X, Y = np.meshgrid(X, Y)
#Y = Y[::-1]  # flip Y axis

points = np.array([X.flatten(), Y.flatten()]).T

# make a function
func = lambda x, y: x**2 + y**2
Z = func(points[:, 0], points[:, 1])

# plot
plt.imshow(Z.reshape((granularity, granularity)), cmap='viridis')
plt.colorbar()

# plt.xticks(np.linspace(0, granularity, 5), np.linspace(0, 1, 5))
# plt.yticks(np.linspace(0, granularity, 5), np.linspace(0, 1, 5))


# define a policy over the mesh
positions = np.random.uniform(0, 1, (granularity, 2))

policy = np.empty((granularity, 2))

# determine the policy as a function of the function values
for i, p in enumerate(positions):

    # determine the value of the function at the closest point
    idx = np.argmin(np.linalg.norm(points - p, axis=1))
    z = Z[idx]

    policy[i] = z if z > 0.5 else 0


# plot the policy
plt.scatter(positions[:, 0]*100, positions[:, 1]*100, c=policy[:, 0], cmap='hot')
plt.colorbar()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()



