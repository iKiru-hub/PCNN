import numpy as np
import matplotlib.pyplot as plt
import pcnn_core as pcr
import pclib



n = 7
N = n**2
K = 10

# connectivity matrix with some clusters in it
M = np.zeros((N, N))
M[:K, :K] = np.random.binomial(1, 0.5, (K, K))
M[K:K+K, K:K+K] = np.random.binomial(1, 0.5, (K, K))
M *= 1 - np.eye(N)

# make place cells
pc = pcr.PClayer(n=n,
                 sigma=0.01)


fig, ax = plt.subplots(1, 1)

# set of initial positions over the grid
X, Y = np.meshgrid(np.linspace(0, 1, 10),
                     np.linspace(0, 1, 10))
X = X.flatten()
Y = Y.flatten()
pos = np.array([X, Y]).T


for i in range(len(pos)):

    x = pos[i]

    traj = [x.tolist()]
    for t in range(100):

        # forward
        a = pc(x=x)

        # recurrent
        a = M @ a

        # new position
        x = pcr.calc_position_from_centers(a=a,
                                           centers=pc.centers)

        traj.append(x.tolist())

    traj = np.array(traj)

    # ax.clear()
    ax.plot(traj[:, 0], traj[:, 1], 'ko-',
            alpha=0.1)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'iter={i}')
    plt.pause(0.001)

plt.show()


