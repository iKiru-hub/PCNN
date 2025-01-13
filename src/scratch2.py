import libs.pclib as pclib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


SPEED = 0.1

gcn = pclib.GridHexNetwork([
            pclib.GridHexLayer(sigma=0.03, speed=0.1),
            pclib.GridHexLayer(sigma=0.05, speed=0.09),
            pclib.GridHexLayer(sigma=0.04, speed=0.08),
            pclib.GridHexLayer(sigma=0.03, speed=0.07),
            pclib.GridHexLayer(sigma=0.04, speed=0.06)])

pcnn_ = pclib.PCNNgridhex(N=200,
                          Nj=len(gcn),
                          gain=7.,
                          offset=1.5,
                          clip_min=0.01,
                          threshold=0.1,
                          rep_threshold=0.8,
                          rec_threshold=0.1,
                          num_neighbors=8, trace_tau=0.1,
                          xfilter=gcn, name="2D")

da = pclib.BaseModulation(name="DA", size=3, min_v=0.1,
                          offset=0.01, gain=200.0)

action_space = pclib.ActionSampling2D("AS", SPEED)
M = pclib.Brain(da, pcnn_, action_space)



def run_brain(duration, brain):


    a = []
    x, y = 0.2, 0.2
    x0, y0 = 0.2, 0.2
    traj = [[x, y]]
    acc = np.zeros((len(gcn), 100))

    size = 20
    s = np.array([SPEED, SPEED])
    collision = 0.

    for t in tqdm(range(duration)):

        # move
        x += s[0]
        y += s[1]

        # forward
        s = brain([x - x0, y - y0], collision)

        x0 = x
        y0 = y

        # change direction
        # if t % 100 == 0:
        #     s = np.random.uniform(-1, 1, 2)
        #     s = 0.1 * s / np.abs(s).sum()

        # hit wall
        if x <= 0 or x >= size:
            s[0] *= -1
            x += s[0]
            collision = 1.
            print("hit wall x")
        elif y <= 0 or y >= size:
            s[1] *= -1
            y += s[1]
            collision = 1.
            print("hit wall y")
        else:
            collision = 0.

        # record
        traj += [[x, y]]


    # --- plot
    traj = np.array(traj)
    fig, axs = plt.subplots(1, 2, figsize=(5, 10))
    ax = axs[0]
    ax.plot(traj[:, 0], traj[:, 1], 'r')

    ax = axs[1]
    ax.imshow(da.get_weights().reshape(1, -1), aspect='auto', cmap='hot')

    plt.show()


if __name__ == "__main__":
    run_brain(1000, M)

