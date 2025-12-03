import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pclib
import pytest
import numpy as np
import matplotlib.pyplot as plt



def test_brain2():

    da = pclib.BaseModulation(name="DA", size=3, min_v=0.1,
                              offset=0.01, gain=200.0)
    bnd = pclib.BaseModulation(name="BND", size=3, min_v=0.1,
                               offset=0.01, gain=200.0)
    cir = pclib.Circuits(da, bnd)

    layers = [
        pclib.GridHexLayer(0.03, 0.1),
        pclib.GridHexLayer(0.05, 0.9),
        pclib.GridHexLayer(0.04, 0.08),
        pclib.GridHexLayer(0.03, 0.07),
        pclib.GridHexLayer(0.04, 0.05)
    ]
    xfilter = pclib.GridHexNetwork(layers)
    space = pclib.PCNNgridhex(200, len(xfilter),
                              7, 1.5, 0.01, 0.1, 0.8, 0.1,
                              8, 0.1, xfilter, "2D")

    sampler = pclib.ActionSampling2D("default", 1)
    wrec = space.get_wrec()
    trgp = pclib.TargetProgram(0., wrec, da, 20, 0.8)
    brain = pclib.BrainHex(cir, space, sampler, trgp)

    pos = [0., 0.]
    posh = []
    v = [0., 0.]
    for _ in range(1900):
        v = brain(v, 0., 0., pos)
        #print("\n", pos)
        pos[0] += v[0]
        pos[1] += v[1]
        #print(pos, pos[0], pos[1], v[0], v[1], "sum: ", pos[0]+v[0])
        posh += [[pos[0], pos[1]]]

    assert len(posh) == 1900, f"Position history is not correct {len(posh)}"




def activity_over_grid(model):

    # make a list of points over a square 0,1 x 0,1
    n = 100
    x = np.linspace(0., 1., n)
    y = np.linspace(0., 1., n)
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # compute the activity of the model over the grid
    activity = []
    for i in range(len(grid)):
        activity.append(model.fwd_ext(grid[i]).max())

    # plot the activity
    print(f"Len={len(pcnn)}")
    activity = np.array(activity).reshape(n, n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(activity.T, cmap='hot', interpolation='nearest',
               vmin=0., vmax=1.)
    ax1.set_title(f"Length {len(pcnn)}," + \
        f" max={activity.max():.3f}min={activity.min():.3f}")

    ax2.imshow(pcnn.get_wff(), cmap='hot', interpolation='nearest')
    ax2.set_title("Weights")

    plt.show()


def main1():

    pclib.set_debug(False)

    n = 10
    Ni = 10
    sigma = 0.09
    bounds = np.array([0., 1., 0., 1.])
    xfilter = pclib.PCLayer(n, sigma, bounds)

    # definition
    pcnn = pclib.PCNN(N=Ni, Nj=n**2, gain=3., offset=1.5,
                      clip_min=0.09, threshold=0.5,
                      rep_threshold=0.5, rec_threshold=0.01,
                      num_neighbors=8, trace_tau=0.1,
                      xfilter=xfilter, name="2D")

    print(f"Len={len(pcnn)}")
    print(f"Size={pcnn.get_size()}")

    # learn position 1
    x = np.array([0.4, 0.5])
    _ = pcnn(x)

    # learn position 2
    x = np.array([0.6, 0.5])
    _ = pcnn(x)

    activity_over_grid(pcnn)




if __name__ == "__main__":


    test_brain2()

