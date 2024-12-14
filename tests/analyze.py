import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src"))
import libs.pclib as pclib
from utils_core import setup_logger

logger = setup_logger("Analysis")
N = 30
Nj = 10**2

""" TRAJECTORY """

duration = 100
dx = 0.1
trajectory = np.stack((np.arange(duration),
                       np.zeros(duration))) * 0.1
tot = 7
fig, axs = plt.subplots(tot, 1, figsize=(10, tot*3))

for j, rpt in enumerate(np.linspace(0/tot, 0.9, tot)):
    xfilter = pclib.RandLayer(Nj)
    pcnn2D = pclib.PCNNrand(N=N, Nj=Nj,
                        gain=4., offset=1.4,
                        clip_min=0.09,
                        threshold=0.5,
                        rep_threshold=0.4,
                        rec_threshold=0.1,
                        num_neighbors=8, trace_tau=0.1,
                        xfilter=xfilter, name="2D")
    activity = np.zeros((N, duration))
    for i, x in enumerate(trajectory.T):
        activity[:, i] = pcnn2D(x)
        pcnn2D.update()
    axs[j].set_title(f"rep_threshold={rpt:.4f}")
    axs[j].imshow(activity, aspect="auto", cmap="Greys")
    axs[j].set_xticks(())

plt.show()
