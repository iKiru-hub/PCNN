# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from pprint import pprint
import time
from copy import deepcopy

import os
import main
import src.models as mm
from src.models import logger
from src.minimal_model import minPCNN, fullPCNN, Policy
import src.visualizations as vis
import src.utils as utils
import inputools.Trajectory as it
from tools.utils import clf, tqdm_enumerate, save_image, AnimationMaker


# %%
""" PARAMETERS """

# parameters
duration = 10
seed = 4321
online = True
lr = 0.2
tau = 800
ach_threshold = 0.3
save = False
show = True
animate = False
verbose = True

N = 200
Nj = 25**2

params = {
    "N": N,
    "Nj": Nj,
    "tau": 10.0,
    "alpha": 0.2,
    "beta": 20.0,
    "lr": lr,
    "threshold": 0.02,
    "ach_threshold": ach_threshold,
    "da_threshold": 0.5,
    "tau_ach": 200.,  # 2.
    "eq_ach": 1.,
    "tau_da": tau,  # 2.
    "eq_da": 0.,
    "epsilon": 0.1,
}

# %%
""" 1st TRAINING """

# make trajectory
trajectory, whole_track, inputs, whole_track_layer, layer = main.make_trajectory(
                        plot=False,
                        Nj=Nj,
                        duration=duration,
                        is2d=True,
                        sigma=0.004)
logger(">>> trajectory 1")

# %%

model = minPCNN(**params)
if verbose:
    logger(f"%{model}")
    logger("training 1...")

# train
for t, x in tqdm_enumerate(inputs, disable=not verbose):
    model.step(x=x.reshape(-1, 1))
    # if t % 100:
    #     model.clear_connections()

# --- PLOT
# --- with old pc estimate
if verbose:
    logger("plotting 1...")

if not online:
    fig, ax = plt.subplots()
    fig.suptitle(f"$\eta=${lr} - $\\tau=${tau}")
else:
    ax = None

# model.clear_connections(epsilon=0.05)
centers1, connections1, nodes1, edges1 = main.make_centers(model=model,
                 trajectory=trajectory,
                 whole_track=whole_track,
                 whole_track_layer=whole_track_layer,
                 plot=ax is not None, show=False, color='grey',
                 verbose=verbose,
                 ax=ax)
model.add_many_centers(centers=centers1)
if verbose:
    logger(f"{len(centers1)} place cells")



# %%
old_model = deepcopy(model)
logger(">>> `old_model`")

# --- train 2 -----------------------------------------------
# make trajectory
trajectory, whole_track, inputs, whole_track_layer, layer = main.make_trajectory(
                        plot=False,
                        Nj=Nj,
                        duration=duration,
                        is2d=True,
                        sigma=0.004)
logger(">>> trajectory 2")

# policy
startime = 200
trg = np.array([0.5, 0.5])
policy = Policy(eq_da=1.,
                trg=trg,
                threshold=0.15,
                startime=startime)

online = False
if online:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    da = []
    ach = []
    dwd = []
    umax = []
    ax1.scatter(trg[0], trg[1], c='g', s=20, alpha=0.8, marker='x')

if animate:
    animation_maker = AnimationMaker(fps=7, use_logger=True,
                                     path=ANIM_PATH)

if verbose:
    logger(f"%{policy}")
    logger("training 2...")

#
umax = []
ft = []
umask0 = model._umask.copy()
tper = 1000

model.unfreeze()
for t, x in tqdm_enumerate(inputs, disable=not verbose):
    eq_da = policy(pos=trajectory[t, :], t=t)
    model(x=x, eq_da=eq_da)

    # - calc nodes
    idxs = np.where(model._umask - umask0)[0]
    umask0 = model._umask.copy()
    if len(idxs) > 0:
        ft += [[trajectory[t, 0], trajectory[t, 1]]]
        model.add_center(center=[trajectory[t, 0], trajectory[t, 1]],
                         idx=idxs[0])

    # online plot
    if online:
        da += [model._DA]
        ach += [model._ACh]
        dwd += [model.dwda.max()]
        umax += [model.u.max()]
        if len(da) > 1000:
            del da[0]
            del ach[0]
            del dwd[0]
            del umax[0]

        if t % tper == 0:
            # model.clear_connections(epsilon=0.05)
            centers2, connections2, nodes2, edges2 = make_centers(model=model,
                            trajectory=trajectory,
                            whole_track=whole_track,
                            whole_track_layer=whole_track_layer,
                            plot=False, show=False, color='black',
                            verbose=False,
                            alpha=0.95,
                            ax=ax)

            model.unfreeze()

            online_plot(t=t, tper=tper, model=model,
                        trajectory=trajectory,
                        policy_eq=eq_da,
                        policy_time=startime, with_policy=True,
                        ax1=ax1, ax2=ax2, fig=fig,
                        umax=umax, da=da, ach=ach,
                        dwd=dwd,
                        centers=centers2,
                        connections=connections2,
                        plot_trajectory=True,
                        show=True)

            if animate:
                animation_maker.add_frame(fig)

# --- PLOT 2
# --- with old pc estimate
if verbose:
    logger("plotting 2...")

if online:
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle(f"$\eta=${lr} - $\\tau=${args.tau} - " + \
                 f"$\\theta_{{ach}}$={ach_threshold} - " + \
                 f"$\\theta_{{da}}$={params['da_threshold']}")

if animate:
    animation_maker.make_animation(name=f"roaming_{time.strftime('%H%M%S')}")
    logger(f"animation saved at {ANIM_PATH}")
    animation_maker.play_animation(return_Image=False)

# %%
plt.imshow(old_model._Wff, aspect='auto')
plt.show()


# %%
plt.subplot(211)
plt.imshow(model._Wff, aspect='auto', vmin=0., vmax=0.3)
plt.colorbar()
plt.title("new model")
plt.subplot(212)
plt.imshow(old_model._Wff, aspect='auto',
           vmin=0., vmax=0.3)
plt.colorbar()
plt.title("old model")
plt.show()

# %%
centers2, connections2, nodes2, edges2 = main.make_centers(model=model,
                trajectory=trajectory,
                whole_track=whole_track,
                whole_track_layer=whole_track_layer,
                plot=False, show=False, color='black',
                verbose=verbose,
                alpha=0.95,
                ax=ax)

if verbose:
    logger(f"{len(centers2)} place cells")

# nodes = nodes1 * nodes2

# print(connections1.shape, connections2.shape)
# print(nodes1.shape, nodes2.shape, nodes.shape)

# centers 2 for each actual node, final shape
# is matches the total number of neurons
# centers = []
# idxn = 0
# idxe = 0
# connections = []
# for i, (n2, n) in enumerate(zip(nodes2, nodes)):
#     if n2 > 0:
#         if n > 0:
#             centers += [centers2[idxn].tolist()]
#             connections += [connections1[idxe].tolist()]
#             idxe += 1
#         idxn += 1

# centers = np.array(centers)
# connections = np.array(connections)

connections0 = connections2.copy()

# logger.debug(f"{connections0.shape} {connections1.shape} {connections2.shape}")

for i in range(min((len(centers1), len(centers2)))):
    if nodes1[i] > 0:
        for j in range(min((len(connections1[i]), len(connections2[i])))):
            connections0[i, j] = connections1[i, j]

# print("final graph: ", centers.shape, connections.shape)

# plot
ax.scatter(trg[0], trg[1], c='g', s=40, alpha=0.8,
           marker='x',
           label="target")
vis.plot_graph(centers=centers1,
               connectivity=connections1,
               ax=ax, alpha=1., color="black",
               marker='o',
               label="old centers",
               plot_connections=True,
               plot_centers=True)
vis.plot_graph(centers=centers2,
               connectivity=connections2,
               ax=ax, alpha=0.8, color="green",
               marker='v',
               label="new centers",
               plot_connections=True,
               plot_centers=True)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal', 'box')

ax.set_title(f"{len(centers1)} PCs")
plt.legend()

if save:
    plt.savefig(f"media/online_{lr}_{args.tau}.png")
    plt.close()
elif show:
    plt.show()



