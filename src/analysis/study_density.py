import numpy as np
import matplotlib.pyplot as plt

import sys, os
from tqdm import tqdm
import time

sys.path.append(os.path.join(os.getcwd().split("PCNN")[0], "PCNN/src/"))
from game.constants import *
import core.build.pclib as pclib
import libs.pclib2 as pclib2
import utils
from game.envs import *
import game.objects as objects
import simulations as sim
logger = utils.setup_logger('An.DNS', level=3)

""" FUNCTIONS """

def calc_density_values(graph: np.ndarray, rw_position: np.ndarray,
                        rw_radius: float, area_radius: float,
                        env_length: float) -> tuple:

    """
    calculation of the average density around the reward location in
    an area of radius #area_radius (but not within the reward radius)
    and outside.

    Parameters
    ----------
    graph: np.ndarray
        node centers, as a (N, 2) array
    rw_position: np.ndarray
        reward position
    rw_radius: float
    area_radius: float
    env_length: float

    Returns
    -------
    tuple: (float, float)
        density near, density far
    """

    rx, ry = rw_position
    num_near = 0
    num_far = 0

    # calculate counts
    for c in graph:
        dist = np.sqrt((c[0]-rx)**2 + (c[1]-ry)**2)
        if dist < area_radius:
            num_near += 1
        else:
            num_far += 1

    # calculate areas
    area_near = 2 * np.pi * (area_radius - rw_radius)
    area_far = env_length ** 2

    # densities
    return num_near / area_near, num_far / area_far


def compute_place_field(positions, activations, grid_size=20,
                        method='max', threshold=0.1):
    """
    Create a 2D place field map by binning the space and aggregating neuron activation.

    Args:
        positions: (T, 2) positions (x, y) over time
        activations: (T,) activation of the neuron over time
        grid_size: int, spatial resolution
        method: 'max' or 'mean'

    Returns:
        field: (grid_size, grid_size) 2D place field
    """
    field = np.zeros((grid_size, grid_size))
    counts = np.zeros_like(field)

    # Normalize positions to [0, 1]
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    norm_positions = (positions - pos_min) / (pos_max - pos_min + 1e-8)

    # Compute bin indices
    bins = np.floor(norm_positions * grid_size).astype(int)
    bins = np.clip(bins, 0, grid_size - 1)

    for t in range(len(activations)):
        x, y = bins[t]
        if method == 'max':
            field[y, x] = max(field[y, x], activations[t])
            field[y, x] = field[y, x] if field[y, x] > threshold else 0.
        elif method == 'mean':
            field[y, x] += activations[t]
            counts[y, x] += 1

    if method == 'mean':
        field = np.divide(field, counts, where=counts > 0)

    return field


""" SETTINGS """

reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.08 * GAME_SCALE,
    "rw_sigma": 0.7,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 5,
    "silent_duration": 5_000,
    "fetching_duration": 4,
    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "tau": 200,# * GAME_SCALE,
    "move_threshold": 2005,# * GAME_SCALE,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move both",
    "rendering": False,
    "rendering_pcnn": False,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 10_000,
    "room_thickness": 20,
    "t_teleport": 1_500,
    "limit_position_len": -1,
    "start_position_idx": 0,
    "seed": None,
    "pause": -1,
    "verbose": True
}

global_parameters = {
    "local_scale": 0.02,
    "N": 32**2,
    "use_sprites": False,
    "speed": 1.,
    "min_weight_value": 0.5
}

parameters = {
     'gain': 33.0,
     'offset': 1.0,
     'threshold': 0.4,
     'rep_threshold': 0.86,
     'rec_threshold': 63,
     'tau_trace': 140,

     'remap_tag_frequency': 3,
     'num_neighbors': 20,
     'min_rep_threshold': 0.87,

     'lr_da': 0.99,
     'lr_pred': 0.1,
     'threshold_da': 0.04,
     'tau_v_da': 2.0,

     'lr_bnd': 0.6,
     'threshold_bnd': 0.3,
     'tau_v_bnd': 4.0,

     'tau_ssry': 437.0,
     'threshold_ssry': 1.986, # <-----------------
     'threshold_circuit': 0.9,

     'rwd_weight': 2.96,
     'rwd_sigma': 33.6,
     'col_weight': 0.06,
     'col_sigma': 20.6,
     'rwd_field_mod': 0.0,
     'col_field_mod': -0.6,

     'action_delay': 120.0,
     'edge_route_interval': 5000,
     'forced_duration': 1,
     'min_weight_value': 0.2
}

""" making the brain """

brain = pclib2.Brain(
                local_scale=global_parameters["local_scale"],
                N=global_parameters["N"],
                rec_threshold=parameters["rec_threshold"],
                speed=global_parameters["speed"],
                min_rep_threshold=parameters["min_rep_threshold"],
                num_neighbors=parameters["num_neighbors"],
                gain=parameters["gain"],
                offset=parameters["offset"],
                threshold=parameters["threshold"],
                rep_threshold=parameters["rep_threshold"],
                tau_trace=parameters["tau_trace"],
                remap_tag_frequency=parameters["remap_tag_frequency"],
                lr_da=parameters["lr_da"],
                lr_pred=parameters["lr_pred"],
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                rwd_weight=parameters["rwd_weight"],
                rwd_sigma=parameters["rwd_sigma"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                rwd_field_mod=parameters["rwd_field_mod"],
                col_field_mod=parameters["col_field_mod"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                min_weight_value=parameters["min_weight_value"])


""" TRAIN WITHOUT REWARD """

def _train(brain: object, is_reward: bool):

    """ make game environment """

    verbose = False
    verbose_min = True
    game_settings["rendering"] = False
    reward_settings["silent_duration"] = 0 if is_reward else 40_000
    game_settings["max_duration"] = 40_000 if is_reward else 20_000
    game_settings["rw_event"] = "none"

    room_name = "Square.v0"

    if verbose and verbose_min:
        logger(f"room_name={room_name}")

    room = make_room(name=room_name,
                     thickness=game_settings["room_thickness"],
                     bounds=[0, 1, 0, 1])
    room_bounds = [room.bounds[0]+10, room.bounds[2]-10,
                   room.bounds[1]+10, room.bounds[3]-10]

    # ===| objects |===

    possible_positions = room.get_room_positions()

    rw_position_idx = np.random.randint(0, len(possible_positions))
    rw_position = [300, 300]
    agent_possible_positions = possible_positions.copy()
    agent_position = possible_positions[game_settings['start_position_idx']]

    rw_tau = reward_settings["tau"] if "tau" in reward_settings else 400

    reward_obj = objects.RewardObj(
                # position=reward_settings["rw_position"],
                position=rw_position,
                possible_positions=possible_positions,
                radius=reward_settings["rw_radius"],
                sigma=reward_settings["rw_sigma"],
                fetching=reward_settings["rw_fetching"],
                value=reward_settings["rw_value"],
                bounds=room_bounds,
                delay=reward_settings["delay"],
                silent_duration=reward_settings["silent_duration"],
                fetching_duration=reward_settings["fetching_duration"],
                use_sprites=global_parameters["use_sprites"],
                move_threshold=20000,
                transparent=not is_reward)

    body = objects.AgentBody(
                position=agent_position,
                speed=global_parameters["speed"],
                possible_positions=agent_possible_positions,
                bounds=game_settings["agent_bounds"],
                use_sprites=global_parameters["use_sprites"],
                limit_position_len=game_settings["limit_position_len"],
                room=room,
                color=(10, 10, 10))

    logger(reward_obj)


    # --- env
    env = Environment(room=room,
                      agent=body,
                      reward_obj=reward_obj,
                      duration=game_settings["max_duration"],
                      rw_event=game_settings["rw_event"],
                      verbose=False,
                      visualize=game_settings["rendering"])
    logger(env)

    if verbose_min:
        logger("[@simulations.py]")
    record = sim.run_game(env=env,
             brain=brain,
             renderer=None,
             plot_interval=game_settings["plot_interval"],
             pause=-1,
             record_flag=True,
             verbose=verbose,
             verbose_min=verbose_min)

    logger(f"rw_count={env.rw_count}")


def _make_pc_fields(brain: object):

    gc = brain.get_gc_network()
    wff = brain.get_wff()
    logger(gc)

    # random walk
    speed = 0.9
    size = 500.
    lims = (-250, 250)

    points = [[0., 0.]]

    s = np.array([speed, speed])
    x, y = points[0]
    old_point = points[0]

    tot = 200_000

    # record
    activity = np.zeros((len(gc), tot))
    plotting = False

    for t in tqdm(range(tot)):
        
        x += s[0]
        y += s[1]

        # hit wall
        if x <= lims[0] or x >= lims[1]:
            s[0] *= -1
            x += s[0]
        elif y <= lims[0] or y >= lims[1]:
            s[1] *= -1
            y += s[1]

        points += [[x, y]]
        if t % 1000 == 0:
            s = np.random.uniform(-1, 1, 2)
            s = speed * s / np.abs(s).sum()
            s = np.around(s, 3)

        activity[:, t] = gc([points[-1][0]-old_point[0],
                             points[-1][1]-old_point[1]])
        
        old_point = points[-1]

        if t % 100 == 0 and plotting:
            clear_output(wait=True)
            plt.figure(figsize=(4, 4))
            #plt.subplot(121)
            plt.plot(*np.array(points).T, 'k-', lw=0.3, alpha=0.3)
            
            plt.title(f"{t=}")
            plt.scatter(*points[-1], s=100, c='r')

            plt.xlim((-4, size+4))
            plt.ylim((-4, size+4))
            plt.pause(0.001)

    logger("[trajectory done]")

    pc = []
    trajectory = np.array(points)
    activity_pc = wff @ activity
    for i in tqdm(range(len(activity_pc))):
        pc += [compute_place_field(trajectory[1:],
                                   activity_pc[i],
                                   grid_size=500,
                                   threshold=1.)]
    pc = np.stack(pc)
    cc = brain.get_space_centers()
    cc = cc[np.where(cc[:, 0]>-2000)[0]]

    return pc, cc


if __name__ == "__main__":

    brain = pclib2.Brain(
                local_scale=global_parameters["local_scale"],
                N=global_parameters["N"],
                rec_threshold=parameters["rec_threshold"],
                speed=global_parameters["speed"],
                min_rep_threshold=parameters["min_rep_threshold"],
                num_neighbors=parameters["num_neighbors"],
                gain=parameters["gain"],
                offset=parameters["offset"],
                threshold=parameters["threshold"],
                rep_threshold=parameters["rep_threshold"],
                tau_trace=parameters["tau_trace"],
                remap_tag_frequency=parameters["remap_tag_frequency"],
                lr_da=parameters["lr_da"],
                lr_pred=parameters["lr_pred"],
                threshold_da=parameters["threshold_da"],
                tau_v_da=parameters["tau_v_da"],
                lr_bnd=parameters["lr_bnd"],
                threshold_bnd=parameters["threshold_bnd"],
                tau_v_bnd=parameters["tau_v_bnd"],
                tau_ssry=parameters["tau_ssry"],
                threshold_ssry=parameters["threshold_ssry"],
                threshold_circuit=parameters["threshold_circuit"],
                rwd_weight=parameters["rwd_weight"],
                rwd_sigma=parameters["rwd_sigma"],
                col_weight=parameters["col_weight"],
                col_sigma=parameters["col_sigma"],
                rwd_field_mod=parameters["rwd_field_mod"],
                col_field_mod=parameters["col_field_mod"],
                action_delay=parameters["action_delay"],
                edge_route_interval=parameters["edge_route_interval"],
                forced_duration=parameters["forced_duration"],
                min_weight_value=parameters["min_weight_value"])

    # without reward
    logger("[-] without reward...")
    _train(brain=brain, is_reward=False)
    pc_before, cc_before = _make_pc_fields(brain=brain)
    logger(f"#centers={len(cc_before)}")

    # with reward
    logger("[+] with reward...")
    _train(brain=brain, is_reward=True)
    pc_after, cc_after = _make_pc_fields(brain=brain)
    logger(f"#centers={len(cc_after)}")


    """ field sum """

    _vmax = 4000

    plt.figure(figsize=(13, 4))
    plt.subplot(131)
    plt.imshow(pc_before.sum(axis=0), vmax=_vmax)
    plt.title("before")
    plt.subplot(132)
    plt.imshow(pc_after.sum(axis=0), vmax=_vmax)
    plt.title("after")
    plt.subplot(133)
    plt.imshow(pc_after.sum(axis=0)-pc_before.sum(axis=0), cmap="RdBu_r")
    plt.title("difference")
    plt.colorbar()

    plt.figure(figsize=(7, 7))
    plt.scatter(*np.array(cc_before).T, color='blue')
    plt.scatter(*np.array(cc_after).T, color='red')

    plt.show()
