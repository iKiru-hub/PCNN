import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm

import pandas as pd
import scikit_posthocs as ph
import game.envs as games

import simulations as sim
import utils

logger = utils.setup_logger(__name__, level=5, is_debugging=False)

GAME_SCALE = games.SCREEN_WIDTH

ENVIRONMENT = "Arena.0000"

reward_settings = {
    "rw_fetching": "deterministic",
    "rw_value": "discrete",
    "rw_position": np.array([0.5, 0.3]) * GAME_SCALE,
    "rw_radius": 0.05 * GAME_SCALE,
    "rw_sigma": 0.8,# * GAME_SCALE,
    "rw_bounds": np.array([0.23, 0.77,
                           0.23, 0.77]) * GAME_SCALE,
    "delay": 2,
    "silent_duration": 1_000,
    "fetching_duration": 2,
    "transparent": False,
    "beta": 40.,
    "alpha": 0.06,# * GAME_SCALE,
    "move_threshold": 4,# * GAME_SCALE,
    "rw_position_idx": 0,
}

game_settings = {
    "plot_interval": 5,
    "rw_event": "move both",
    "rendering": False,
    "rendering_pcnn": False,
    "agent_bounds": np.array([0.23, 0.77,
                              0.23, 0.77]) * GAME_SCALE,
    "max_duration": 5_000,
    "room_thickness": 20,
    "t_teleport": 2_500,
    "limit_position_len": -1,
    "start_position_idx": 0,
    "seed": None,
    "pause": -1,
    "verbose": False
}

global_parameters = {
    "local_scale": 0.015,
    "N": 32**2,
    "use_sprites": bool(1),
    "speed": 1.0,
    "min_weight_value": 0.5
}

parameters = {
      "gain": 102.4,
      "offset": 1.02,
      "threshold": 0.2,
      "rep_threshold": 0.955,
      "rec_threshold": 33,
      "tau_trace": 10,
      "remap_tag_frequency": 1,
      "num_neighbors": 4,
      "min_rep_threshold": 0.99,
      "lr_da": 0.9,
      "lr_pred": 0.95,
      "threshold_da": 0.10,
      "tau_v_da": 1.0,
      "lr_bnd": 0.9,
      "threshold_bnd": 0.1,
      "tau_v_bnd": 1.0,
      "tau_ssry": 437.0,
      "threshold_ssry": 1.986,
      "threshold_circuit": 0.9,
      "rwd_weight": -0.11,
      "rwd_sigma": 96.8,
      "rwd_threshold": 0.,
      "col_weight": -0.53,
      "col_sigma": 16.1,
      "col_threshold": 0.37,
      "rwd_field_mod": 4.6,
      "col_field_mod": 4.4,
      "action_delay": 120.0,
      "edge_route_interval": 50,
      "forced_duration": 19,
      "min_weight_value": 0.1,
    "modulation_options": [True]*4
}


def run_simulations(num_reps: int) -> dict:

    """ run num_reps repetitions and return the results about the pc gains """


    data = {'rwd': [], 'bnd': [], 'control': []}

    for _ in range(num_reps):
        out, info = sim.run_model(parameters=parameters,
                                  global_parameters=global_parameters,
                                  reward_settings=reward_settings,
                                  game_settings=game_settings,
                                  room_name=ENVIRONMENT,
                                  record_flag=True,
                                  limit_position_len=-1,
                                  verbose=False, pause=game_settings["pause"])

        brain = info['brain']
        env = info['env']
        reward_obj = info['reward_obj']
        record = info['record']

        # da indexes
        da = brain.get_da_weights()
        daidx = np.where(da>0.05)[0]

        # bnd indexes
        bnd = brain.get_bnd_weights()
        bndidx = np.where(bnd>0.05)[0]

        # other indexes
        noidx = [i for i in np.arange(len(brain.space)).tolist() if i not in bndidx and i not in daidx and i < len(brain.space)]

        # cells
        gg = np.array(brain.get_gain())
        dgg = gg[daidx]
        bgg = gg[bndidx]
        ngg = gg[noidx]

        data['rwd'] += [dgg.tolist()]
        data['bnd'] += [bgg.tolist()]
        data['control'] += [ngg.tolist()]

    data = {'rwd': [np.mean(d) for d in data['rwd']],
            'bnd': [np.mean(d) for d in data['bnd']],
            'control': [np.mean(d) for d in data['control']]}

    return data


def get_statistics(data: dict):

    """ 1. Perform Dunnett's Test with Corrected Syntax """

    num_reps = len(data['control'])

    df_agg = pd.DataFrame({
        'value': np.concatenate([data['control'], data['rwd'], data['bnd']]),
        'group': ['control'] * num_reps + ['rwd'] * num_reps + ['bnd'] * num_reps
    })

    # REMOVE the p_adjust keyword argument entirely.
    dunnett_results = ph.posthoc_dunnett(
        a=df_agg, 
        val_col='value', 
        group_col='group', 
        control='control' 
    )

    # Filter rows that are not 'Control' and select the 'Control' column
    comparison_results = dunnett_results[dunnett_results.index != 'control'][['control']]

    comparison_results.columns = ['p-value vs. control (Dunnett-adjusted)']
    logger(f"{comparison_results=}")

    return comparison_results.columns


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()


    logger("starting simulation..")
    data = run_simulations(num_reps=args.reps)
    logger("[simulation done]")

    pvalues = get_statistics(data=data)
    logger("[done]")

