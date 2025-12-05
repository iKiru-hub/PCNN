import numpy as np
import matplotlib.pyplot as plt
import argparse, json
from tqdm import tqdm
import os

import pandas as pd
import scikit_posthocs as ph
from scipy import stats

import game.envs as games

import simulations as sim
import utils


""" settings """

logger = utils.setup_logger(__name__, level=5, is_debugging=False)


GAME_SCALE = games.SCREEN_WIDTH
ENVIRONMENT = "Arena.0001"
MIN_WEIGHT = 0.01


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
    "max_duration": 15_000,
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

# parameters = {
#       "gain": 102.4,
#       "offset": 1.02,
#       "threshold": 0.2,
#       "rep_threshold": 0.955,
#       "rec_threshold": 33,
#       "tau_trace": 10,
#       "remap_tag_frequency": 1,
#       "num_neighbors": 4,
#       "min_rep_threshold": 0.99,
#       "lr_da": 0.9,
#       "lr_pred": 0.95,
#       "threshold_da": 0.10,
#       "tau_v_da": 1.0,
#       "lr_bnd": 0.9,
#       "threshold_bnd": 0.1,
#       "tau_v_bnd": 1.0,
#       "tau_ssry": 437.0,
#       "threshold_ssry": 1.986,
#       "threshold_circuit": 0.9,
#       "rwd_weight": -0.11,
#       "rwd_sigma": 96.8,
#       "rwd_threshold": 0.,
#       "col_weight": -0.53,
#       "col_sigma": 16.1,
#       "col_threshold": 0.37,
#       "rwd_field_mod": 4.6,
#       "col_field_mod": 4.4,
#       "action_delay": 120.0,
#       "edge_route_interval": 50,
#       "forced_duration": 19,
#       "min_weight_value": 0.1,
#     "modulation_options": [True]*4
# }

parameters = utils.load_parameters(idx=90)

def get_sig_stars(p_value: float) -> str:

    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'

def run_simulations(num_reps: int) -> tuple:

    """ run num_reps repetitions and return the results about the pc gains """


    data = {'rwd': [], 'rwd_dx': [], 'bnd': [], 'bnd_dx': [], 'control': [], 'control_dx': []}

    pbar = tqdm(range(num_reps))
    for _ in pbar:
        pbar.set_description(f"rep={_+1}|{num_reps}")
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
        centers = np.array(brain.get_space_centers())
        centers_init = np.array(brain.get_space_centers_original())

        # da indexes
        rwd = brain.get_da_weights()
        rwd_idxs = np.where(rwd>MIN_WEIGHT)[0]
        rwd_dx = np.linalg.norm(centers[rwd_idxs] - centers_init[rwd_idxs])

        # !if no rwd pc are present, skip the rep
        if len(rwd_idxs) == 0:
            logger.warning("no rwd pc -> skipping rep")
            continue

        # bnd indexes
        bnd = brain.get_bnd_weights()
        bnd_idxs = np.where(bnd>MIN_WEIGHT)[0]
        bnd_centers = centers[bnd_idxs]
        bnd_dx = np.linalg.norm(centers[bnd_idxs] - centers_init[bnd_idxs])

        # other indexes
        ctrl_idxs = [i for i in np.arange(len(brain.space)).tolist() if i not in bnd_idxs and i not in rwd_idxs and i < len(brain.space)]

        # cells
        gains = np.array(brain.get_gain())
        rwd_gains = gains[rwd_idxs]
        bnd_gains = gains[bnd_idxs]
        ctrl_gains = gains[ctrl_idxs]

        data['rwd'] += [rwd_gains.tolist()]
        data['rwd_dx'] += [rwd_dx.tolist()]
        data['bnd'] += [bnd_gains.tolist()]
        data['bnd_dx'] += [bnd_dx.tolist()]
        data['control'] += [ctrl_gains.tolist()]

    # ---

    data_gains = {'rwd': [np.mean(d) for d in data['rwd']],
                  'bnd': [np.mean(d) for d in data['bnd']],
                  'control': [np.mean(d) for d in data['control']]}

    data_dx = {'rwd_dx': [np.mean(d) for d in data['rwd_dx']],
               'bnd_dx': [np.mean(d) for d in data['bnd_dx']],
               'control': [np.float64(0) for _ in data['bnd_dx']]}

    print(data_gains)
    print(data_dx)

    return data_gains, data_dx


def get_statistics(data: dict, num_reps: int):

    """ perform Dunnett's Test

    data: dict
        {'group1': [mean1, mean2, .. meanK],
        'group2': [mean1, mean2, .. meanK],
        'control': [mean1, mean2, .. meanK]}

    num_reps: int
        numeber K of simulations
    """

    groups = list(data.keys())
    non_control_groups = [k for k in groups if k != 'control']
    num_reps = len(data['control'])

    # Create lists of values and labels for the DataFrame
    values = []
    labels = []

    # Ensure control is the first item in the concatenation for logical ordering,
    # but the key order in the dict is arbitrary for Dunnett's.
    for key in ['control'] + non_control_groups:
        values.extend(data[key])
        labels.extend([key] * num_reps)

    df_agg = pd.DataFrame({'value': values, 'group': labels})

    # 2. Data Centering (Mitigate RuntimeWarning from near-zero variance)
    center_value = df_agg['value'].mean()
    df_agg['value_centered'] = df_agg['value'] - center_value

    # 3. Perform Dunnett's Test
    dunnett_results = ph.posthoc_dunnett(
        a=df_agg,
        val_col='value_centered', # Use centered data for stability
        group_col='group',
        control='control'
    )

    # Extract FWER-controlled p-values for non-control groups
    p_values = dunnett_results[
        dunnett_results.index != 'control'
    ][['control']].squeeze() # squeeze converts the 1-column DF to a Series

    # 4. Calculate Descriptive Statistics (using ORIGINAL data)
    group_stats = df_agg.groupby('group')['value'].agg(
        mean='mean',
        std='std'
    )
    # Calculate SEM: std / sqrt(N)
    group_stats['SEM'] = group_stats['std'] / np.sqrt(num_reps)

    # 5. Compile Final Dictionary
    final_dict = {}

    for group_name in non_control_groups:
        # Get stats
        mean = group_stats.loc[group_name, 'mean']
        sem = group_stats.loc[group_name, 'SEM']

        # Get p-value and significance
        p = p_values[group_name]
        sig_stars = get_sig_stars(p)

        final_dict[group_name] = {
            'mean_sem': [mean, sem],
            'p_value': p,
            'significance': sig_stars
        }

    # Include control group statistics for reference
    final_dict['control'] = {
        'mean_sem': [group_stats.loc['control', 'mean'], group_stats.loc['control', 'SEM']],
        'p_value': None,
        'significance': 'control'
    }

    return final_dict

def save(file: dict):

    name = utils.DATA_PATH + "/eval_gain_data"

    num = len([f for f in os.listdir(utils.DATA_PATH) if "eval_gain_data" in f])

    with open(f"{name}_{num}.json", 'w') as f:
        json.dump(file, f)

    logger("[file saved]")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # --

    logger("starting simulation..")
    data_gains, data_dx = run_simulations(num_reps=args.reps)
    logger("[simulation done]")

    # adjust for skipped reps
    num_reps = len(data_gains['rwd'])
    if num_reps == 0:
        logger.warning("no valid rep")
        import sys
        sys.exit()

    logger("statistics for 'gain'")
    pvalues_gains = get_statistics(data=data_gains, num_reps=num_reps)

    logger("statistics for 'dx'")
    pvalues_dx = get_statistics(data=data_dx, num_reps=num_reps)

    # -- save
    if save:
        file = {"gain": {"data": data_gains,
                         "results": pvalues_gains},
                "dx": {"data": data_dx,
                       "results": pvalues_dx},
                "notes": "using genome 91"}

        save(file=file)

    logger("[done]")

