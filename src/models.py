import json




model_1 = {

    "gain_fine": 11.,
    "offset_fine": 1.1,
    "threshold_fine": 0.35,
    "rep_threshold_fine": 0.88,
    "rec_threshold_fine": 26.,
    "min_rep_threshold": 0.95,
    "remap_tag_frequency": 1,
    "tau_trace_fine": 10.0,

    "gain_coarse": 11.,
    "offset_coarse": 1.1,
    "threshold_coarse": 0.35,
    "rep_threshold_coarse": 0.89,
    "rec_threshold_coarse": 60.,
    "tau_trace_coarse": 20.0,

    "lr_da": 0.4,
    "lr_pred": 0.4,
    "threshold_da": 0.08,
    "tau_v_da": 1.0,

    "lr_bnd": 0.4,
    "threshold_bnd": 0.04,
    "tau_v_bnd": 1.0,

    "tau_ssry": 100.,
    "threshold_ssry": 0.995,

    "threshold_circuit": 0.7,
    "remapping_flag": 0,

    "rwd_weight": 0.0,
    "rwd_sigma": 40.0,
    "col_weight": 0.0,
    "col_sigma": 2.0,

    "action_delay": 180.,
    "edge_route_interval": 47,

    "forced_duration": 32,
    "fine_tuning_min_duration": 29
}
