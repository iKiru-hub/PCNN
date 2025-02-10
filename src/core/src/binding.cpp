#include "pcnn.hpp"
#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>

#define PCNN_REF PCNNsqv2
#define CIRCUIT_SIZE 5

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

    // function `set_debug`
    m.def("set_debug", &set_debug,
          py::arg("flag"));

    /* PCNN MODEL */
    // (InputFilter) Grid Layer
    py::class_<GridLayerSq>(m, "GridLayerSq")
        .def(py::init<float, float, std::array<float, 4>>(),
             py::arg("sigma"),
             py::arg("speed"),
             py::arg("bounds") = std::array<float, 4>({0.0f, 1.0f, 0.0f, 1.0f}))
        .def("__call__", &GridLayerSq::call,
             py::arg("v"))
        .def("__str__", &GridLayerSq::str)
        .def("__repr__", &GridLayerSq::repr)
        .def("__len__", &GridLayerSq::len)
        .def("simulate", &GridLayerSq::simulate,
             py::arg("v"),
             py::arg("sim_gc_positions"))
        .def("simulate_one_step", &GridLayerSq::simulate_one_step,
             py::arg("v"))
        .def("get_centers", &GridLayerSq::get_centers)
        .def("get_activation", &GridLayerSq::get_activation)
        .def("get_positions", &GridLayerSq::get_positions)
        .def("reset", &GridLayerSq::reset,
             py::arg("v"));

    // Grid Layer Network
    py::class_<GridNetworkSq>(m, "GridNetworkSq")
        .def(py::init<std::vector<GridLayerSq>>(),
             py::arg("layers"))
        .def("__call__", &GridNetworkSq::call,
                py::arg("x"))
        .def("__str__", &GridNetworkSq::str)
        .def("__repr__", &GridNetworkSq::repr)
        .def("__len__", &GridNetworkSq::len)
        .def("simulate", &GridNetworkSq::simulate,
             py::arg("v"),
             py::arg("sim_gc_positions"))
        .def("simulate_one_step", &GridNetworkSq::simulate_one_step,
             py::arg("v"))
        .def("get_activation", &GridNetworkSq::get_activation)
        .def("get_centers", &GridNetworkSq::get_centers)
        .def("get_num_layers", &GridNetworkSq::get_num_layers)
        .def("get_positions", &GridNetworkSq::get_positions);

    // Grid layer with pre-defined hexagon layer
    py::class_<GridHexLayer>(m, "GridHexLayer")
        .def(py::init<float, float, float, float>(),
             py::arg("sigma"),
             py::arg("speed"),
             py::arg("offset_dx") = 0.0f,
             py::arg("offset_dy") = 0.0f)
        .def("__str__", &GridHexLayer::str)
        .def("__repr__", &GridHexLayer::repr)
        .def("__len__", &GridHexLayer::len)
        .def("__call__", &GridHexLayer::call,
             py::arg("v"))
        .def("get_positions", &GridHexLayer::get_positions)
        .def("get_centers", &GridHexLayer::get_centers)
        .def("reset", &GridHexLayer::reset,
             py::arg("v"));

    // Grid Hexagonal Layer Network
    py::class_<GridHexNetwork>(m, "GridHexNetwork")
        .def(py::init<std::vector<GridHexLayer>>(),
             py::arg("layers"))
        .def("__call__", &GridHexNetwork::call,
                py::arg("v"))
        .def("__str__", &GridHexNetwork::str)
        .def("__repr__", &GridHexNetwork::repr)
        .def("__len__", &GridHexNetwork::len)
        .def("fwd_position", &GridHexNetwork::fwd_position,
                py::arg("v"))
        .def("get_activation", &GridHexNetwork::get_activation)
        .def("get_centers", &GridHexNetwork::get_centers)
        .def("get_num_layers", &GridHexNetwork::get_num_layers)
        .def("get_positions", &GridHexNetwork::get_positions)
        .def("reset", &GridHexNetwork::reset,
             py::arg("v"));

    // PCNN network model [Grid]
    py::class_<PCNNgridhex>(m, "PCNNgridhex")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, GridHexNetwork, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain"),
             py::arg("offset"),
             py::arg("clip_min"),
             py::arg("threshold"),
             py::arg("rep_threshold"),
             py::arg("rec_threshold"),
             py::arg("num_neighbors"),
             py::arg("trace_tau"),
             py::arg("xfilter"),
             py::arg("name"))
        .def("__call__", &PCNNgridhex::call,
             py::arg("v"),
             py::arg("frozen") = false,
             py::arg("traced") = false)
        .def("__str__", &PCNNgridhex::str)
        .def("__len__", &PCNNgridhex::len)
        .def("__repr__", &PCNNgridhex::repr)
        .def("update", &PCNNgridhex::update,
             py::arg("x") = -1.0,
             py::arg("y") = -1.0)
        /* .def("ach_modulation", &PCNNgridhex::ach_modulation, */
        /*      py::arg("ach")) */
        .def("get_activation", &PCNNgridhex::get_activation)
        .def("get_activation_gcn",
             &PCNNgridhex::get_activation_gcn)
        .def("get_size", &PCNNgridhex::get_size)
        .def("get_trace", &PCNNgridhex::get_trace)
        .def("get_wff", &PCNNgridhex::get_wff)
        .def("get_wrec", &PCNNgridhex::get_wrec)
        .def("get_connectivity",\
             &PCNNgridhex::get_connectivity)
        .def("get_centers", &PCNNgridhex::get_centers,
             py::arg("nonzero") = false)
        .def("get_delta_update",\
             &PCNNgridhex::get_delta_update)
        .def("get_positions_gcn", \
             &PCNNgridhex::get_positions_gcn)
        .def("fwd_ext", &PCNNgridhex::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNNgridhex::fwd_int,
             py::arg("a"))
        .def("reset_gcn", &PCNNgridhex::reset_gcn,
             py::arg("v"));

    // PCNN network model [GridSq]
    py::class_<PCNNbase>(m, "PCNNbase")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, int, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain") = 10.0f,
             py::arg("offset") = 0.5f,
             py::arg("clip_min") = 0.001f,
             py::arg("threshold") = 0.5f,
             py::arg("rep_threshold") = 0.5f,
             py::arg("rec_threshold") = 0.5f,
             py::arg("num_neighbors") = 8,
             py::arg("length"),
             py::arg("name") = "PCNNbase")
        .def("__call__", &PCNNbase::call,
             py::arg("v"),
             py::arg("frozen") = false,
             py::arg("traced") = true)
        .def("__str__", &PCNNbase::str)
        .def("__len__", &PCNNbase::len)
        .def("__repr__", &PCNNbase::repr)
        .def("update", &PCNNbase::update,
             py::arg("x") = -1.0,
             py::arg("y") = -1.0)
        .def("set_xfilter", &PCNNbase::set_xfilter)
        .def("get_activation", &PCNNbase::get_activation)
        .def("get_activation_gcn",
             &PCNNbase::get_activation_gcn)
        .def("ach_modulation", &PCNNbase::ach_modulation,
             py::arg("ach"))
        .def("get_size", &PCNNbase::get_size)
        .def("get_trace", &PCNNbase::get_trace)
        .def("get_wff", &PCNNbase::get_wff)
        .def("get_wrec", &PCNNbase::get_wrec)
        .def("get_connectivity", &PCNNbase::get_connectivity)
        .def("get_centers", &PCNNbase::get_centers,
             py::arg("nonzero") = false)
        .def("get_basis", &PCNNbase::get_basis)
        .def("get_delta_update", &PCNNbase::get_delta_update)
        .def("fwd_ext", &PCNNbase::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNNbase::fwd_int,
             py::arg("a"))
        .def("reset_gcn", &PCNNbase::reset_gcn,
             py::arg("v"));

    // PCNN network model [GridSq]
    py::class_<PCNNsqv2>(m, "PCNNsqv2")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, GridNetworkSq&, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain"),
             py::arg("offset"),
             py::arg("clip_min"),
             py::arg("threshold"),
             py::arg("rep_threshold"),
             py::arg("rec_threshold"),
             py::arg("num_neighbors"),
             py::arg("xfilter"),
             py::arg("name"))
        .def("__call__", &PCNNsqv2::call,
             py::arg("v"))
        .def("__str__", &PCNNsqv2::str)
        .def("__len__", &PCNNsqv2::len)
        .def("__repr__", &PCNNsqv2::repr)
        .def("update", &PCNNsqv2::update)
        .def("get_activation", &PCNNsqv2::get_activation)
        .def("get_activation_gcn",
             &PCNNsqv2::get_activation_gcn)
        /* .def("ach_modulation", &PCNNsqv2::ach_modulation, */
        /*      py::arg("ach")) */
        .def("get_size", &PCNNsqv2::get_size)
        .def("get_wff", &PCNNsqv2::get_wff)
        .def("get_wrec", &PCNNsqv2::get_wrec)
        .def("make_edges", &PCNNsqv2::make_edges)
        .def("get_connectivity", &PCNNsqv2::get_connectivity)
        .def("get_centers", &PCNNsqv2::get_centers,
             py::arg("nonzero") = false)
        .def("get_node_degrees", &PCNNsqv2::get_node_degrees)
        .def("get_delta_update", &PCNNsqv2::get_delta_update)
        /* .def("get_nodes_max_angle", &PCNNsqv2::get_nodes_max_angle) */
        .def("get_position", &PCNNsqv2::get_position)
        .def("simulate", &PCNNsqv2::simulate,
             py::arg("v"),
             py::arg("sim_gc_positions"))
        .def("simulate_one_step", &PCNNsqv2::simulate_one_step,
             py::arg("v"))
        .def("calculate_angle_gap", &PCNNsqv2::calculate_angle_gap,
             py::arg("idx"),
             py::arg("centers"),
             py::arg("connectivity"))
        /* .def("fwd_int", &PCNNsqv2::fwd_int, */
        /*      py::arg("a")) */
        .def("modulate_gain", &PCNNsqv2::modulate_gain,
             py::arg("mod"))
        .def("modulate_rep", &PCNNsqv2::modulate_rep,
             py::arg("mod"))
        .def("modulate_threshold", &PCNNsqv2::modulate_threshold,
             py::arg("mod"))
        .def("get_gain", &PCNNsqv2::get_gain)
        .def("get_rep", &PCNNsqv2::get_rep)
        .def("get_threshold", &PCNNsqv2::get_threshold)
        .def("reset_gcn", &PCNNsqv2::reset_gcn,
             py::arg("v"));

    /* MODULATION MODULES */

    // LeakyVariable 1D
    py::class_<LeakyVariable1D>(m, "LeakyVariable1D")
        .def(py::init<std::string, float, float,
             float>(),
             py::arg("name"),
             py::arg("eq"),
             py::arg("tau"),
             py::arg("min_v") = 0.0)
        .def("__call__", &LeakyVariable1D::call,
             py::arg("x") = 0.0,
             py::arg("simulate") = false)
        .def("__str__", &LeakyVariable1D::str)
        .def("__len__", &LeakyVariable1D::len)
        .def("__repr__", &LeakyVariable1D::repr)
        .def("get_v", &LeakyVariable1D::get_v)
        .def("get_name", &LeakyVariable1D::get_name)
        .def("set_eq", &LeakyVariable1D::set_eq,
             py::arg("eq"))
        .def("reset", &LeakyVariable1D::reset);

    /* py::class_<PopulationMaxProgram>(m, "PopulationMaxProgram") */
    /*     .def(py::init<>()) */
    /*     .def("__str__", &PopulationMaxProgram::str) */
    /*     .def("__repr__", &PopulationMaxProgram::repr) */
    /*     .def("__call__", &PopulationMaxProgram::call, */
    /*          py::arg("u")) */
    /*     .def("len", &PopulationMaxProgram::len) */
    /*     .def("get_value", &PopulationMaxProgram::get_value); */

    /* py::class_<MemoryRepresentation>(m, "MemoryRepresentation") */
    /*     .def(py::init<int, float, float>(), */
    /*          py::arg("size"), */
    /*          py::arg("decay"), */
    /*          py::arg("threshold")) */
    /*     .def("__str__", &MemoryRepresentation::str) */
    /*     .def("__repr__", &MemoryRepresentation::repr) */
    /*     .def("__call__", &MemoryRepresentation::call, */
    /*          py::arg("representation"), */
    /*          py::arg("simulate") = false) */
    /*     .def_readwrite("tape", &MemoryRepresentation::tape); */

    /* py::class_<MemoryAction>(m, "MemoryAction") */
    /*     .def(py::init<float>(), */
    /*          py::arg("decay")) */
    /*     .def("__str__", &MemoryAction::str) */
    /*     .def("__repr__", &MemoryAction::repr) */
    /*     .def("__call__", &MemoryAction::call, */
    /*          py::arg("idx"), */
    /*          py::arg("simulate") = false) */
    /*     .def_readwrite("tape", &MemoryAction::tape); */

    /* MODULATION */
    py::class_<BaseModulation>(m, "BaseModulation")
        .def(py::init<std::string, int, float, float,
             float, float, float, float, float>(),
             py::arg("name"),
             py::arg("size"),
             py::arg("lr") = 0.1f,
             py::arg("threshold") = 0.0f,
             py::arg("max_w") = 1.0f,
             py::arg("tau_v") = 5.0f,
             py::arg("eq_v") = 0.0f,
             py::arg("min_v") = 0.01f,
             py::arg("mask_threshold") = 0.01f)
        .def("__str__", &BaseModulation::str)
        .def("__repr__", &BaseModulation::repr)
        .def("__len__", &BaseModulation::len)
        .def("__call__", &BaseModulation::call,
             py::arg("u"),
             py::arg("x") = 0.0f,
             py::arg("simulate") = false)
        .def("get_output", &BaseModulation::get_output)
        .def("get_leaky_v", &BaseModulation::get_leaky_v)
        .def("get_weights", &BaseModulation::get_weights);

    py::class_<Circuits>(m, "Circuits")
        .def(py::init<BaseModulation&, BaseModulation&,
             float>(),
             /* MemoryRepresentation&, MemoryAction&>(), */
             py::arg("da"),
             py::arg("bnd"),
             py::arg("threshold"))
             /* py::arg("memrepr"), */
             /* py::arg("memact")) */
        .def("__str__", &Circuits::str)
        .def("__repr__", &Circuits::repr)
        .def("__len__", &Circuits::len)
        .def("__call__", &Circuits::call,
             py::arg("u"),
             py::arg("collision"),
             py::arg("reward"),
             py::arg("simulate") = false)
        .def("get_output", &Circuits::get_output)
        /* .def("get_memory_representation", &Circuits::get_memory_representation) */
        /* .def("get_memory_action", &Circuits::get_memory_action) */
        .def("get_leaky_v", &Circuits::get_leaky_v);

    /* MODULES */

    /* py::class_<TargetProgram>(m, "TargetProgram") */
    /*     /1* .def(py::init<Eigen::VectorXf&, PCNN_REF&, *1/ */
    /*     /1*      float>(), *1/ */
    /*     .def(py::init<PCNN_REF&, float>(), */
    /*          /1* py::arg("wrec"), *1/ */
    /*          /1* py::arg("centers"), *1/ */
    /*          /1* py::arg("da_weights"), *1/ */
    /*          py::arg("space"), */
    /*          py::arg("speed")) */
    /*     .def("__len__", &TargetProgram::len) */
    /*     .def("__str__", &TargetProgram::str) */
    /*     .def("__repr__", &TargetProgram::repr) */
    /*     .def("is_active", &TargetProgram::is_active) */
    /*     .def("update", &TargetProgram::update, */
    /*          py::arg("curr_representation"), */
    /*          py::arg("space_weights"), */
    /*          py::arg("tmp_trg_idx") = true) */
    /*          /1* py::arg("use_episodic_memory") = true) *1/ */
    /*          /1* py::arg("trigger") = true) *1/ */
    /*     .def("step_plan", &TargetProgram::step_plan) */
    /*     /1* .def("make_shortest_path", &TargetProgram::make_shortest_path, *1/ */
    /*     /1*      py::arg("wrec"), *1/ */
    /*     /1*      py::arg("start_idx"), *1/ */
    /*          /1* py::arg("end_idx")) *1/ */
    /*     /1* .def("set_da_weights", &TargetProgram::set_da_weights, *1/ */
    /*     /1*      py::arg("da_weights")) *1/ */
    /*     .def("set_wrec", &TargetProgram::set_wrec, */
    /*          py::arg("wrec")) */
    /*     /1* .def("get_trg_idx", *1/ */
    /*     /1*      &TargetProgram::get_trg_idx) *1/ */
    /*     .def("get_plan", &TargetProgram::get_plan); */

    // 2 layer network
    py::class_<ExplorationModule>(m, "ExplorationModule")
        .def(py::init<float, Circuits&, PCNN_REF&, float, int>(),
             /* std::array<float, CIRCUIT_SIZE>, float, int>(), */
             py::arg("speed"),
             py::arg("circuits"),
             py::arg("space"),
             /* py::arg("weights"), */
             py::arg("action_delay") = 1.0f,
             py::arg("edge_route_interval") = 100)
        .def("__call__", &ExplorationModule::call,
             py::arg("directive"),
             py::arg("rejection"))
        .def("__str__", &ExplorationModule::str)
        .def("__repr__", &ExplorationModule::repr)
        .def("confirm_edge_walk", &ExplorationModule::confirm_edge_walk)
        .def("reset_rejected_indexes", &ExplorationModule::reset_rejected_indexes);
        /* .def("get_actions", &ExplorationModule::get_actions); */
        /* .def("get_plan", &ExplorationModule::get_plan) */
        /* .def("get_all_plan_values", &ExplorationModule::get_all_plan_values) */
        /* .def("get_all_plan_scores", &ExplorationModule::get_all_plan_scores); */

    // Hexagon
    py::class_<Hexagon>(m, "Hexagon")
        .def(py::init<>())
        .def("__call__", &Hexagon::call,
             py::arg("x"),
             py::arg("y"))
        .def("__str__", &Hexagon::str)
        .def("get_centers", &Hexagon::get_centers);

    // Density Policy
    py::class_<DensityPolicy>(m, "DensityPolicy")
        .def(py::init<float, float,
             float, float>(),
             py::arg("rwd_weight"),
             py::arg("rwd_sigma"),
             py::arg("col_weight"),
             py::arg("col_sigma"))
        .def("__call__", &DensityPolicy::call,
             py::arg("space"),
             py::arg("da_weights"),
             py::arg("bnd_weights"),
             py::arg("displacement"),
             py::arg("da_value"),
             py::arg("bnd_value"),
             py::arg("reward"),
             py::arg("collision"))
        .def("__str__", &DensityPolicy::str)
        .def("__repr__", &DensityPolicy::repr)
        .def("get_rwd_mod", &DensityPolicy::get_rwd_mod)
        .def("get_col_mod", &DensityPolicy::get_col_mod);

    // Stationary Sensory
    py::class_<StationarySensory>(m, "StationarySensory")
        .def(py::init<int, float, float, float>(),
             py::arg("size"),
             py::arg("tau"),
             py::arg("threshold") = 0.2f,
             py::arg("min_cosine") = 0.5f)
        .def("__call__", &StationarySensory::call,
             py::arg("representation"))
        .def("__str__", &StationarySensory::str)
        .def("__repr__", &StationarySensory::repr)
        .def("get_v", &StationarySensory::get_v);

    // Brain
    py::class_<Brain>(m, "Brain")
        .def(py::init<Circuits&,
             PCNN_REF&,
             PCNN_REF&,
             ExplorationModule&,
             StationarySensory&,
             DensityPolicy&,
             float, float, int, int>(),
             py::arg("circuits"),
             py::arg("space"),
             py::arg("space_coarse"),
             py::arg("expmd"),
             py::arg("ssry"),
             py::arg("dpolicy"),
             py::arg("speed"),
             py::arg("speed_coarse"),
             py::arg("max_attempts") = 2,
             py::arg("forced_duration") = 10)
        .def("__call__", &Brain::call,
             py::arg("velocity"),
             py::arg("collision"),
             py::arg("reward"),
             py::arg("trigger"))
        .def("get_representation",
             &Brain::get_representation)
        .def("get_representation_coarse",
             &Brain::get_representation_coarse)
        .def("get_trg_representation",
             &Brain::get_trg_representation)
        .def("get_directive", &Brain::get_directive)
        .def("__str__", &Brain::str)
        .def("__repr__", &Brain::repr)
        .def("__len__", &Brain::len)
        .def("get_expmd", &Brain::get_expmd)
        .def("get_space", &Brain::get_space)
        .def("get_trg_idx", &Brain::get_trg_idx)
        .def("get_space_position_fine", &Brain::get_space_position_fine)
        .def("get_plan_idxs_coarse", &Brain::get_plan_idxs_coarse)
        /* .def("get_episodic_memory", &Brain::get_episodic_memory) */
        /* .def("make_edges_value", &Brain::make_edges_value) */
        .def("get_trg_position", &Brain::get_trg_position)
        .def("get_plan_idxs_fine", &Brain::get_plan_idxs_fine)
        .def("get_plan_idxs_coarse", &Brain::get_plan_idxs_coarse)
        .def("get_leaky_v", &Brain::get_leaky_v)
        .def("get_space_position_fine", &Brain::get_space_position_fine)
        .def("get_space_position_coarse", &Brain::get_space_position_coarse)
        .def("get_edge_representation", &Brain::get_edge_representation)
        /* .def("get_plan_positions", &Brain::get_plan_positions) */
        /* .def("get_plan_score", &Brain::get_plan_score) */
        /* .def("get_plan_scores", &Brain::get_plan_scores) */
        /* .def("get_plan_values", &Brain::get_plan_values) */
        /* .def("set_plan_positions", &Brain::set_plan_positions, */
        /*      py::arg("position")) */
        .def("reset", &Brain::reset);

    py::class_<Brainv2>(m, "Brainv2")

        .def(py::init<
             float, float, int, float, float, float,
             float, float, float, float,
             float, float, float, float,
             float, float, float,
             float, float, float,
             float, float,
             float,
             float, float, float, float,
             float, int,
             int, int
             >(),
             py::arg("local_scale_fine"),
             py::arg("local_scale_coarse"),
             py::arg("N"),
             py::arg("rec_threshold_fine"),
             py::arg("rec_threshold_coarse"),
             py::arg("speed"),

             py::arg("gain_fine"),
             py::arg("offset_fine"),
             py::arg("threshold_fine"),
             py::arg("rep_threshold_fine"),

             py::arg("gain_coarse"),
             py::arg("offset_coarse"),
             py::arg("threshold_coarse"),
             py::arg("rep_threshold_coarse"),

             py::arg("lr_da"),
             py::arg("threshold_da"),
             py::arg("tau_v_da"),

             py::arg("lr_bnd"),
             py::arg("threshold_bnd"),
             py::arg("tau_v_bnd"),

             py::arg("tau_ssry"),
             py::arg("threshold_ssry"),

             py::arg("threshold_circuit"),

             py::arg("rwd_weight"),
             py::arg("rwd_sigma"),
             py::arg("col_weight"),
             py::arg("col_sigma"),

             py::arg("action_delay"),
             py::arg("edge_route_interval"),

             py::arg("forced_duration"),
             py::arg("fine_tuning_min_duration"))
        .def("__call__", &Brainv2::call,
             py::arg("velocity"),
             py::arg("collision"),
             py::arg("reward"),
             py::arg("trigger"))
        .def("__str__", &Brainv2::str)
        .def("__repr__", &Brainv2::repr)
        .def("__len__", &Brainv2::len)
        .def("get_cell_count", &Brainv2::get_cell_count)
        .def("reset", &Brainv2::reset);


}


