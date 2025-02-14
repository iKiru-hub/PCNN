#include "pcnn.hpp"
/* #include "utils.hpp" */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>

// === DEFINE GCN PARAMETERS ===


#define GCS 1
#define GCH 2

// Set the current GCN_REF
#define GCN_REF_ID GCH

// Set GCN_SIZE based on GCN_REF
#if GCN_REF_ID == GCS
    #define GCL_SIZE 36
    #define GCL_SIZE_SQRT 6
    #define GCN_REF GridNetworkSq
    #define GCL_REF GridLayerkSq
#elif GCN_REF_ID == GCH
    #define GCL_SIZE 25
    #define GCL_SIZE_SQRT 5
    #define GCN_REF GridHexNetwork
    #define GCL_REF GridHexLayer
#else
    #error "Unknown GCN_REF value"
#endif

// === DEFINE the rest ===

#define CIRCUIT_SIZE 5


// === DEFINE the rest ===

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {


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

    // Grid Layer Network
    py::class_<GridNetwork<Layer>>(m, "GridNetwork")
        .def(py::init<std::vector<Layer>>(), py::arg("layers_input"))
        .def("__call__", &GridNetwork::call,
                py::arg("x"))
        .def("__str__", &GridNetwork::str)
        .def("__repr__", &GridNetwork::repr)
        .def("__len__", &GridNetwork::len)
        .def("simulate_one_step", &GridNetwork::simulate_one_step,
             py::arg("v"))
        .def("get_activation", &GridNetwork::get_activation)
        .def("get_centers", &GridNetwork::get_centers)
        .def("get_num_layers", &GridNetwork::get_num_layers)
        .def("get_positions", &GridNetwork::get_positions);

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
        .def("get_activation", &GridHexNetwork::get_activation)
        .def("get_centers", &GridHexNetwork::get_centers)
        .def("get_num_layers", &GridHexNetwork::get_num_layers)
        .def("get_positions", &GridHexNetwork::get_positions)
        .def("reset", &GridHexNetwork::reset,
             py::arg("v"));


    // PCNN network model
    py::class_<PCNN>(m, "PCNN")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, GCN_REF&, std::string>(),
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
        .def("__call__", &PCNN::call,
             py::arg("v"))
        .def("__str__", &PCNN::str)
        .def("__len__", &PCNN::len)
        .def("__repr__", &PCNN::repr)
        .def("update", &PCNN::update)
        .def("get_activation", &PCNN::get_activation)
        .def("get_activation_gcn",
             &PCNN::get_activation_gcn)
        .def("get_size", &PCNN::get_size)
        .def("get_wff", &PCNN::get_wff)
        .def("get_wrec", &PCNN::get_wrec)
        .def("make_edges", &PCNN::make_edges)
        .def("get_connectivity", &PCNN::get_connectivity)
        .def("get_centers", &PCNN::get_centers,
             py::arg("nonzero") = false)
        .def("get_node_degrees", &PCNN::get_node_degrees)
        .def("get_delta_update", &PCNN::get_delta_update)
        .def("get_position", &PCNN::get_position)
        /* .def("simulate", &PCNN::simulate, */
        /*      py::arg("v"), */
        /*      py::arg("sim_gc_positions")) */
        .def("simulate_one_step", &PCNN::simulate_one_step,
             py::arg("v"))
        .def("modulate_gain", &PCNN::modulate_gain,
             py::arg("mod"))
        .def("modulate_rep", &PCNN::modulate_rep,
             py::arg("mod"))
        .def("modulate_threshold", &PCNN::modulate_threshold,
             py::arg("mod"))
        .def("get_gain", &PCNN::get_gain)
        .def("get_rep", &PCNN::get_rep)
        .def("get_threshold", &PCNN::get_threshold)
        .def("reset_gcn", &PCNN::reset_gcn,
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
             py::arg("da"),
             py::arg("bnd"),
             py::arg("threshold"))
        .def("__str__", &Circuits::str)
        .def("__repr__", &Circuits::repr)
        .def("__len__", &Circuits::len)
        .def("__call__", &Circuits::call,
             py::arg("u"),
             py::arg("collision"),
             py::arg("reward"),
             py::arg("simulate") = false)
        .def("get_output", &Circuits::get_output)
        .def("get_leaky_v", &Circuits::get_leaky_v);

    /* MODULES */

    // 2 layer network
    py::class_<ExplorationModule>(m, "ExplorationModule")
        .def(py::init<float, Circuits&, PCNN&, float, int>(),
             py::arg("speed"),
             py::arg("circuits"),
             py::arg("space"),
             py::arg("action_delay") = 1.0f,
             py::arg("edge_route_interval") = 100)
        .def("__call__", &ExplorationModule::call,
             py::arg("directive"),
             py::arg("rejection"))
        .def("__str__", &ExplorationModule::str)
        .def("__repr__", &ExplorationModule::repr)
        .def("confirm_edge_walk", &ExplorationModule::confirm_edge_walk)
        .def("reset_rejected_indexes", &ExplorationModule::reset_rejected_indexes);

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
             PCNN&,
             PCNN&,
             ExplorationModule&,
             StationarySensory&,
             DensityPolicy&,
             float, float, int, int, float>(),
             py::arg("circuits"),
             py::arg("space"),
             py::arg("space_coarse"),
             py::arg("expmd"),
             py::arg("ssry"),
             py::arg("dpolicy"),
             py::arg("speed"),
             py::arg("speed_coarse"),
             py::arg("max_attempts") = 2,
             py::arg("forced_duration") = 10,
             py::arg("min_weight_value") = 0.3)
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
        .def("get_trg_position", &Brain::get_trg_position)
        .def("get_plan_idxs_fine", &Brain::get_plan_idxs_fine)
        .def("get_plan_idxs_coarse", &Brain::get_plan_idxs_coarse)
        .def("get_leaky_v", &Brain::get_leaky_v)
        .def("get_space_position_fine", &Brain::get_space_position_fine)
        .def("get_space_position_coarse", &Brain::get_space_position_coarse)
        .def("get_edge_representation", &Brain::get_edge_representation)
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
             int, int, float
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
             py::arg("fine_tuning_min_duration"),
             py::arg("min_weight_value") = 0.3)
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


