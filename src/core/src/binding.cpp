#include "pcnn.hpp"
/* #include "utils.hpp" */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>

#define PCNN_REF PCNN
#define CIRCUIT_SIZE 5

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

    /* PCNN MODEL */
    py::class_<GridLayer>(m, "GridLayer")
        .def(py::init<float, float, std::array<float, 4>>(),
             py::arg("sigma"),
             py::arg("speed"),
             py::arg("bounds") = std::array<float, 4>({0.0f, 1.0f, 0.0f, 1.0f}))
        .def("__call__", &GridLayer::call,
             py::arg("v"))
        .def("__str__", &GridLayer::str)
        .def("__repr__", &GridLayer::repr)
        .def("__len__", &GridLayer::len)
        .def("simulate_one_step", &GridLayer::simulate_one_step,
             py::arg("v"))
        .def("get_centers", &GridLayer::get_centers)
        .def("get_activation", &GridLayer::get_activation)
        .def("get_positions", &GridLayer::get_positions)
        .def("reset", &GridLayer::reset,
             py::arg("v"));

    // Grid Layer Network
    py::class_<GridNetwork>(m, "GridNetwork")
        .def(py::init<std::vector<GridLayer>>(),
             py::arg("layers"))
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

    // PCNN network model
    py::class_<PCNN>(m, "PCNN")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             float, GridNetwork, float,
             int, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain"),
             py::arg("offset"),
             py::arg("clip_min"),
             py::arg("threshold"),
             py::arg("rep_threshold"),
             py::arg("rec_threshold"),
             py::arg("min_rep_threshold"),
             py::arg("xfilter"),
             py::arg("tau_trace") = 2.0f,
             py::arg("remap_tag_frequency") = 1,
             py::arg("name") = "fine")
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
        .def("calculate_closest_index", &PCNN::calculate_closest_index,
             py::arg("c"))
        .def("get_neighbourhood_node_degree",
                    &PCNN::get_neighbourhood_node_degree,
             py::arg("idx"))
        .def("get_connectivity", &PCNN::get_connectivity)
        .def("get_centers", &PCNN::get_centers,
             py::arg("nonzero") = false)
        .def("get_node_degrees", &PCNN::get_node_degrees)
        .def("get_delta_update", &PCNN::get_delta_update)
        .def("get_position", &PCNN::get_position)
        .def("get_trace_value", &PCNN::get_trace_value,
             py::arg("idx"))
        .def("get_trace_v2", &PCNN::get_trace_v2)
        .def("remap", &PCNN::remap,
             py::arg("block_weights"),
             py::arg("velocity"),
             py::arg("width"),
             py::arg("magnitude"))
        .def("simulate_one_step", &PCNN::simulate_one_step,
             py::arg("v"))
        .def("reset_gcn", &PCNN::reset_gcn,
             py::arg("v"))
        .def("reset", &PCNN::reset);

    // Density Policy
    py::class_<DensityPolicy>(m, "DensityPolicy")
        .def(py::init<float, float, float,
             float, float, float, std::array<bool, 4>>(),
             py::arg("rwd_weight"),
             py::arg("rwd_sigma"),
             py::arg("col_weight"),
             py::arg("col_sigma"),
             py::arg("rwd_field_mod"),
             py::arg("col_field_mod"),
             py::arg("options") = std::array<bool, 4>({true, true, true, true}))
        .def("__call__", &DensityPolicy::call,
             py::arg("space"),
             py::arg("circuits"),
             py::arg("displacement"),
             py::arg("da_value"),
             py::arg("bnd_value"),
             py::arg("reward"),
             py::arg("collision"))
        .def("__str__", &DensityPolicy::str)
        .def("__repr__", &DensityPolicy::repr)
        .def("get_rwd_mod", &DensityPolicy::get_rwd_mod)
        .def("get_col_mod", &DensityPolicy::get_col_mod);

    /* MODULATION */
    py::class_<BaseModulation>(m, "BaseModulation")
        .def(py::init<std::string, int, float, float,
             float, float, float, float, float, float>(),
             py::arg("name"),
             py::arg("size"),
             py::arg("lr") = 0.1f,
             py::arg("lr_pred") = 0.1f,
             py::arg("threshold") = 0.0f,
             py::arg("max_w") = 1.0f,
             py::arg("tau_v") = 5.0f,
             py::arg("eq_v") = 0.0f,
             py::arg("min_v") = 0.01f,
             py::arg("mask_threshold") = 0.01f)
        .def("__str__", &BaseModulation::str)
        /* .def("__repr__", &BaseModulation::repr) */
        .def("__len__", &BaseModulation::len)
        .def("__call__", &BaseModulation::call,
             py::arg("u"),
             py::arg("x") = 0.0f,
             py::arg("simulate") = false)
        .def("get_output", &BaseModulation::get_output)
        .def("get_leaky_v", &BaseModulation::get_leaky_v)
        .def("get_weights", &BaseModulation::get_weights);

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
        .def("make_value_mask", &Circuits::make_value_mask,
             py::arg("strict") = false)
        .def("make_prediction", &Circuits::make_prediction,
             py::arg("representation"))
        .def("get_output", &Circuits::get_output)
        .def("get_da_weights", &Circuits::get_da_weights)
        .def("get_bnd_weights", &Circuits::get_bnd_weights)
        .def("get_bnd_mask", &Circuits::get_bnd_mask)
        .def("get_bnd_value", &Circuits::get_bnd_value)
        .def("get_value_mask", &Circuits::get_value_mask)
        .def("get_da_leaky_v", &Circuits::get_da_leaky_v)
        .def("get_bnd_leaky_v", &Circuits::get_bnd_leaky_v)
        .def("get_leaky_v", &Circuits::get_leaky_v)
        .def("reset", &Circuits::reset);

    // make a PCNN object 
    m.def("make_space", &pcl::make_space,
          py::arg("gc_sigma"),
          py::arg("gc_scales"),
          py::arg("local_scale"),
          py::arg("N"),
          py::arg("rec_threshold_fine"),
          py::arg("speed"),
          py::arg("min_rep_threshold"),

          py::arg("gain_fine"),
          py::arg("offset_fine"),
          py::arg("threshold_fine"),
          py::arg("rep_threshold_fine"),
          py::arg("tau_trace_fine"),
          py::arg("remap_tag_frequency"));
}


