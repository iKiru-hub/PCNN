#include "pcnn.hpp"
#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>

#define PCNN_REF PCNNsq

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

    // function `set_debug`
    m.def("set_debug", &set_debug,
          py::arg("flag"));

    /* PCNN MODEL */
    // (InputFilter) Grid Layer
    py::class_<GridLayerSq>(m, "GridLayerSq")
        .def(py::init<float, float>(),
             py::arg("sigma"),
             py::arg("speed"))
        .def("__call__", &GridLayerSq::call,
             py::arg("v"))
        .def("__str__", &GridLayerSq::str)
        .def("__repr__", &GridLayerSq::repr)
        .def("__len__", &GridLayerSq::len)
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
        .def("fwd_position", &GridNetworkSq::fwd_position,
                py::arg("v"))
        .def("get_activation", &GridNetworkSq::get_activation)
        .def("get_centers", &GridNetworkSq::get_centers)
        .def("get_num_layers", &GridNetworkSq::get_num_layers)
        .def("get_positions", &GridNetworkSq::get_positions);

    // (InputFilter) Grid Layer
    py::class_<GridLayer>(m, "GridLayer")
        .def(py::init<int, float, float,
             std::array<float, 4>,
             std::string,
             std::string>(),
             py::arg("N"),
             py::arg("sigma"),
             py::arg("speed"),
             py::arg("init_bounds"),
             py::arg("boundary_type") = "square",
             py::arg("basis_type") = "square")
        .def("__call__", &GridLayer::call,
             py::arg("v"))
        .def("__str__", &GridLayer::str)
        .def("__repr__", &GridLayer::repr)
        .def("__len__", &GridLayer::len)
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
        .def("fwd_position", &GridNetwork::fwd_position,
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
        .def("ach_modulation", &PCNNgridhex::ach_modulation,
             py::arg("ach"))
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


    // PCNN network model [Grid]
    py::class_<PCNNgrid>(m, "PCNNgrid")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, GridNetwork, std::string>(),
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
        .def("__call__", &PCNNgrid::call,
             py::arg("v"),
             py::arg("frozen") = false,
             py::arg("traced") = true)
        .def("__str__", &PCNNgrid::str)
        .def("__len__", &PCNNgrid::len)
        .def("__repr__", &PCNNgrid::repr)
        .def("update", &PCNNgrid::update,
             py::arg("x") = -1.0,
             py::arg("y") = -1.0)
        .def("get_activation", &PCNNgrid::get_activation)
        .def("get_activation_gcn",
             &PCNNgrid::get_activation_gcn)
        .def("ach_modulation", &PCNNgrid::ach_modulation,
             py::arg("ach"))
        .def("get_size", &PCNNgrid::get_size)
        .def("get_trace", &PCNNgrid::get_trace)
        .def("get_wff", &PCNNgrid::get_wff)
        .def("get_wrec", &PCNNgrid::get_wrec)
        .def("get_connectivity", &PCNNgrid::get_connectivity)
        .def("get_centers", &PCNNgrid::get_centers,
             py::arg("nonzero") = false)
        .def("get_delta_update", &PCNNgrid::get_delta_update)
        .def("fwd_ext", &PCNNgrid::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNNgrid::fwd_int,
             py::arg("a"))
        .def("reset_gcn", &PCNNgrid::reset_gcn,
             py::arg("v"));

    // (InputFilter) Random Layer
    py::class_<RandLayer>(m, "RandLayer")
        .def(py::init<int>(),
             py::arg("N"))
        .def("__call__", &RandLayer::call,
             py::arg("x"))
        .def("__str__", &RandLayer::str)
        .def("__len__", &RandLayer::len)
        .def("__repr__", &RandLayer::repr)
        .def("get_centers", &RandLayer::get_centers)
        .def("get_activation", &RandLayer::get_activation);

    // PCNN network model [Rand]
    py::class_<PCNNrand>(m, "PCNNrand")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, RandLayer, std::string>(),
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
        .def("__call__", &PCNNrand::call,
             py::arg("x"),
             py::arg("frozen") = false,
             py::arg("traced") = true)
        .def("__str__", &PCNNrand::str)
        .def("__len__", &PCNNrand::len)
        .def("__repr__", &PCNNrand::repr)
        .def("update", &PCNNrand::update)
        .def("ach_modulation", &PCNNrand::ach_modulation,
             py::arg("ach"))
        .def("get_size", &PCNNrand::get_size)
        .def("get_trace", &PCNNrand::get_trace)
        .def("get_wff", &PCNNrand::get_wff)
        .def("get_wrec", &PCNNrand::get_wrec)
        .def("get_connectivity", &PCNNrand::get_connectivity)
        .def("get_delta_update", &PCNNrand::get_delta_update)
        .def("get_centers", &PCNNrand::get_centers)
        .def("fwd_ext", &PCNNrand::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNNrand::fwd_int,
             py::arg("a"));

    // PCNN network model [GridSq]
    py::class_<PCNNsq>(m, "PCNNsq")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, GridNetworkSq&, std::string>(),
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
        .def("__call__", &PCNNsq::call,
             py::arg("v"),
             py::arg("frozen") = false,
             py::arg("traced") = true)
        .def("__str__", &PCNNsq::str)
        .def("__len__", &PCNNsq::len)
        .def("__repr__", &PCNNsq::repr)
        .def("update", &PCNNsq::update,
             py::arg("x") = -1.0,
             py::arg("y") = -1.0)
        .def("get_activation", &PCNNsq::get_activation)
        .def("get_activation_gcn",
             &PCNNsq::get_activation_gcn)
        .def("ach_modulation", &PCNNsq::ach_modulation,
             py::arg("ach"))
        .def("get_size", &PCNNsq::get_size)
        .def("get_trace", &PCNNsq::get_trace)
        .def("get_wff", &PCNNsq::get_wff)
        .def("get_wrec", &PCNNsq::get_wrec)
        .def("get_connectivity", &PCNNsq::get_connectivity)
        .def("get_centers", &PCNNsq::get_centers,
             py::arg("nonzero") = false)
        .def("get_delta_update", &PCNNsq::get_delta_update)
        .def("fwd_ext", &PCNNsq::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNNsq::fwd_int,
             py::arg("a"))
        .def("reset_gcn", &PCNNsq::reset_gcn,
             py::arg("v"));

    // (InputFilter) Place Cell Layer
    py::class_<PCLayer>(m, "PCLayer")
        .def(py::init<int, float, std::array<float, 4>>(),
             py::arg("n"),
             py::arg("sigma"),
             py::arg("bounds"))
        .def("__call__", &PCLayer::call,
             py::arg("x"))
        .def("__str__", &PCLayer::str)
        .def("__len__", &PCLayer::len)
        .def("__repr__", &PCLayer::repr)
        .def("get_centers", &PCLayer::get_centers)
        .def("get_activation", &RandLayer::get_activation);

    // PCNN network model
    py::class_<PCNN>(m, "PCNN")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, PCLayer, std::string>(),
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
        .def("__call__", &PCNN::call,
             py::arg("x"),
             py::arg("frozen") = false,
             py::arg("traced") = true)
        .def("__str__", &PCNN::str)
        .def("__len__", &PCNN::len)
        .def("__repr__", &PCNN::repr)
        .def("update", &PCNN::update)
        .def("ach_modulation", &PCNN::ach_modulation,
             py::arg("ach"))
        .def("get_size", &PCNN::get_size)
        .def("get_trace", &PCNN::get_trace)
        .def("get_wff", &PCNN::get_wff)
        .def("get_wrec", &PCNN::get_wrec)
        .def("get_connectivity", &PCNN::get_connectivity)
        .def("get_delta_update", &PCNN::get_delta_update)
        .def("get_centers", &PCNN::get_centers)
        .def("fwd_ext", &PCNN::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNN::fwd_int,
             py::arg("a"));


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

    // LeakyVariable ND
    py::class_<LeakyVariableND>(m, "LeakyVariableND")
        .def(py::init<std::string, float, float, int,
             float>(),
             py::arg("name"),
             py::arg("eq"),
             py::arg("tau"),
             py::arg("ndim"),
             py::arg("min_v") = 0.0)
        .def("__call__", &LeakyVariableND::call,
             py::arg("x"),
             py::arg("simulate") = false)
        .def("__str__", &LeakyVariableND::str)
        .def("__repr__", &LeakyVariableND::repr)
        .def("__len__", &LeakyVariableND::len)
        .def("print_v", &LeakyVariableND::print_v)
        .def("get_v", &LeakyVariableND::get_v)
        .def("get_name", &LeakyVariableND::get_name)
        .def("set_eq", &LeakyVariableND::set_eq,
             py::arg("eq"))
        .def("reset", &LeakyVariableND::reset);

    // Density modulation
    py::class_<DensityMod>(m, "DensityMod")
        .def(py::init<std::array<float, 5>, float>(),
             py::arg("weights"),
             py::arg("theta"))
        .def("__str__", &DensityMod::str)
        .def("__call__", &DensityMod::call,
             py::arg("x"))
        .def("get_value", &DensityMod::get_value);

    /* MODULATION */
    py::class_<BaseModulation>(m, "BaseModulation")
        .def(py::init<std::string, int, float, float,
             float, float, float, float, float,
             float>(),
             py::arg("name"),
             py::arg("size"),
             py::arg("lr") = 0.1f,
             py::arg("threshod") = 0.0f,
             py::arg("offset") = 0.0f,
             py::arg("gain") = 5.0f,
             py::arg("clip") = 0.001f,
             py::arg("eq") = 0.0f,
             py::arg("tau") = 5.0f,
             py::arg("min_v") = 0.01f)
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
        .def(py::init<BaseModulation&, BaseModulation&>(),
             py::arg("da"),
             py::arg("bnd"))
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
    py::class_<TargetProgram>(m, "TargetProgram")
        .def(py::init<float,
             Eigen::MatrixXf,
             BaseModulation&, int, float>(),
             py::arg("threshold1"),
             py::arg("wrec"),
             py::arg("modulator"),
             py::arg("max_depth") = 20,
             py::arg("threshold2") = 0.8f)
        .def("__len__", &TargetProgram::len)
        .def("__str__", &TargetProgram::str)
        .def("__repr__", &TargetProgram::repr)
        .def("is_active", &TargetProgram::is_active)
        .def("update", &TargetProgram::update,
             py::arg("activation"))
        .def("evaluate", &TargetProgram::evaluate,
             py::arg("next_representation"),
             py::arg("curr_representation"))
        .def("set_wrec", &TargetProgram::set_wrec,
             py::arg("wrec"))
        .def("get_trg_representation",
             &TargetProgram::get_trg_representation);

    // 2 layer network
    py::class_<ExperienceModule>(m, "ExperienceModule")
        .def(py::init<float, Circuits&, TargetProgram&,
             PCNN_REF&, OneLayerNetwork&>(),
             py::arg("speed"),
             py::arg("circuits"),
             py::arg("trgp"),
             py::arg("space"),
             py::arg("eval_network"))
        .def("__call__", &ExperienceModule::call,
             py::arg("directive"))
        .def("__str__", &ExperienceModule::str)
        .def("__repr__", &ExperienceModule::repr)
        .def("get_action_seq", &ExperienceModule::get_action_seq)
        .def("get_plan", &ExperienceModule::get_plan);

    /* ACTION SAMPLING MODULE */

    // Sampling Module
    py::class_<ActionSampling2D>(m, "ActionSampling2D")
        .def(py::init<std::string, float>(),
             py::arg("name"),
             py::arg("speed"))
        .def("__call__", &ActionSampling2D::call,
             py::arg("keep") = false)
        .def("update", &ActionSampling2D::update,
             py::arg("score") = 0.0)
        .def("__len__", &ActionSampling2D::len)
        .def("__str__", &ActionSampling2D::str)
        .def("__repr__", &ActionSampling2D::repr)
        .def("reset", &ActionSampling2D::reset)
        .def("sample_once", &ActionSampling2D::sample_once)
        .def("is_done", &ActionSampling2D::is_done)
        .def("get_idx", &ActionSampling2D::get_idx)
        .def("get_counter", &ActionSampling2D::get_counter)
        .def("get_values", &ActionSampling2D::get_values);

    // 2 layer network
    py::class_<TwoLayerNetwork>(m, "TwoLayerNetwork")
        .def(py::init<std::array<std::array<float, 2>, 5>,
             std::array<float, 2>>(),
             py::arg("w_hidden"),
             py::arg("w_output"))
        .def("__call__", &TwoLayerNetwork::call,
             py::arg("x"))
        .def("__str__", &TwoLayerNetwork::str);

    // 1 layer network
    py::class_<OneLayerNetwork>(m, "OneLayerNetwork")
        .def(py::init<std::vector<float>>(),
             py::arg("weights"))
        .def("__call__", &OneLayerNetwork::call,
             py::arg("x"))
        .def("__str__", &OneLayerNetwork::str)
        .def("len", &OneLayerNetwork::len)
        .def("get_weights", &OneLayerNetwork::get_weights);

    // Hexagon
    py::class_<Hexagon>(m, "Hexagon")
        .def(py::init<>())
        .def("__call__", &Hexagon::call,
             py::arg("x"),
             py::arg("y"))
        .def("__str__", &Hexagon::str)
        .def("get_centers", &Hexagon::get_centers);

    // Brain
    py::class_<Brain>(m, "Brain")
        .def(py::init<Circuits&,
             PCNN_REF&,
             TargetProgram&,
             ExperienceModule&>(),
             py::arg("circuits"),
             py::arg("pcnn"),
             py::arg("trgp"),
             py::arg("expmd"))
        .def("__call__", &Brain::call,
             py::arg("v"),
             py::arg("collision") = 0.0f,
             py::arg("reward") = 0.0f,
             py::arg("position"))
        .def("get_representation",
             &Brain::get_representation)
        .def("get_trg_representation",
             &Brain::get_trg_representation)
        .def("__str__", &Brain::str)
        .def("__repr__", &Brain::repr);

    // Brain Hex
    py::class_<BrainHex>(m, "BrainHex")
        .def(py::init<Circuits&,
             PCNNgridhex&, ActionSampling2D&,
             TargetProgram&>(),
             py::arg("circuits"),
             py::arg("pcnn"),
             py::arg("sampler"),
             py::arg("trgp"))
        .def("__call__", &BrainHex::call,
             py::arg("v"),
             py::arg("collision") = 0.0f,
             py::arg("reward") = 0.0f,
             py::arg("position"))
        .def("get_representation",
             &BrainHex::get_representation)
        .def("get_trg_representation",
             &BrainHex::get_trg_representation)
        .def("__str__", &BrainHex::str)
        .def("__repr__", &BrainHex::repr);

}


