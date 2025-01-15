#include "pcnn.hpp"
#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <Eigen/Dense>

#define PCNN_REF PCNNbase
#define CIRCUIT_SIZE 3

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

    // function `set_debug`
    m.def("set_debug", &set_debug,
          py::arg("flag"));

    /* PCNN MODEL */
    // (InputFilter) Grid Layer
    py::class_<GridLayerSq>(m, "GridLayerSq")
        .def(py::init<float, float, std::array<float, 2>>(),
             py::arg("sigma"),
             py::arg("speed"),
             py::arg("bounds") = std::array<float, 2>({0.0f, 1.0f}))
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

    // PCNN network model [GridSq]
    py::class_<PCNNbase>(m, "PCNNbase")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, GridNetworkSq&, int, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain") = 10.0f,
             py::arg("offset") = 0.5f,
             py::arg("clip_min") = 0.001f,
             py::arg("threshold") = 0.5f,
             py::arg("rep_threshold") = 0.5f,
             py::arg("rec_threshold") = 0.5f,
             py::arg("num_neighbors") = 8,
             py::arg("trace_tau") = 5.0f,
             py::arg("xfilter"),
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

    py::class_<PopulationMaxProgram>(m, "PopulationMaxProgram")
        .def(py::init<>())
        .def("__str__", &PopulationMaxProgram::str)
        .def("__repr__", &PopulationMaxProgram::repr)
        .def("__call__", &PopulationMaxProgram::call,
             py::arg("u"))
        .def("len", &PopulationMaxProgram::len)
        .def("get_value", &PopulationMaxProgram::get_value);

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
             PCNN_REF&, OneLayerNetwork&, int>(),
             py::arg("speed"),
             py::arg("circuits"),
             py::arg("trgp"),
             py::arg("space"),
             py::arg("eval_network"),
             py::arg("action_delay") = 1)
        .def("__call__", &ExperienceModule::call,
             py::arg("directive"),
             py::arg("position"))
        .def("__str__", &ExperienceModule::str)
        .def("__repr__", &ExperienceModule::repr)
        .def("get_action_seq", &ExperienceModule::get_action_seq)
        .def("get_plan", &ExperienceModule::get_plan)
        .def_readonly("new_plan", &ExperienceModule::new_plan)
        .def("get_values", &ExperienceModule::get_values);

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
        .def(py::init<std::array<float, CIRCUIT_SIZE+1>>(),
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
        .def("get_directive", &Brain::get_directive)
        .def("__str__", &Brain::str)
        .def("__repr__", &Brain::repr)
        .def("get_expmd", &Brain::get_expmd);

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


