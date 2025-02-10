/* #include "utils.hpp" */
#include <iostream>
#include <Eigen/Dense>
#include <unordered_map>
#include <memory>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <cassert>



/* ========================================== */

/* things that are delicate:
 *
 * o) velocity initial position
 *
 *
 *
 *
 */


/* ========================================== */

#define GCL_SIZE 36
#define GCL_SIZE_SQRT 6
#define PCNN_REF PCNNsqv2
#define GCN_REF GridNetworkSq
#define CIRCUIT_SIZE 2
#define ACTION_SPACE_SIZE 16
#define POLICY_INPUT 5
#define POLICY_OUTPUT 6
#define POLICY_HIDDEN 3



/* ========================================== */

class Hexagon {
public:
    Hexagon();
    std::array<float, 2> call(float x, float y) {}
    std::string str() {}
    std::string repr() {}
    std::array<std::array<float, 2>, 6> get_centers() {}
};


struct GSparams { GSparams(float offset, float gain, float clip) {} };


struct VelocitySpace {
    VelocitySpace(int size, float threshold) {}
    std::array<float, 2> call(const std::array<float, 2> v) {}
    void update(int idx, int current_size, bool update_center = true) {}
    void remap_center(int idx, int size, std::array<float, 2> displacement) {}
    void add_blocked_edge(int i, int j) {}
    Eigen::MatrixXf get_centers(bool nonzero=false) {}
    std::vector<std::array<std::array<float, 2>, 2>> make_edges() {}
    std::vector<std::array<std::array<float, 2>, 3>> make_edges_value(
               Eigen::MatrixXf& values) {}
    int calculate_closest_index(std::array<float, 2> c) {}
};


