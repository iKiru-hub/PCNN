#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <memory>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <cassert>
#include <vector>
#include <queue>


int SEED = 0;


/* ========================================== */

// blank log function
void LOG(const std::string& msg) { std::cout << msg << std::endl; }

/* ------------------------------------------ */
/* PATH ALGORITHMS */

std::vector<int> weighted_en_shortest_path(const Eigen::MatrixXf&, const Eigen::MatrixXf&,
                                           const Eigen::VectorXf&, int, int);

std::vector<int> spatial_shortest_path(const Eigen::MatrixXf& connectivity_matrix,
                                       // Nx2 matrix with (x,y) coordinates
                                       const Eigen::MatrixXf& node_coordinates,
                                       // Optional node penalties/costs
                                       const Eigen::VectorXf& node_weights,
                                       int start_node, int end_node);


/* ------------------------------------------ */
/* LINEAR ALGEBRA */


inline float cosine_similarity_vec(const Eigen::VectorXf&,
                                   const Eigen::VectorXf&);

Eigen::VectorXf cosine_similarity_vector_matrix(const Eigen::VectorXf&,
                                                const Eigen::MatrixXf&);

Eigen::MatrixXf cosine_similarity_matrix(const Eigen::MatrixXf&);

float max_cosine_similarity_in_rows(const Eigen::MatrixXf&, int);

float euclidean_distance(const std::array<float, 2>&,
                         const std::array<float, 2>&);

float gaussian_distance(const Eigen::VectorXf& v1,
                        const Eigen::VectorXf& v2,
                        float sigma = 1.0f);


/* MISCELLANEOUS */


Eigen::MatrixXf generate_lattice(int, int);

// Return type that contains both intersection status and coordinates
struct IntersectionResult {
    bool intersects;
    float x;
    float y;
};

// Alternative using std::tuple if preferred
using IntersectionTuple = std::tuple<bool, float, float>;

IntersectionResult get_segments_intersection(float, float, float, float,
                                             float, float, float, float);

// Alternative version using std::tuple if preferred
IntersectionTuple get_segments_intersection_tuple(float, float, float, float,
                                                  float, float, float, float);

std::array<float, 2> reflect_point_over_segment(float, float, float,
                                                float, float, float);

Eigen::VectorXf linspace(float, float, int);

std::vector<float> linspace_vec(float start, float end, int num,
                                bool startpoint = true,
                                bool endpoint = true);

Eigen::MatrixXf connectivity_matrix(const Eigen::MatrixXf& matrix,
                                    float threshold = 0.5f);


/* ACTIVATION FUNCTIONS */


float generalized_sigmoid(float x, float offset = 1.0f,
                          float gain = 1.0f, float clip = 0.0f);

inline Eigen::VectorXf generalized_sigmoid_vec(const Eigen::VectorXf& x,
    float offset = 1.0f, float gain = 1.0f, float clip = 0.0f);

inline std::vector<float> generalized_sigmoid_vec(
    const std::vector<float>& x, float offset = 1.0f,
    float gain = 1.0f, float clip = 0.0f);

