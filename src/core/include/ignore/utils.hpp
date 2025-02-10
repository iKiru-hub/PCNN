#pragma once

#ifndef UTILS_HPP
#define UTILS_HPP

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
#include <array>
#include <Eigen/Dense>
#include <vector>
#include <random>



/* RANDOM */


int random_int(int min, int max, unsigned int seed = 0);

float random_float(float min, float max, unsigned int seed = 0);


/* MISCELLANEOUS */


int get_segments_intersection(float p0_x, float p0_y,
                              float p1_x, float p1_y,
                              float p2_x, float p2_y,
                              float p3_x, float p3_y,
                              float *i_x, float *i_y) {}


std::array<float, 2> reflect_point_over_segment(
                    float x, float y, float x1,
                    float y1, float x2, float y2) {}


Eigen::VectorXf linspace(float start, float end, int num) {}


/* ALGORITHMS */


std::vector<int> weighted_en_shortest_path(const Eigen::MatrixXf& connectivity_matrix,
                                           const Eigen::MatrixXf& edge_weights,
                                           const Eigen::VectorXf& node_weights,
                                           int start_node, int end_node) {}


std::vector<int> spatial_shortest_path(const Eigen::MatrixXf& connectivity_matrix,
                                       const Eigen::MatrixXf& node_coordinates,  // Nx2 matrix with (x,y) coordinates
                                       const Eigen::VectorXf& node_weights,      // Optional node penalties/costs
                                       int start_node, int end_node) {}


/* LINEAR ALGEBRA */


inline float cosine_similarity_vec(const Eigen::VectorXf& v1,
                                   const Eigen::VectorXf& v2) {}


inline float cosine_similarity_vec(const std::vector<float>& v1,
                                   const std::vector<float>& v2) {}


Eigen::VectorXf cosine_similarity_vector_matrix(
    const Eigen::VectorXf& vector,
    const Eigen::MatrixXf& matrix) {}


Eigen::MatrixXf cosine_similarity_matrix(
    const Eigen::MatrixXf& matrix) {}


float max_cosine_similarity_in_rows(
    const Eigen::MatrixXf& matrix, int idx) {}


float euclidean_distance(const std::array<float, 2>& v1,
                         const std::array<float, 2>& v2) {}


Eigen::MatrixXf connectivity_matrix(
    const Eigen::MatrixXf& matrix,
    float threshold = 0.5f) {}


Eigen::MatrixXf generate_lattice(int N, int length) {}


/* ACTIVATION FUNCTIONS */


inline Eigen::VectorXf generalized_sigmoid_vec(
    const Eigen::VectorXf& x, float offset = 1.0f,
    float gain = 1.0f, float clip = 0.0f) {}


inline std::vector<float> generalized_sigmoid_vec(
    const std::vector<float>& x, float offset = 1.0f,
    float gain = 1.0f, float clip = 0.0f) {}


#endif // UTILS_HPP
