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



struct DensityPolicy {

    float rwd_weight;
    float rwd_sigma;
    float col_weight;;
    float col_sigma;
    float rwd_drive = 0.0f ;
    float col_drive = 0.0f;

    float rwd_field_mod_fine;
    float rwd_field_mod_coarse;
    float col_field_mod_fine;
    float col_field_mod_coarse;

    bool remapping_flag;
    // da(fine), da(coarse), bnd(fine), bnd(coarse)
    std::array<bool, 4> remapping_option;
    // density(da), gain(da), density(bnd), gain(bnd)
    std::array<bool, 4> modulation_option;

    void call(PCNN_REF& space_fine,
              PCNN_REF& space_coarse,
              Circuits& circuits,
              GoalModule& goalmd,
              std::array<float, 2> displacement,
              float curr_da, float curr_bnd,
              float reward, float collision) {

        if (remapping_flag < 0 || space_fine.len() < 3) { return; }

        // +reward -collision
        if (reward > 0.1 && circuits.get_da_leaky_v() > 0.01f) {

            // update & remap
            rwd_drive = rwd_weight * curr_da;

            if (remapping_option[0]) {
                if (modulation_option[0]) {
                    space_fine.remap(circuits.get_da_weights(),
                                     displacement, rwd_sigma, rwd_drive);
                }
                if (modulation_option[1]) {
                    space_fine.modulate_gain(rwd_field_mod_fine);
                }
            }
            if (remapping_option[1]) {
                if (modulation_option[0]) {
                    space_coarse.remap(displacement, rwd_sigma, rwd_drive);
                }
                if (modulation_option[1]) {
                    space_coarse.modulate_gain(rwd_field_mod_coarse);
                }
            }
        } else if (collision > 0.1) {

            // udpate & remap
            col_drive = col_weight * curr_bnd;

            if (remapping_option[2]) {
                if (modulation_option[2]) {
                    space_fine.remap(circuits.get_bnd_weights(),
                                     {-1.0f*displacement[0], -1.0f*displacement[1]},
                                     col_sigma, col_drive);
                }
                if (modulation_option[3]) {
                    space_fine.modulate_gain(col_field_mod_fine);
                }
            }
            if (remapping_option[3]) {
                if (modulation_option[2]) {
                    space_coarse.remap({-1.0f*displacement[0], -1.0f*displacement[1]},
                                       col_sigma, col_drive);
                }
                if (modulation_option[3]) {
                    space_coarse.modulate_gain(col_field_mod_coarse);
                }
            }
        }
    }

    DensityPolicy(float rwd_weight, float rwd_sigma,
                  float col_weight, float col_sigma,
                  float rwd_field_mod_fine,
                  float rwd_field_mod_coarse,
                  float col_field_mod_fine,
                  float col_field_mod_coarse,
                  std::array<bool, 4> modulation_option,
                  int remapping_flag = -1):
        rwd_weight(rwd_weight), rwd_sigma(rwd_sigma),
        col_sigma(col_sigma), col_weight(col_weight),
        rwd_field_mod_fine(rwd_field_mod_fine),
        rwd_field_mod_coarse(rwd_field_mod_coarse),
        col_field_mod_fine(col_field_mod_fine),
        col_field_mod_coarse(col_field_mod_coarse),
        modulation_option(modulation_option),
        remapping_option(remapping_options(remapping_flag)) {}

    std::string str() { return "DensityPolicy"; }
    std::string repr() { return "DensityPolicy"; }
    float get_rwd_mod() { return rwd_drive; }
    float get_col_mod() { return col_drive; }

private:

    void remap_space(Eigen::VectorXf& block_weights,
                     PCNN_REF& space,
                     GoalModule& goalmd,
                     std::array<float, 2> displacement,
                     float sigma, float drive) {

        // get plan
        std::vector<int> plan_idxs = goalmd.get_plan_idxs_fine();

        // check: plan is empty
        if (plan_idxs.size() < 1) { return; }

        // current vspace position
        /* std::array<float, 2> v = space.get_position(); */
        std::array<float, 2> curr_positions = space.get_position();
        Eigen::MatrixXf centers = space.get_centers();

        // calculate the displacement between each point of the plan
        // and the previous one
        Eigen::MatrixXf prev_center = centers.row(plan_idxs[0]);
        float magnitude_i;
        float dist;
        float displacement_trg;
        for (int i = 1; i < plan_idxs.size(); i++) {

            // center of the current point
            Eigen::MatrixXf curr_center = centers.row(plan_idxs[i]);

            // block weights
            if (curr_center(0) < -900.0f || block_weights(i) > 0.0f) { continue; }

            // displacement from the previous point
            std::array<float, 2> displacement = {
                curr_center(0) - prev_center(0),
                curr_center(1) - prev_center(1)};

            // distance from the current point
            displacement_trg = std::sqrt((curr_center(0) - curr_positions[0]) * \
                                    (curr_center(0) - curr_positions[0]) + \
                                    (curr_center(1) - curr_positions[1]) * \
                                    (curr_center(1) - curr_positions[1]));
            dist = std::exp(-displacement_trg * displacement_trg / sigma);

            magnitude_i = dist * drive;

            // remap the space
            space.single_remap(plan_idxs[i], displacement, magnitude_i);

            prev_center = centers.row(plan_idxs[i]);
        }
    }
};


#endif // UTILS_HPP
