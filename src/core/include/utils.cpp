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




Eigen::MatrixXf generate_lattice(int N, int length) {

    // Determine the number of points along each axis
    int grid_size = static_cast<int>(std::sqrt(N));
    if (grid_size * grid_size < N) {
        grid_size += 1; // Ensure we have at least N points
    }
    float grid_size_f = static_cast<float>(grid_size);
    float length_f = static_cast<float>(length);
    // Step size between points
    float step = length_f / (grid_size_f - 1);

    // Initialize a matrix to store the points
    Eigen::MatrixXf points(N, 2);

    // Generate lattice points
    int count = 0;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            if (count >= N) break; // Stop if we have enough points

            points(count, 0) = i * step;
            points(count, 1) = j * step;
            ++count;
        }
        if (count >= N) break;
    }

    return points.topRows(count); // Return only the rows with valid points
}


/* ALGORITHMS */


std::vector<int> weighted_en_shortest_path(const Eigen::MatrixXf& connectivity_matrix,
                                           const Eigen::MatrixXf& edge_weights,
                                           const Eigen::VectorXf& node_weights,
                                           int start_node, int end_node) {
    int num_nodes = connectivity_matrix.rows();
    std::vector<float> distances(num_nodes, std::numeric_limits<float>::infinity());
    std::vector<int> parent(num_nodes, -1);
    std::vector<bool> finalized(num_nodes, false);

    // Use priority queue with pair of (distance, node)
    std::priority_queue<std::pair<float, int>> pq;

    // Initialize start node
    distances[start_node] = node_weights(start_node);
    pq.push({-distances[start_node], start_node});

    while (!pq.empty()) {
        int current_node = pq.top().second;
        float current_dist = -pq.top().first;
        pq.pop();

        // Skip if we've already finalized this node or found a better path
        if (finalized[current_node] || current_dist > distances[current_node]) {
            continue;
        }

        // Mark this node as finalized
        finalized[current_node] = true;

        // If we've found the end node, we're done since we've found the shortest path
        if (current_node == end_node) {
            break;
        }

        // Check all neighbors
        for (int neighbor = 0; neighbor < num_nodes; ++neighbor) {
            // Check if there's a connection and the neighbor is not finalized
            if (connectivity_matrix(current_node, neighbor) == 1 && !finalized[neighbor]) {
                // Calculate new distance:
                // 1. Current path distance
                // 2. Node weight of the current node
                // 3. Edge weight between current and neighbor
                // 4. Node weight of the neighbor
                float new_distance = distances[current_node]
                                     + node_weights(current_node)
                                     + edge_weights(current_node, neighbor)
                                     + node_weights(neighbor);

                // If we've found a better path
                if (new_distance < distances[neighbor]) {
                    distances[neighbor] = new_distance;
                    parent[neighbor] = current_node;
                    pq.push({-new_distance, neighbor});
                }
            }
        }
    }

    // Reconstruct the path
    std::vector<int> path;
    int current = end_node;

    // Check if end node is reachable
    if (distances[end_node] == std::numeric_limits<float>::infinity()) {
        return {};
    }

    while (current != -1) {
        path.push_back(current);
        current = parent[current];
    }
    std::reverse(path.begin(), path.end());

    // Verify the path starts at the start node
    if (path.empty() || path[0] != start_node) {
        return {};
    }

    return path;
}


std::vector<int> spatial_shortest_path(const Eigen::MatrixXf& connectivity_matrix,
                                       const Eigen::MatrixXf& node_coordinates,
                                       const Eigen::VectorXf& node_weights,
                                       int start_node, int end_node) {

    LOG("calculating spatial shortest path...");

    int num_nodes = connectivity_matrix.rows();
    std::vector<float> distances(num_nodes, std::numeric_limits<float>::infinity());
    std::vector<int> parent(num_nodes, -1);
    std::vector<bool> finalized(num_nodes, false);

    // Priority queue with pair of (distance, node)
    std::priority_queue<std::pair<float, int>> pq;

    // Initialize start node
    distances[start_node] = 0;  // Start with zero distance
    pq.push({0, start_node});

    float new_distance = 0.0f;

    while (!pq.empty()) {
        int current_node = pq.top().second;
        float current_dist = -pq.top().first;
        pq.pop();

        if (finalized[current_node] || current_dist > distances[current_node])
            { continue; }

        finalized[current_node] = true;

        if (current_node == end_node)
            { break; }

        // Check all neighbors
        for (int neighbor = 0; neighbor < num_nodes; ++neighbor) {
            if (connectivity_matrix(current_node, neighbor) == 1 && !finalized[neighbor]) {

                if (node_weights(neighbor) < -1000.0f) {
                    /* continue; */
                    /* new_distance = node_weights(neighbor); */
                    LOG("Distance penalty: " + std::to_string(new_distance) + " | " + std::to_string(neighbor));
                    continue;
                } else {

                    // Calculate Euclidean distance between nodes
                    float dx = node_coordinates(current_node, 0) - node_coordinates(neighbor, 0);
                    float dy = node_coordinates(current_node, 1) - node_coordinates(neighbor, 1);
                    float edge_distance = std::sqrt(dx*dx + dy*dy);

                    // Add optional node weight as a penalty/cost factor
                    /* float node_penalty = node_weights(neighbor); */

                    // New distance is current distance plus edge length and node penalty
                    new_distance = distances[current_node] + edge_distance;// - node_weights(neighbor);
                }

                if (new_distance < distances[neighbor]) {
                    distances[neighbor] = new_distance;
                    parent[neighbor] = current_node;
                    pq.push({-new_distance, neighbor});
                }
            }
        }
    }

    // Reconstruct the path
    std::vector<int> path;
    int current = end_node;

    if (distances[end_node] == std::numeric_limits<float>::infinity())
        { return {}; }

    while (current != -1) {
        path.push_back(current);
        current = parent[current];
    }
    std::reverse(path.begin(), path.end());

    if (path.empty() || path[0] != start_node)
        { return {}; }

    return path;
}


/* LINEAR ALGEBRA */


inline float cosine_similarity_vec(const Eigen::VectorXf& v1,
                                   const Eigen::VectorXf& v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm());
}


Eigen::VectorXf cosine_similarity_vector_matrix(
    const Eigen::VectorXf& vector,
    const Eigen::MatrixXf& matrix) {

    // Normalize the vector to unit norm
    Eigen::VectorXf normalized_vector = vector.normalized();

    // set NaN values to zero
    normalized_vector = (normalized_vector.array().isNaN()).select(
        Eigen::VectorXf::Zero(matrix.rows()), normalized_vector);

    // Normalize each row of the matrix to unit norm
    Eigen::MatrixXf normalized_matrix = matrix.rowwise().normalized();

    // set NaN values to zero
    normalized_matrix = (normalized_matrix.array().isNaN()).select(
        Eigen::MatrixXf::Zero(matrix.rows(), matrix.cols()), normalized_matrix);

    // Compute the cosine similarity
    Eigen::VectorXf similarity_vector = \
        normalized_matrix * normalized_vector;

    return similarity_vector;
}


Eigen::MatrixXf cosine_similarity_matrix(
    const Eigen::MatrixXf& matrix) {

    int n = matrix.rows();
    Eigen::MatrixXf similarity_matrix(n, n);

    // Normalize each row to unit norm
    Eigen::MatrixXf normalized_matrix = matrix.rowwise().normalized();

    // set NaN values to zero
    normalized_matrix = (normalized_matrix.array().isNaN()).select(
        Eigen::MatrixXf::Zero(n, n), normalized_matrix);

    // Compute the cosine similarity (normalized dot product)
    similarity_matrix = normalized_matrix * normalized_matrix.transpose();

    // take out the diagonal
    similarity_matrix.diagonal().setZero();

    return similarity_matrix;
}


float max_cosine_similarity_in_rows(
    const Eigen::MatrixXf& matrix, int idx) {
    // Compute the cosine similarity matrix
    Eigen::MatrixXf similarity_matrix = cosine_similarity_matrix(matrix);

    // Check that idx is within bounds
    if (idx < 0 || idx >= similarity_matrix.rows()) {
        throw std::out_of_range("Index is out of bounds.");
    }

    // Get the row at the specified index
    Eigen::VectorXf column = similarity_matrix.col(idx);

    // Find the maximum value in the column
    float max_similarity = column.maxCoeff();

    return max_similarity;
}


float euclidean_distance(const std::array<float, 2>& v1,
                         const std::array<float, 2>& v2) {
    return std::sqrt(std::pow(v1[0] - v2[0], 2) +
                     std::pow(v1[1] - v2[1], 2));
}


float gaussian_distance(const Eigen::VectorXf& v1,
                        const Eigen::VectorXf& v2,
                        float sigma) {
    // Calculate the squared Euclidean distance
    float squared_distance = (v1 - v2).squaredNorm();

    // Calculate the Gaussian distance
    float distance = std::exp(-squared_distance / (2 * sigma * sigma));

    return distance;
}


/* MISCELLANEOUS */


// Return type that contains both intersection status and coordinates
struct IntersectionResult {
    bool intersects;
    float x;
    float y;
};

// Alternative using std::tuple if preferred
using IntersectionTuple = std::tuple<bool, float, float>;

IntersectionResult get_segments_intersection(
    float p0_x, float p0_y,
    float p1_x, float p1_y,
    float p2_x, float p2_y,
    float p3_x, float p3_y)
{
    // Calculate segment vectors
    float s10_x = p1_x - p0_x;
    float s10_y = p1_y - p0_y;
    float s32_x = p3_x - p2_x;
    float s32_y = p3_y - p2_y;

    // Calculate denominator
    float denom = s10_x * s32_y - s32_x * s10_y;
    if (denom == 0) {
        return {false, 0.0f, 0.0f}; // Collinear
    }

    bool denomPositive = denom > 0;

    // Calculate vector between first points of each segment
    float s02_x = p0_x - p2_x;
    float s02_y = p0_y - p2_y;

    // Calculate numerators
    float s_numer = s10_x * s02_y - s10_y * s02_x;
    if ((s_numer < 0) == denomPositive) {
        return {false, 0.0f, 0.0f}; // No collision
    }

    float t_numer = s32_x * s02_y - s32_y * s02_x;
    if ((t_numer < 0) == denomPositive) {
        return {false, 0.0f, 0.0f}; // No collision
    }

    if (((s_numer > denom) == denomPositive) ||
        ((t_numer > denom) == denomPositive)) {
        return {false, 0.0f, 0.0f}; // No collision
    }

    // Collision detected, calculate intersection point
    float t = t_numer / denom;
    float intersection_x = p0_x + (t * s10_x);
    float intersection_y = p0_y + (t * s10_y);

    return {true, intersection_x, intersection_y};
}

// Alternative version using std::tuple if preferred
IntersectionTuple get_segments_intersection_tuple(
    float p0_x, float p0_y, float p1_x, float p1_y,
    float p2_x, float p2_y, float p3_x, float p3_y) {
    auto result = get_segments_intersection(p0_x, p0_y, p1_x, p1_y,
                                          p2_x, p2_y, p3_x, p3_y);
    return {result.intersects, result.x, result.y};
}


std::array<float, 2> reflect_point_over_segment(
                    float x, float y, float x1,
                    float y1, float x2, float y2) {

    if (x1 == x2) {
        return {2*x1 - x, y};
    } else if (y1 == y2) {
        return {x, 2*y1 - y};
    }

    /* printf("\nx1: %f, y1: %f, x2: %f, y2: %f\n", x1, y1, x2, y2); */
    /* printf("x: %f, y: %f\n", x, y); */

    // define line equation
    float m = (y2-y1) / (x2 - x1);
    float q = (x1-x2)*y1 + (y2-y1)*x1;

    /* printf("\nm: %f, q: %f\n", m, q); */

    // rotate x2,y2 around x1,y1 by 90 degrees
    float x3 = x2 - x1;
    float y3 = y2 - y1;
    float _x3 = -y3 + x1;
    float _y3 = x3 + y1;
    /* printf("_x3: %f, _y3: %f\n", _x3, _y3); */

    float _m = (_y3 - y1) / (_x3 - x1);

    // calculate q for perpendicular line
    // y - _m*x = q
    float _q = y - _m*x;
    /* printf("_m: %f, _q: %f\n", _m, _q); */

    // find intersection between line and perpendicular line
    // _m * x + _q = m * x + q
    // x = (q - _q) / (_m - m)
    float xc = (q - _q) / (_m - m);
    float yc = m * xc + q;

    // reflect x,y over xc,yc
    float xr = 2*xc - x;
    float yr = 2*yc - y;
    /* printf("x: %f, y: %f\n", x, y); */
    /* printf("m, _m: %f, %f\n", m, _m); */
    /* printf("q, _q: %f, %f\n", q, _q); */
    /* printf("xc: %f, yc: %f\n", xc, yc); */
    /* printf("xr: %f, yr: %f\n", xr, yr); */

    return {xr, yr};
}


Eigen::VectorXf linspace(float start, float end, int num)
{
    /* int num = num; // Ensure `num` is an integer */

    if (num <= 0) {
        return Eigen::VectorXf(); // Return an empty vector
    }

    if (num == 1) {
        Eigen::VectorXf linspaced(1);
        linspaced(0) = start;
        return linspaced;
    }

    Eigen::VectorXf linspaced(num);
    float delta = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        linspaced(i) = start + delta * i;
    }

    return linspaced;
}


std::vector<float> linspace_vec(float start, float end, int num,
                                bool startpoint,
                                bool endpoint) {
    if (num <= 0) {
        return std::vector<float>(); // Return an empty vector
    }

    if (num == 1) {
        return std::vector<float>{start};
    }

    if (!endpoint) {
        end = end - (end - start) / num;
    }
    if (!startpoint) {
        start = start + (end - start) / num;
    }

    std::vector<float> linspaced(num);
    float delta = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        linspaced[i] = start + delta * i;
    }

    return linspaced;
}


Eigen::MatrixXf connectivity_matrix(const Eigen::MatrixXf& matrix,
                                    float threshold) {

    // Compute the row to row similarity matrix
    Eigen::MatrixXf similarity_matrix = \
        cosine_similarity_matrix(matrix);

    // Threshold the similarity matrix
    Eigen::MatrixXf connectivity = (similarity_matrix.array() > threshold).cast<float>();

    return connectivity;
}


/* ACTIVATION FUNCTIONS */


float generalized_sigmoid(float x, float offset, float gain,
                          float clip) {
    // Offset the input by `offset`, apply the gain,
    // and then compute the sigmoid
    float result = 1.0f / (1.0f + std::exp(-gain * (x - offset)));

    return result >= clip ? result : 0.0f;
}


inline Eigen::VectorXf generalized_sigmoid_vec(const Eigen::VectorXf& x,
    float offset, float gain, float clip) {
    // Offset each element by `offset`, apply the gain,
    // and then compute the sigmoid
    Eigen::VectorXf result = 1.0f / (1.0f + \
        (-gain * (x.array() - offset)).exp());

    return (result.array() >= clip).select(result, 0.0f);
}


inline std::vector<float> generalized_sigmoid_vec(const std::vector<float>& x,
                        float offset, float gain, float clip) {

    // Offset each element by `offset`, apply the gain,
    // and then compute the sigmoid
    std::vector<float> result;
    for (size_t i = 0; i < x.size(); i++) {
        float val = 1.0f / (1.0f + \
            exp(-gain * (x[i] - offset)));
        result.push_back(val >= clip ? val : 0.0f);
    }

    return result;
}

