/* #ifndef PCNN_HPP */
/* #define PCNN_HPP */

/* #include "utils.hpp" */

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
#define NUM_GCL 9
#define CIRCUIT_SIZE 2

/* ========================================== */

// blank log function
void LOG(const std::string& msg) {
    /* return; */
    std::cout << msg << std::endl;
}


int SEED = 0;


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
                                       const Eigen::MatrixXf& node_coordinates,  // Nx2 matrix with (x,y) coordinates
                                       const Eigen::VectorXf& node_weights,      // Optional node penalties/costs
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
                    /* LOG("Distance: " + std::to_string(new_distance) + " | nw: " + std::to_string(node_weights(neighbor))); */
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
                        float sigma = 1.0f) {
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
    float p0_x, float p0_y,
    float p1_x, float p1_y,
    float p2_x, float p2_y,
    float p3_x, float p3_y)
{
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
                                bool startpoint = true,
                                bool endpoint = true)
{
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


Eigen::MatrixXf connectivity_matrix(
    const Eigen::MatrixXf& matrix,
    float threshold = 0.5f) {

    // Compute the row to row similarity matrix
    Eigen::MatrixXf similarity_matrix = \
        cosine_similarity_matrix(matrix);

    // Threshold the similarity matrix
    Eigen::MatrixXf connectivity = (similarity_matrix.array() > threshold).cast<float>();

    return connectivity;
}


/* ACTIVATION FUNCTIONS */


float generalized_sigmoid(float x, float offset = 1.0f,
                          float gain = 1.0f, float clip = 0.0f) {
    // Offset the input by `offset`, apply the gain,
    // and then compute the sigmoid
    float result = 1.0f / (1.0f + std::exp(-gain * (x - offset)));

    return result >= clip ? result : 0.0f;
}


inline Eigen::VectorXf generalized_sigmoid_vec(
    const Eigen::VectorXf& x,
    float offset = 1.0f,
    float gain = 1.0f,
    float clip = 0.0f) {
    // Offset each element by `offset`, apply the gain,
    // and then compute the sigmoid
    Eigen::VectorXf result = 1.0f / (1.0f + \
        (-gain * (x.array() - offset)).exp());

    return (result.array() >= clip).select(result, 0.0f);
}


inline std::vector<float> generalized_sigmoid_vec(
    const std::vector<float>& x, float offset = 1.0f,
    float gain = 1.0f, float clip = 0.0f) {

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


/* ========================================== */
/* ================= SPACE  ================= */
/* ========================================== */


class GridLayer {

    // parameters
    float sigma;
    float speed;
    std::array<float, 4> bounds;  // it's square

    // variables
    std::array<std::array<float, 2>, GCL_SIZE> basis;
    std::array<std::array<float, 2>, GCL_SIZE> positions;
    std::array<float, GCL_SIZE> y;

    // define basis type | maybe constexpr ??
    void square_basis() {


        float dx = 1.0f / (static_cast<float>(GCL_SIZE_SQRT) + 0.0f);

        // define the centers as a grid:
        // - excluding the endpoints
        // - assuming bounds (-1, 1), (-1, 1)
        std::vector<float> linex = linspace_vec(
                        bounds[0], bounds[1], GCL_SIZE_SQRT,
                        true, false);
        std::vector<float> liney = linspace_vec(
                        bounds[2], bounds[3], GCL_SIZE_SQRT,
                        true, false);

        for (std::size_t i=0; i<GCL_SIZE; i++) {
            float xi = linex[i / GCL_SIZE_SQRT];
            float yi = liney[i % GCL_SIZE_SQRT];
            basis[i][0] = xi;
            basis[i][1] = yi;
        }
    }

    // define boundary type
    void boundary_conditions(std::array<std::array<float, 2>,
                             GCL_SIZE>& _positions) {
        for (int i = 0; i < GCL_SIZE; i++) {
            std::array<float, 2> new_position = \
                apply_boundary(_positions[i][0],
                               _positions[i][1]);
            _positions[i][0] = new_position[0];
            _positions[i][1] = new_position[1];
        }
    }

    // define boundary type
    void boundary_conditions() {
        for (int i = 0; i < GCL_SIZE; i++) {
            std::array<float, 2> new_position = \
                apply_boundary(positions[i][0],
                               positions[i][1]);
            positions[i][0] = new_position[0];
            positions[i][1] = new_position[1];
        }
    }

    // gaussian distance of each position to the centers
    void calc_activation() {
        float dist_squared;
        for (int i = 0; i < GCL_SIZE; i++) {
            dist_squared = std::pow(positions[i][0], 2) + \
                std::pow(positions[i][1], 2);
            y[i] = std::exp(-dist_squared / sigma);
        }
    }

    std::array<float, 2> apply_boundary(float x, float y) {
        if (x < bounds[0]) { x += 2.0f*std::abs(bounds[0]); }
        else if (x > bounds[1]) { x -= 2.0f*bounds[1]; }
        if (y < bounds[2]) { y += 2.0f*std::abs(bounds[2]); }
        else if (y > bounds[3]) { y -= 2.0f*bounds[3]; }

        return {x, y};
    }

public:

    GridLayer(float sigma, float speed,
                std::array<float, 4> bounds = \
                    {-1.0, 1.0, -1.0, 1.0}):
        sigma(sigma), speed(speed), bounds(bounds) {

        // record positions
        square_basis();

        // record initial positions in the basis
        // pass by value
        positions = basis;
    }

    // CALL
    std::array<float, GCL_SIZE> call(std::array<float, 2> v) {

        // update position with velociy
        for (int i = 0; i < GCL_SIZE; i++) {
            positions[i][0] += speed * v[0];
            positions[i][1] += speed * v[1];
        }

        // apply boundary conditions
        boundary_conditions();

        // compute the activation
        calc_activation();
        return y;
    }

    // SIMULATE
    std::array<float, GCL_SIZE> simulate_one_step(
        std::array<float, 2>& v) {

        std::array<std::array<float, 2>, GCL_SIZE> new_positions;
        for (int i = 0; i < GCL_SIZE; i++) {
            new_positions[i][0] = positions[i][0] + speed * v[0];
            new_positions[i][1] = positions[i][1] + speed * v[1];
        }

        boundary_conditions(new_positions);

        // compute the activation
        std::array<float, GCL_SIZE> yfwd = {0.0f};
        float dist_squared;
        for (int i = 0; i < GCL_SIZE; i++) {
            dist_squared = std::pow(new_positions[i][0],
                                    2) + \
                std::pow(new_positions[i][1], 2);
            yfwd[i] = std::exp(-dist_squared / sigma);
        }

        return yfwd;
    }

    int len() { return GCL_SIZE; }
    std::string str() { return "GridLayer"; }
    std::string repr() { return "GridLayer"; }
    std::array<std::array<float, 2>, GCL_SIZE> \
        get_positions() { return positions; }
    std::array<std::array<float, 2>, GCL_SIZE> \
        get_centers() { return basis; }
    std::array<float, GCL_SIZE> \
        get_activation() { return y; }
    void reset(std::array<float, 2> v) {
        this->positions = basis;
        call(v);
    }


};


class GridNetwork {

    std::vector<GridLayer> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;

public:

    GridNetwork(std::vector<GridLayer> layers)
        : layers(layers) {

        // Initialize the variables
        this->num_layers = layers.size();

        int total_size = 0;
        full_repr = "(";
        for (auto& layer : layers) {
                total_size += layer.len();
                full_repr += layer.repr();
            }
        full_repr += ")";
        this->N = total_size;
        y = Eigen::VectorXf::Zero(N);
        basis = Eigen::MatrixXf::Zero(total_size, 2);
    }

    // CALL
    Eigen::VectorXf call(const std::array<float, 2> x) {

        for (int i = 0; i < num_layers; i++) {
            // Convert the output of layers[i].call(x) to
            // an Eigen::VectorXf
            Eigen::VectorXf layer_output = \
                Eigen::Map<const Eigen::VectorXf>(
                    layers[i].call(x).data(), GCL_SIZE);

            // Assign the converted vector to
            // the corresponding segment of y
            y.segment(i * GCL_SIZE, GCL_SIZE) = layer_output;
        }

        return y;
    }

    Eigen::VectorXf simulate_one_step(std::array<float, 2> v) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {

            // Convert the output of layers[i].call(x) to
            // an Eigen::VectorXf
            Eigen::VectorXf layer_output = \
                Eigen::Map<const Eigen::VectorXf>(
                    layers[i].simulate_one_step(v).data(), GCL_SIZE);

            // Assign the converted vector to
            // the corresponding segment of y
            yfwd.segment(i * GCL_SIZE, GCL_SIZE) = layer_output;
        }

        return yfwd;
    }

    int len() { return N; }
    int get_num_layers() { return num_layers; }
    std::string str()  { return "GridNetwork"; }
    std::string repr() {
        return str() + "(" + full_repr + ", N=" + \
        std::to_string(N) + ")"; }
    Eigen::VectorXf get_activation() const { return y; }
    Eigen::MatrixXf get_centers() {
        for (int i = 0; i < num_layers; i++) {
            // Get the positions vector
            const auto& positions = layers[i].get_positions();

            // Create matrix directly and fill it
            Eigen::MatrixXf layer_positions(positions.size(), 2);
            for (size_t j = 0; j < positions.size(); j++) {
                layer_positions(j, 0) = positions[j][0];
                layer_positions(j, 1) = positions[j][1];
            }

            // Assign to basis
            basis.block(i * layers[i].len(), 0,
                        layers[i].len(), 2) = layer_positions;
        }
        return basis;
    }

    Eigen::MatrixXf get_positions() {
        for (int i = 0; i < num_layers; i++) {
            // Get the positions vector
            const auto& positions = layers[i].get_positions();

            // Create matrix directly and fill it
            Eigen::MatrixXf layer_positions(positions.size(), 2);
            for (size_t j = 0; j < positions.size(); j++) {
                layer_positions(j, 0) = positions[j][0];
                layer_positions(j, 1) = positions[j][1];
            }

            // Assign to basis
            basis.block(i * layers[i].len(), 0,
                        layers[i].len(), 2) = layer_positions;
        }
        return basis;
    }

    std::vector<std::array<std::array<float, 2>, GCL_SIZE>> get_positions_vec() {
        std::vector<std::array<std::array<float, 2>, GCL_SIZE>> positions_vec;
        for (int i = 0; i < num_layers; i++) {
            positions_vec.push_back(layers[i].get_positions());
        }
        return positions_vec;
    }

    void reset(std::array<float, 2> v) {
        for (int i = 0; i < num_layers; i++) {
            layers[i].reset(v);
        }
    }
};


/* === PCNN === */


struct VelocitySpace {

    void update_node_degree(int idx, int current_size) {

        // update the node itself
        node_degrees(idx) = connectivity.row(idx).sum();

        // update the neighbors
        int degree = 0;
        for (int j = 0; j < current_size; j++) {
            if (connectivity(idx, j) == 1.0) {
                node_degrees(j) = connectivity.row(j).sum();
            }
        }
    }

public:

    Eigen::MatrixXf centers;
    Eigen::MatrixXf centers_original;
    Eigen::MatrixXf connectivity;
    Eigen::MatrixXf weights;
    Eigen::VectorXf nodes_max_angle;
    Eigen::VectorXf node_degrees;
    Eigen::VectorXf one_trace;
    std::array<float, 2> position;
    int size;
    float threshold;

    std::vector<std::array<int, 2>> blocked_edges;

    VelocitySpace(int size, float threshold)
        : size(size), threshold(threshold) {
        centers = Eigen::MatrixXf::Constant(size, 2, -9999.0f);
        centers_original = Eigen::MatrixXf::Constant(size, 2, -9999.0f);
        connectivity = Eigen::MatrixXf::Zero(size, size);
        weights = Eigen::MatrixXf::Zero(size, size);
        one_trace = Eigen::VectorXf::Ones(size);
        position = {0.00124789f, 0.00147891f};
        blocked_edges = {};
        nodes_max_angle = Eigen::VectorXf::Zero(size);
        node_degrees = Eigen::VectorXf::Zero(size);
    }

    // CALL
    std::array<float, 2> call(const std::array<float, 2> v) {
        position[0] += v[0];
        position[1] += v[1];
        return {position[0], position[1]};
    }

    void update(int idx, Eigen::VectorXf& traces,
                bool update_center = true) {

        // update the centers
        if (update_center) {
            centers.row(idx) = Eigen::Vector2f(
                position[0], position[1]);

            if (centers_original(idx, 0) < -700.0f) {
                centers_original.row(idx) = Eigen::Vector2f(
                    position[0], position[1]);
            }
        }

        // add recurrent connections based on nearest neighbors
        for (int idx = 0; idx < size; idx++) {

            // Skip invalid nodes
            if (centers(idx, 0) < -700.0f) { continue; }

            // Create a vector of pairs (distance, index)
            // for all valid nodes
            std::vector<std::pair<float, int>> distances;
            for (int j = 0; j < size; j++) {
                // Skip invalid nodes and self
                if (idx == j || centers(j, 0) < -600.0f ||
                    centers(j, 1) < -600 || centers(j, 0) > 600 ||
                    centers (j, 1) > 600 ) { continue; }

                // check if neighbors was active
                if (traces(j) < 0.1f) { continue; }

                float dist = std::sqrt(
                    (centers(idx, 0) - centers(j, 0)) *
                        (centers(idx, 0) - centers(j, 0)) +
                    (centers(idx, 1) - centers(j, 1)) *
                        (centers(idx, 1) - centers(j, 1))
                );

                if (dist > threshold) { continue; }

                distances.push_back(std::make_pair(dist, j));

            }

            // Sort by distance (ascending)
            std::sort(distances.begin(), distances.end());

            // Connect to the closest max_neighbors neighbors
            int num_connections = static_cast<int>(distances.size());

            for (int k = 0; k < num_connections; k++) {
                int j = distances[k].second;
                float dist = distances[k].first;

                // Establish bidirectional connection
                this->weights(idx, j) = dist;
                this->weights(j, idx) = dist;
                this->connectivity(idx, j) = 1.0;
                this->connectivity(j, idx) = 1.0;
            }
        }

        // update the node angles
        update_node_degree(idx, size);
    }

    void remap_center(int idx, std::array<float, 2> displacement) {

        // check: over bound <<< ad hoc measure
        if (centers(idx, 0) + displacement[0] > 400 ||
            centers(idx, 1) + displacement[1] > 400 ||
            centers(idx, 0) + displacement[0] < -400 ||
            centers(idx, 1) + displacement[1] < -400) {
            return;
        }
        centers(idx, 0) += displacement[0];
        centers(idx, 1) += displacement[1];

        update(idx, one_trace, false);
    }

    Eigen::MatrixXf get_centers(bool nonzero=false) {

        if (!nonzero) { return centers; }

        Eigen::MatrixXf centers_nonzero = Eigen::MatrixXf::Zero(size, 2);
        for (int i = 0; i < size; i++) {
            if (centers(i, 0) > -999.0f) {
                centers_nonzero.row(i) = centers.row(i);
            }
        }
        return centers_nonzero;
    }

    Eigen::MatrixXf get_centers_original(bool nonzero=false) {

        if (!nonzero) { return centers_original; }

        Eigen::MatrixXf centers_nonzero = Eigen::MatrixXf::Zero(size, 2);
        for (int i = 0; i < size; i++) {
            if (centers_original(i, 0) > -999.0f) {
                centers_nonzero.row(i) = centers_original.row(i);
            }
        }
        return centers_nonzero;
    }

    std::vector<std::array<std::array<float, 2>, 2>> make_edges() {
        // make a list of edges from the connectivity matrix
        std::vector<std::array<std::array<float, 2>, 2>> edges;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (connectivity(i, j) == 1.0) {
                    edges.push_back({std::array<float, 2>{centers(i, 0),
                                                          centers(i, 1)},
                                     std::array<float, 2>{centers(j, 0),
                                                          centers(j, 1)}});
                }
            }
        }
        return edges;
    }

    std::vector<std::array<std::array<float, 2>, 3>> make_edges_value(
               Eigen::MatrixXf& values) {
        // make a list of edges from the connectivity matrix
        std::vector<std::array<std::array<float, 2>, 3>> edges;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (connectivity(i, j) == 1.0) {
                    edges.push_back({std::array<float, 2>{centers(i, 0),
                                                          centers(i, 1)},
                                     std::array<float, 2>{centers(j, 0),
                                                          centers(j, 1)},
                                     std::array<float, 2>{values(i, j),
                                                          values(j, i)}});
                }
            }
        }
        return edges;  // Add this line to return the edges vector
    }

    int calculate_closest_index(std::array<float, 2> c) {
        // calculate the closest index to the velocity
        float min_dist = 1000.0f;
        int idx = -1;
        for (int i = 0; i < size; i++) {
            if (centers(i, 0) < -999.0f) {
                continue;
            }
            float dist = std::sqrt(
                (c[0] - centers(i, 0)) * (c[0] - centers(i, 0)) +
                (c[1] - centers(i, 1)) * (c[1] - centers(i, 1))
            );
            if (dist < min_dist) {
                min_dist = dist;
                idx = i;
            }
        }
        return idx;
    }

    void delete_node(int idx) {

        // delete the node by shifting the all next nodes
        for (int i = idx; i < size-1; i++) {
            centers.row(i) = centers.row(i+1);
            connectivity.row(i) = connectivity.row(i+1);
            weights.row(i) = weights.row(i+1);
            nodes_max_angle(i) = nodes_max_angle(i+1);
            node_degrees(i) = node_degrees(i+1);
        }
    }

    void delete_edge(int i, int j) {
        connectivity(i, j) = 0.0f;
        connectivity(j, i) = 0.0f;
        weights(i, j) = 0.0f;
        weights(j, i) = 0.0f;
    }

    bool check_edge(int i, int j) {
        return connectivity(i, j) == 1.0f;
    }

    bool is_too_close(int idx, std::array<float, 2> velocity,
                      float rec_threshold = 0.1f) {

        // check if the center at idx can be moved:
        // the distance from the other centers should be greater than 0.1
        std::array<float, 2> c = {centers(idx, 0), centers(idx, 1)};

        float mind = 1000.0f;
        for (int i = 0; i < size; i++) {
            if (i == idx) { continue; }
            if (centers(i, 0) < -999.0f) { continue; }

            float dist = std::sqrt(
                (c[0] - centers(i, 0)) * (c[0] - centers(i, 0)) +
                (c[1] - centers(i, 1)) * (c[1] - centers(i, 1))
            );

            if (dist < rec_threshold) {
                /* std::cout << "too close:)" << std::endl; */
                return true;
            }
            if (dist < mind) {
                mind = dist;
            }
        }
        /* std::cout << "not too close:)" << mind << std::endl; */
        return false;
    }

};


class PCNN {

    // internal components
    VelocitySpace vspace;
    GridNetwork xfilter;

    // constant parameters
    const int N;
    const int Nj;
    const float offset;
    const float clip_min;
    const std::string name;
    const float threshold_const;
    const float rep_threshold_const;
    const float gain_const;
    const float tau_trace;
    const float tau_trace_v2 = 1000.0f;

    // ~activation
    float rep_threshold;
    float min_rep_threshold;
    float threshold;
    /* float gain; */

    // remappint
    const int remap_tag_frequency;
    int t_update = 0;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    Eigen::MatrixXf Wff_original;
    float delta_wff;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf connectivity;
    Eigen::VectorXf mask;
    std::vector<int> fixed_indexes;
    std::vector<int> free_indexes;
    std::vector<int> _indexes;
    int cell_count;
    Eigen::VectorXf u;
    Eigen::VectorXf traces;
    Eigen::VectorXf traces_v2;
    Eigen::VectorXf gain_v;
    std::vector<bool> remap_tag = {};
    Eigen::VectorXf x_filtered;

    // @brief check if one of the fixed neurons
    int check_fixed_indexes() {

        // if there are no fixed neurons, return -1
        if (fixed_indexes.size() == 0) {
            return -1;
        };

        // loop through the fixed indexes's `u` value
        // and return the index with the highest value
        int max_idx = -1;
        float max_u = 0.0;
        for (int i = 0; i < fixed_indexes.size(); i++) {
            if (u(fixed_indexes[i]) > max_u) {
                max_u = u(fixed_indexes[i]);
                max_idx = fixed_indexes[i];
            };
        };

        return max_idx;
    }

    // @brief Quantify the indexes.
    void make_indexes() {
        free_indexes = {};
        for (int i = 0; i < N; i++) {
            if (Wff.row(i).sum() > 0.0f) { fixed_indexes.push_back(i); }
            else { free_indexes.push_back(i); }
        }
    }

public:
    const float rec_threshold;

    PCNN(int N, int Nj, float gain, float offset,
         float clip_min, float threshold, float rep_threshold,
         float rec_threshold, float min_rep_threshold,
         GridNetwork xfilter, float tau_trace = 2.0f,
         int remap_tag_frequency = 1, std::string name = "fine")
        : N(N), Nj(Nj), gain_const(gain), //gain(gain),
        offset(offset), clip_min(clip_min),
        min_rep_threshold(min_rep_threshold), rep_threshold(rep_threshold),
        rep_threshold_const(rep_threshold), rec_threshold(rec_threshold),
        threshold_const(threshold), threshold(threshold),
        xfilter(xfilter), tau_trace(tau_trace), name(name),
        remap_tag_frequency(remap_tag_frequency),
        vspace(VelocitySpace(N, rec_threshold)) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wff_original = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        traces = Eigen::VectorXf::Zero(N);
        traces_v2 = Eigen::VectorXf::Zero(N);
        gain_v = Eigen::VectorXf::Zero(N);
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            gain_v(i) = gain_const;
            free_indexes.push_back(i);
            _indexes.push_back(i);
        }
        fixed_indexes = {};

        /* std::cout << "gain " << gain << ", offset " << */
        /*     offset << "thr" << threshold << "\n"; */

    }

    // CALL
    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(std::array<float, 2> v) {

        vspace.call(v);

        // pass the input through the filter layer
        x_filtered = xfilter.call(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        /* u = Wff * x_filtered; */
        u = cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(),
                                                          0.00001);


        for (int i = 0; i < N; i++) {
            u(i) = generalized_sigmoid(u(i), offset, gain_v(i), clip_min);
        }
        /* u = generalized_sigmoid_vec(u, offset, gain, clip_min); */

        traces = traces - traces / tau_trace + u;
        traces = traces.cwiseMin(1.0f);

        traces_v2 = traces_v2 - traces_v2 / tau_trace_v2 + u;
        traces_v2 = traces_v2.cwiseMin(1.0f);

        return std::make_pair(u, x_filtered);
    }

    // UPDATE
    void update() {

        make_indexes();

        // exit: a fixed neuron is above threshold
        if (check_fixed_indexes() != -1) { return void(); }

        // exit: there are no free neurons
        if (cell_count == N) { return void(); }

        // pick new index
        // int _count = cell_count == 0 ? 0 : cell_count / 2;
        // int idx = free_indexes[cell_count];
        int idx = _indexes[cell_count+1];

        // determine weight update | <<<< previously
        Eigen::VectorXf dw = x_filtered - Wff.row(idx).transpose();

        // trim the weight update | <<<< previously
        delta_wff = dw.norm();

        if (delta_wff > 0.0) {

            // update weights | <<<< previously
            Wff.row(idx) += dw.transpose();

            // calculate the similarity among the rows
            float similarity = \
                max_cosine_similarity_in_rows(
                    Wff, idx);

            // check repulsion (similarity) level
            if (similarity > rep_threshold) {
                Wff.row(idx) = Wffbackup.row(idx);
                return void();
            }

            // update count and backup
            if (idx > cell_count) { cell_count++; }
            // cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);
            // Wff_original.row(idx) = Wff.row(idx);

            // record new center
            vspace.update(idx, traces);
            this->Wrec = vspace.weights;
            this->connectivity = vspace.connectivity;

            // update the remap tags
            remap_tag.push_back(t_update % remap_tag_frequency == 0);
            t_update++;
        }
    }

    Eigen::VectorXf& simulate_one_step(const std::array<float, 2>& v) {

        // pass the input through the filter layer
        x_filtered = xfilter.simulate_one_step(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        /* u = generalized_sigmoid_vec(u, offset, */
        /*                                    gain, clip_min); */
        for (int i = 0; i < N; i++) {
            u(i) = generalized_sigmoid(u(i), offset, gain_v(i), clip_min);
        }
        return u;
    }

    void remap(Eigen::VectorXf& block_weights,
               std::array<float, 2> velocity,
               float width, float magnitude,
               float threshold) {

        if ((magnitude * magnitude) < 0.00001f) { return; }

        float magnitude_i;;
        for (int i = 0; i < N; i++) {

            // check remap tag
            // if (!remap_tag[i] || traces(i) < 0.1) { continue; }
            if (traces(i) < threshold) { continue; }

            // skip blocked edges
            if (vspace.centers(i, 0) < -900.0f)// || block_weights(i) > 0.0f)
                { continue; }

            std::array<float, 2> displacement = \
                {vspace.position[0] - vspace.centers(i, 0),
                 vspace.position[1] - vspace.centers(i, 1)};

            // gaussian activation function centered at zero

            if (vspace.is_too_close(i, {displacement[0] * magnitude,
                                        displacement[1] * magnitude},
                                    min_rep_threshold)) { continue; }

            // weight the displacement
            std::array<float, 2> gc_displacement = {
                                displacement[0] * magnitude,
                                displacement[1] * magnitude};

            // pass the input through the filter layer
            x_filtered = xfilter.simulate_one_step(gc_displacement);

            // update the weights & centers
            Wff.row(i) = x_filtered.transpose();

            Wffbackup.row(i) = Wff.row(i);
            vspace.remap_center(i, gc_displacement);
        }
    }

    int calculate_closest_index(const std::array<float, 2>& c)
        { return vspace.calculate_closest_index(c); }

    int get_neighbourhood_node_degree(int idx) {
        int node_degree = vspace.node_degrees(idx);
        int count = 1;

        for (int i = 0; i < N; i++) {
            if (vspace.connectivity(idx, i) > 0.0f) {
                node_degree += vspace.node_degrees(i);
                count++;
            }
        }
        return node_degree / count;
    }

    void modulate_gain(float modulation) {
        for (int i = 0; i < N; i++) {
            if (u(i) > 0.1) {
                gain_v(i) = modulation * gain_const * traces(i) + \
                    (1 - traces(i)) * gain_const;

                // mimumum value of 5.0f
                gain_v(i) = std::max(gain_v(i), 5.0f);
            }
        }
    }

    void update_upon_collision() {

        // check highest activation
        if (u.maxCoeff() < threshold) { return; }

        // get argmax
        Eigen::Index maxIndex;
        float maxValue = u.maxCoeff(&maxIndex);
        int idx = static_cast<int>(maxIndex);

        // delete node
        vspace.delete_node(idx);
        this->Wrec = vspace.weights;
        this->connectivity = vspace.connectivity;

        // update variables
        /* cell_count--; */
        /* fixed_indexes.erase(std::remove(fixed_indexes.begin(), */
        /*                                 fixed_indexes.end(), idx), */
        /*                     fixed_indexes.end()); */
        /* free_indexes.push_back(idx); */

    }

    void delete_edge(int i, int j) {
        vspace.delete_edge(i, j);
        this->Wrec = vspace.weights;
        this->connectivity = vspace.connectivity;
    }

    bool check_edge(int i, int j) {
        return vspace.check_edge(i, j);
    }

    void reset() {
        u = Eigen::VectorXf::Zero(N);
        traces = Eigen::VectorXf::Zero(N);
    }

    // Getters
    int len() { return cell_count; }
    int get_size() { return N; }
    std::string str() { return "PCNN." + name; }
    std::string repr() {
        return "PCNN(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain_const) + \
            std::to_string(offset) + \
            std::to_string(rec_threshold) + ")";
    }
    Eigen::VectorXf& get_activation() { return u; }
    Eigen::VectorXf get_activation_gcn()
        { return xfilter.get_activation(); }
    Eigen::MatrixXf& get_wff() { return Wff; }
    Eigen::MatrixXf& get_wff_original() { return Wff_original; }
    Eigen::MatrixXf& get_wrec() { return Wrec; }
    float get_trace_value(int idx) { return traces(idx); }
    Eigen::VectorXf get_trace_v2() { return traces_v2; }
    float get_max_activation() { return u.maxCoeff(); }
    std::vector<std::array<std::array<float, 2>, 2>> make_edges()
        { return vspace.make_edges(); }
    std::vector<std::array<std::array<float, 2>, 3>> make_edges_value(
        Eigen::MatrixXf& values) { return vspace.make_edges_value(values); }
    Eigen::MatrixXf& get_connectivity() { return connectivity; }
    Eigen::MatrixXf get_centers(bool nonzero = false)
        { return vspace.get_centers(nonzero); }
    Eigen::MatrixXf get_centers_original(bool nonzero = false)
        { return vspace.get_centers_original(nonzero); }
    Eigen::VectorXf& get_node_degrees() { return vspace.node_degrees; }
    Eigen::VectorXf& get_gain() { return gain_v; }
    float get_delta_update() { return delta_wff; }
    Eigen::MatrixXf get_positions_gcn()
        { return xfilter.get_positions(); }
    std::array<float, 2> get_position()
        { return vspace.position; }
    /* std::vector<std::array<std::array<float, 2>, GCL_SIZE>> get_gc_positions_vec() */
    /*     { return xfilter.get_positions_vec(); } */
    void reset_gcn(std::array<float, 2> v) { xfilter.reset(v); }

};


/* ========================================== */
/* ============== MODULATION ================ */
/* ========================================== */


class LeakyVariable1D {
public:
    std::string name;

    // CALL
    float call(float x = 0.0,
               bool simulate = false) {

        // simulate
        if (simulate) {
            float z = v + (eq - v) * tau + x;
            if (z < min_v) { z = min_v; }
            return z;
        }

        v = v + (eq - v) * tau + x;

        if (v < min_v) { v = 0.0f; }
        return v;
    }

    LeakyVariable1D(std::string name, float eq,
                    float tau, float min_v = 0.0)
        : name(std::move(name)), eq(eq), tau(1.0/tau),
        v(eq), min_v(min_v){}

    float get_v() { return v; }
    std::string str() { return "LeakyVariable." + name; }
    std::string repr() {
        return "LeakyVariable." + name + "(eq=" + \
            std::to_string(eq) + ", tau=" + std::to_string(tau) + ")";
    }
    int len() const { return 1; }
    std::string get_name() { return name; }
    void set_eq(float eq) { this->eq = eq; }
    void reset() { v = eq; }

private:
    const float min_v;
    const float tau;
    float v;
    float eq;
};


class BaseModulation{

    // parameters
    LeakyVariable1D leaky;
    std::string name;
    int size;
    float lr;
    float lt_p = 1.0f;
    float threshold;
    float mask_threshold;

    // variables
    Eigen::VectorXf weights;
    Eigen::VectorXf mask;
    float output;

    // prediction
    Eigen::VectorXf prediction;
    float lr_pred;

public:
    float max_w;

    BaseModulation(std::string name, int size,
                   float lr, float lr_pred, float threshold, float max_w = 1.0f,
                   float tau_v = 1.0f, float eq_v = 1.0f,
                   float min_v = 0.0f, float mask_threshold = 0.01f):
        name(name), size(size), lr(lr), threshold(threshold),
        max_w(max_w), mask_threshold(mask_threshold),
        leaky(LeakyVariable1D(name, eq_v, tau_v, min_v)),
        lr_pred(lr_pred) {
        weights = Eigen::VectorXf::Zero(size);
        mask = Eigen::VectorXf::Zero(size);
        prediction = Eigen::VectorXf::Zero(size);
    }

    // CALL
    float call(const Eigen::VectorXf& u,
               float x = 0.0f, bool simulate = false) {

        // forward to the leaky variable
        float v = leaky.call(x, simulate);

        // update the weights | assumption: u and uc have the same size
        if (!simulate && v > 0.00001f) {
            float maxval = 0.0f;
            for (int i = 0; i < size; i++) {

                // forward, delta from current input
                float ui = u[i] > threshold ? u[i] : 0.0;
                /* float dw = lr * v * ui; */
                if (maxval < ui) {maxval = ui; }

                // backward, prediction error
                float pred_err = 0.0f;

                if (name == "DA") {
                    pred_err = lr_pred * (prediction[i] - v * ui);
                    if (pred_err < 0.0) { pred_err = 0.0; }

                    // clip the prediction error if within a certain range
                    pred_err = pred_err > 0.001f && pred_err < 0.01 ? 0.01 : pred_err;
                }

                if (pred_err > 0.01 && name == "DA") {
                    /* LOG("[+] prediction error: " + std::to_string(pred_err)); */
                }

                // update weights
                // weights[i] += lr * v * ui - pred_err;
                weights[i] += lr * ui * (v - weights[i]);

                // clip the weights in (0, max_w)
                if (weights[i] < 0.01) { weights[i] = 0.0; }
                else if (weights[i] > max_w) { weights[i] = max_w; }
            }

            // highest da value
            if (name == "DA") {
                // get the max value and index (target index
                Eigen::Index maxIndex;
                float maxValue = weights.maxCoeff(&maxIndex);
                int trg_idx = static_cast<int>(maxIndex);
                /* LOG("[+] update " + name + " target: [" + */
                /*     std::to_string(trg_idx) + "] " + std::to_string(maxValue)  + */
                /*     ", max ui: " + std::to_string(maxval)); */
            }
        }

        // compute the output
        output = 0.0f;
        for (int i = 0; i < size; i++) { output += weights[i] * u[i]; }
        return output;
    }

    void make_prediction(const Eigen::VectorXf& u) {
        int idx = -1;
        float value = 0.0f;
        for (int i = 0; i < size; i++) {
            prediction[i] = weights[i] * u[i];
            if (prediction[i] > value) {
                value = prediction[i];
                idx = i;
            }
        }
    }

    float get_output() { return output; }

    Eigen::VectorXf& get_weights() { return weights; }
    Eigen::VectorXf& make_mask() {

        // the nodes that are fresh in memory will affect action selection,
        // thus acting as a mask
        for (int i = 0; i < size; i++) {
            this->mask(i) = weights(i) > mask_threshold ? 0.0f : 1.0f;
        }
        return mask;
    }
    float get_leaky_v() { return leaky.get_v(); }
    std::string str() { return name; }
    int len() { return size; }
    float get_modulation_value(int idx) { return weights(idx); }
    void reset() { leaky.reset(); }
};


struct StationarySensory {

    Eigen::VectorXf prev_representation;
    float v = 0.0f;
    float tau;
    float threshold;
    float min_cosine = 0.5f;

    bool call(Eigen::VectorXf& representation) {

        float cosim = cosine_similarity_vec(representation, prev_representation);

        // if cosine similarity is not nan
        if (std::isnan(cosim) || cosim < min_cosine) { cosim = 0.0f; }
        v += (cosim - v) / tau;
        prev_representation = representation;
        return v > threshold;
    }

    StationarySensory(int size, float tau,
                      float threshold = 0.2,
                      float min_cosine = 0.5):
        tau(tau), threshold(threshold),
        prev_representation(Eigen::VectorXf::Zero(size)) {}

    std::string str() { return "StationarySensory"; }
    std::string repr() { return "StationarySensory"; }
    float get_v() { return v; }
};


class Circuits {

    // external components
    BaseModulation& da;
    BaseModulation& bnd;

    // parameters
    int space_size;
    float threshold;

    // variables
    std::array<float, CIRCUIT_SIZE> output;
    Eigen::VectorXf value_mask;

public:

    Circuits(BaseModulation& da, BaseModulation& bnd, float threshold):
        da(da), bnd(bnd), value_mask(Eigen::VectorXf::Ones(da.len())),
        space_size(da.len()), threshold(threshold) {
        /* LOG("[+] Circuits created, threshold=" + std::to_string(threshold)); */
    }

    // CALL
    std::array<float, CIRCUIT_SIZE> call(Eigen::VectorXf& representation,
            float collision, float reward, bool simulate = false) {

        output[0] = bnd.call(representation,
                             collision, simulate);
        output[1] = da.call(representation,
                            reward, simulate);

        return output;
    }

    Eigen::VectorXf& make_value_mask(bool strict = false) {

        Eigen::VectorXf& bnd_weights = bnd.get_weights();

        for (int i = 0; i < space_size; i++) {
            if (get_bnd_value(i) < 0.001f) { value_mask(i) = 1.0f; }
            else { value_mask(i) = -10000.0f; }
        }

        return value_mask;
    }

    void make_prediction(Eigen::VectorXf& representation) {
        da.make_prediction(representation);
        bnd.make_prediction(representation);
    }

    std::string str() { return "Circuits"; }
    std::string repr() { return "Circuits"; }
    int len() { return CIRCUIT_SIZE; }
    std::array<float, CIRCUIT_SIZE> get_output() { return output; }
    std::array<float, 2> get_leaky_v() { return {da.get_leaky_v(), bnd.get_leaky_v()}; }
    float get_da_leaky_v() { return da.get_leaky_v(); }
    float get_bnd_leaky_v() { return bnd.get_leaky_v(); }
    float get_bnd_value(int idx) { return bnd.get_modulation_value(idx); }
    Eigen::VectorXf& get_da_weights() { return da.get_weights(); }
    Eigen::VectorXf& get_bnd_weights() { return bnd.get_weights(); }
    Eigen::VectorXf get_bnd_mask() { return bnd.make_mask(); }
    Eigen::VectorXf& get_value_mask() { return value_mask; }
    void reset() {
        da.reset();
        bnd.reset();
    }
};


struct DensityPolicy {

    float rwd_weight;
    float rwd_sigma;
    float rwd_threshold;
    float rwd_drive = 0.0f ;

    float col_weight;;
    float col_sigma;
    float col_threshold;
    float col_drive = 0.0f;

    float rwd_field_mod;
    float col_field_mod;

    std::array<bool, 4> options;

    void call(PCNN& space,
              Circuits& circuits,
              std::array<float, 2> displacement,
              float curr_da, float curr_bnd,
              float reward, float collision) {

        // +reward -collision
        if (reward > 0.1) {

            // update & remap
            rwd_drive = rwd_weight * curr_da;

            if (options[0]) {
                space.remap(circuits.get_da_weights(),
                            displacement, rwd_sigma, rwd_drive,
                            rwd_threshold);
            }

            if (options[1]) {
                space.modulate_gain(rwd_field_mod);
            }

        } else if (collision > 0.1) {

            // udpate & remap
            col_drive = col_weight * curr_bnd;

            if (options[2]) {
                space.remap(circuits.get_bnd_weights(),
                            // {-1.0f*displacement[0], -1.0f*displacement[1]},
                            displacement, col_sigma, col_drive,
                            col_threshold);
            }
            if (options[3]) {
                space.modulate_gain(col_field_mod);
            }
        }
    }

    DensityPolicy(float rwd_weight, float rwd_sigma,
                  float rwd_threshold, float col_sigma,
                  float col_weight, float col_threshold,
                  float rwd_field_mod,
                  float col_field_mod,
                  std::array<bool, 4> options):
        rwd_weight(rwd_weight), rwd_sigma(rwd_sigma),
        rwd_threshold(rwd_threshold), col_threshold(col_threshold),
        col_sigma(col_sigma), col_weight(col_weight),
        rwd_field_mod(rwd_field_mod),
        col_field_mod(col_field_mod),
        options(options){}

    std::string str() { return "DensityPolicy"; }
    std::string repr() { return "DensityPolicy"; }
    float get_rwd_mod() { return rwd_drive; }
    float get_col_mod() { return col_drive; }
};



/* ========================================== */
/* ========================================== */


namespace pcl {

std::pair<PCNN, GridNetwork>
make_space(float gc_sigma, std::array<float, NUM_GCL> gc_scales,
                float local_scale, int N,
                float rec_threshold_fine,
                float speed,
                float min_rep_threshold,

                float gain_fine,
                float offset_fine,
                float threshold_fine,
                float rep_threshold_fine,
                float tau_trace_fine,
                float remap_tag_frequency) {

    std::vector<GridLayer> gc_layers = {
                GridLayer(gc_sigma, gc_scales[0] * local_scale),
                GridLayer(gc_sigma, gc_scales[1] * local_scale),
                GridLayer(gc_sigma, gc_scales[2] * local_scale),
                GridLayer(gc_sigma, gc_scales[3] * local_scale),
                GridLayer(gc_sigma, gc_scales[4] * local_scale),
                GridLayer(gc_sigma, gc_scales[5] * local_scale),
                GridLayer(gc_sigma, gc_scales[6] * local_scale),
                GridLayer(gc_sigma, gc_scales[7] * local_scale),
                GridLayer(gc_sigma, gc_scales[8] * local_scale)};

    GridNetwork gcn = GridNetwork(gc_layers);

    PCNN space = PCNN(N, gcn.len(), gain_fine, offset_fine,
                      0.01f, threshold_fine, rep_threshold_fine,
                      rec_threshold_fine, min_rep_threshold, gcn,
                      tau_trace_fine, remap_tag_frequency, "fine");

    return std::make_pair(space, gcn);
};

};


/* #endif // MAINLIB_HPP */
