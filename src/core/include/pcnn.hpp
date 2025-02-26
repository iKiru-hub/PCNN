/* #ifndef PCNN_HPP */
/* #define PCNN_HPP */

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
#include <vector>
#include <random>
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
#define PCNN_REF PCNN
#define GCN_REF GridNetworkSq
#define GCL_REF GridLayerSq
#define CIRCUIT_SIZE 2
#define ACTION_SPACE_SIZE 16
#define POLICY_INPUT 5
#define POLICY_OUTPUT 6
#define POLICY_HIDDEN 3

/* ========================================== */

// blank log function
void LOG(const std::string& msg) {
    return;
    std::cout << msg << std::endl;
}


int SEED = 0;



std::mt19937& get_rng(unsigned int seed = 0) {
    static std::mt19937 rng(std::random_device{}());  // Persistent generator
    if (seed) rng.seed(seed);  // Reset the seed if provided
    return rng;
}


int random_int(int min, int max, unsigned int seed) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(get_rng(seed));
}


double random_float(double min, double max, unsigned int seed) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(get_rng(seed));
}


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
                    /* LOG("Distance penalty: " + std::to_string(new_distance) + " | " + std::to_string(neighbor)); */
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


inline float cosine_similarity_vec(const std::vector<float>& v1,
                                   const std::vector<float>& v2) {
    // Ensure the vectors are of the same size
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    // Calculate dot product
    float dot_product = std::inner_product(v1.begin(),
                                           v1.end(), v2.begin(), 0.0f);

    // Calculate norms of the vectors
    float norm_v1 = std::sqrt(std::inner_product(v1.begin(),
                                                 v1.end(),
                                                 v1.begin(), 0.0f));
    float norm_v2 = std::sqrt(std::inner_product(v2.begin(),
                                                 v2.end(),
                                                 v2.begin(), 0.0f));

    // Handle potential division by zero
    if (norm_v1 == 0.0f || norm_v2 == 0.0f) {
        throw std::invalid_argument(
            "Vectors must not have zero magnitude.");
    }

    // Return cosine similarity
    return dot_product / (norm_v1 * norm_v2);
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
/* ============ local functions ============= */
/* ========================================== */


// @brief boundary conditions for an hexagon
/*
INITIALIZATION:
- an hexagon of side 1
- center at C=(0, 0)
- nodes
- apothem length 0.86602540378

GIVEN:
- point P=(x, y)

PROCEDURE:

% checkpoint: short distance PO < apothem

% calculate the boundary conditions
1. reflect the point wrt the center C -> R
2. determine the two closest nodes A, B
3. calculate the intersection S between the line
   RO and the side of the hexagon AB

% checkpoint: no intersection S, point is inside the hexagon

4. reflect the point OP wrt the intersection S, this is the
   new point P

m = {
    {(-0.5+0.5)/2, (-0.86602540378-0.86602540378)/2},
    {(1+0.5)/2, (-0.86602540378+0)/2},
    {(1+0.5)/2, (0+0.86602540378)/2},
    {(0.5-0.5)/2, (0.86602540378+0.86602540378)/2},
    {(-0.5-1)/2, (0.86602540378+0)/2},
    {(-1-0.5)/2, (0-0.86602540378)/2}
}
for i=1,#m,1
do
    print(m[i][1].."; "..m[i][2])
end
*/


class Hexagon {

    std::array<std::array<float, 2>, 6> centers;
    std::array<size_t, 6> index = {0, 1, 2, 3, 4, 5};

    // @brief check whether p is within the inner circle
    bool apothem_checkpoint(float x, float y) const {
        float dist = std::sqrt(x * x + y * y);
        return dist < 0.86602540378f;
    }

    struct WrapResult {
        float x;
        float y;
        bool wrapped;
    };

    // @brief wrap the point to the boundary
    WrapResult wrap(float x, float y) const {
        // reflect the point p to r wrt the center (0, 0)
        float rx = -x;
        float ry = -y;

        // calculate and sort the distances to the centers
        std::array<float, 6> distances;
        for (int i = 0; i < 6; i++) {
            distances[i] = std::sqrt(
                std::pow(centers[i][0] - rx, 2) +
                std::pow(centers[i][1] - ry, 2)
            );
        }

        // Sort the index array based on the values in the original array
        auto index_copy = index;
        std::sort(index_copy.begin(), index_copy.end(),
            [&distances](const size_t& a, const size_t& b) {
                return distances[a] < distances[b];
            }
        );

        float ax = centers[index_copy[0]][0];
        float ay = centers[index_copy[0]][1];
        float bx = centers[index_copy[1]][0];
        float by = centers[index_copy[1]][1];
        float mx = (ax + bx) / 2.0f;
        float my = (ay + by) / 2.0f;

        // calculate the intersection s between ab and ro
        auto [intersects, sx, sy] = get_segments_intersection(
            ax, ay, bx, by, rx, ry, 0.0f, 0.0f);

        if (!intersects) {
            // checkpoint: no intersection, point is inside the hexagon
            /* LOG("[+] no intersection"); */
            return {x, y, false};
        }

        // reflect the point r wrt the intersection s
        rx = 2 * sx - rx;
        ry = 2 * sy - ry;

        // reflect wrt the line s-center
        std::array<float, 2> z;
        if (sy > 0) {
            z = reflect_point_over_segment(rx, ry, 0.0f, 0.0f, mx, my);
        } else {
            z = reflect_point_over_segment(rx, ry, mx, my, 0.0f, 0.0f);
        }

        return {z[0], z[1], true};
    }

public:
    Hexagon() {
        centers = {{
            {-0.5f, -0.86602540378f},
            {0.5f, -0.86602540378f},
            {1.0f, 0.0f},
            {0.5f, 0.86602540378f},
            {-0.5f, 0.86602540378f},
            {-1.0f, 0.0f}
        }};
    }

    // @brief apply the boundary conditions
    std::array<float, 2> call(float x, float y) const {
        if (!apothem_checkpoint(x, y)) {
            auto result = wrap(x, y);
            if (result.wrapped) {
                return {result.x, result.y};
            }
        }
        return {x, y};
    }

    std::string str() const { return "hexagon"; }
    std::string repr() const { return str(); }

    const std::array<std::array<float, 2>, 6>& get_centers() const {
        return centers;
    }
};


struct GSparams {
    float offset;
    float gain;
    float clip;

    GSparams(float offset, float gain, float clip):
        offset(offset), gain(gain), clip(clip) {}
};


struct VelocitySpace {

    void update_node_degree(int idx, int current_size) {

        // update the node itself
        node_degree(idx) = connectivity.row(idx).sum();

        // update the neighbors
        int degree = 0;
        for (int j = 0; j < current_size; j++) {
            if (connectivity(idx, j) == 1.0) {
                node_degree(j) = connectivity.row(j).sum();
            }
        }
    }

public:

    Eigen::MatrixXf centers;
    Eigen::MatrixXf connectivity;
    Eigen::MatrixXf weights;
    Eigen::VectorXf nodes_max_angle;
    Eigen::VectorXf node_degree;
    Eigen::VectorXf one_trace;
    std::array<float, 2> position;
    int size;
    int num_neighbors;
    float threshold;

    std::vector<std::array<int, 2>> blocked_edges;

    VelocitySpace(int size, float threshold, int num_neighbors)
        : size(size), threshold(threshold), num_neighbors(num_neighbors) {
        centers = Eigen::MatrixXf::Constant(size, 2, -9999.0f);
        connectivity = Eigen::MatrixXf::Zero(size, size);
        weights = Eigen::MatrixXf::Zero(size, size);
        one_trace = Eigen::VectorXf::Ones(size);
        position = {0.00124789f, 0.00147891f};
        blocked_edges = {};
        nodes_max_angle = Eigen::VectorXf::Zero(size);
        node_degree = Eigen::VectorXf::Zero(size);
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
        }

        // add recurrent connections
        /* int max_neighbors = 5; */
        /* for (int j = 0; j < size; j++) { */

        /*     // check if the nodes exist */
        /*     if (centers(idx, 0) < -999.0f || \ */
        /*         centers(j, 0) < -999.0f || \ */
        /*         idx == j) { */
        /*         continue; */
        /*     } */

        /*     // check if neighbors was active */
        /*     /1* if (traces(j) < 0.0001f) { continue; } *1/ */

        /*     float dist = std::sqrt( */
        /*         (centers(idx, 0) - centers(j, 0)) * */
        /*         (centers(idx, 0) - centers(j, 0)) + */
        /*         (centers(idx, 1) - centers(j, 1)) * */
        /*         (centers(idx, 1) - centers(j, 1)) */
        /*     ); */
        /*     if (dist < threshold) { */
        /*         this->weights(idx, j) = dist; */
        /*         this->weights(j, idx) = dist; */
        /*         this->connectivity(idx, j) = 1.0; */
        /*         this->connectivity(j, idx) = 1.0; */
        /*     } */
        /* } */

        /* NEAREST NEIGHBORS */

        // add recurrent connections based on nearest neighbors
        for (int idx = 0; idx < size; idx++) {

            // Skip invalid nodes
            if (centers(idx, 0) < -700.0f) { continue; }

            // Create a vector of pairs (distance, index) for all valid nodes
            std::vector<std::pair<float, int>> distances;
            for (int j = 0; j < size; j++) {
                // Skip invalid nodes and self
                if (centers(j, 0) < -700.0f || idx == j) { continue; }

                // check if neighbors was active
                /* if (traces(j) < 0.01f) { continue; } */

                float dist = std::sqrt(
                    (centers(idx, 0) - centers(j, 0)) * (centers(idx, 0) - centers(j, 0)) +
                    (centers(idx, 1) - centers(j, 1)) * (centers(idx, 1) - centers(j, 1))
                );

                if (dist < threshold) { distances.push_back(std::make_pair(dist, j)); }

            }

            // Sort by distance (ascending)
            std::sort(distances.begin(), distances.end());

            // Connect to the closest max_neighbors neighbors
            int num_connections = std::min(num_neighbors,
                                           static_cast<int>(distances.size()));
            this->weights.row(idx) = Eigen::VectorXf::Zero(size);
            this->connectivity.row(idx) = Eigen::VectorXf::Zero(size);

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
            node_degree(i) = node_degree(i+1);
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


/* ========================================== */
/* ============== INPUT LAYER =============== */
/* ========================================== */

// === purely hexagonal grid network ===

class GridLayerHex {

    const int N = 25;


    const std::array<std::array<float, 2>, 25> basis = {{
        {0.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, std::sin((float)M_PI/3.0f)},
        {0.5f, std::sin((float)M_PI/3.0f)},
        {1.0f, 0.0f},
        {0.5f, -std::sin((float)M_PI/3.0f)},
        {-0.5f, -std::sin((float)M_PI/3.0f)},
        {-1.0f, 0.0f},
        {0.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, 0.0f}
    }};
    std::array<float, 4> init_bounds = {-1.0f, 1.0f, -1.0f, 1.0f};
    std::array<std::array<float, 2>, 25> positions = {{
        {0.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, std::sin((float)M_PI/3.0f)},
        {0.5f, std::sin((float)M_PI/3.0f)},
        {1.0f, 0.0f},
        {0.5f, -std::sin((float)M_PI/3.0f)},
        {-0.5f, -std::sin((float)M_PI/3.0f)},
        {-1.0f, 0.0f},
        {0.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, 2.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-0.5f, 5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.0f, 4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {1.5f, -1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.5f, -5.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.0f, -4.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {-1.5f, 1.0f/3.0f * std::sin((float)M_PI/3.0f)},
        {0.0f, 0.0f}
    }};

    /* Eigen::VectorXf y = Eigen::VectorXf::Zero(25); */
    std::array<float, 25> y = {0.0f};
    float sigma;
    float speed;
    Hexagon hexagon;

    // define boundary type
    std::array<std::array<float, 2>, 25>
    boundary_conditions(std::array<std::array<float, 2>,
                             25> _positions) {
        for (int i = 0; i < 25; i++) {
            std::array<float, 2> new_position = \
                hexagon.call(_positions[i][0],
                             _positions[i][1]);
            _positions[i][0] = new_position[0];
            _positions[i][1] = new_position[1];
        }

        return _positions;
    }

    // define boundary type
    void boundary_conditions() {
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = hexagon.call(
                positions[i][0], positions[i][1]);
            this->positions[i][0] = new_position[0];
            this->positions[i][1] = new_position[1];
        }
    }

    // gaussian distance of each position to the centers
    void calc_activation() {
        float dist_squared;
        for (int i = 0; i < N; i++) {
            dist_squared = std::pow(positions[i][0], 2) + \
                std::pow(positions[i][1], 2);
            y[i] = std::exp(-dist_squared / sigma);
        }
    }

public:

    GridLayerHex(float sigma, float speed,
                 float offset_dx = 0.0f,
                 float offset_dy = 0.0f):
        sigma(sigma), speed(speed), hexagon(Hexagon()){

        // apply the offset by stepping
        if (offset_dx != 0.0f && offset_dy != 0.0f) {
            call({offset_dx, offset_dy});
        }
    }

    // CALL
    std::array<float, 25> \
    call(std::array<float, 2> v) {

        // update position with velociy
        for (int i = 0; i < N; i++) {
            positions[i][0] = positions[i][0] + speed * v[0];
            positions[i][1] = positions[i][1] + speed * v[1];
        }

        // apply boundary conditions
        boundary_conditions();

        // compute the activation
        calc_activation();
        return y;
    }

    // SIMULATE
    std::array<float, 25> simulate_one_step(
        std::array<float, 2> v) {

        std::array<std::array<float, 2>, 25> new_positions;
        for (int i = 0; i < GCL_SIZE; i++) {
            new_positions[i][0] = positions[i][0] + speed * v[0];
            new_positions[i][1] = positions[i][1] + speed * v[1];
        }

        boundary_conditions(new_positions);

        // compute the activation
        std::array<float, 25> yfwd = {0.0f};
        float dist_squared;
        for (int i = 0; i < 25; i++) {
            dist_squared = std::pow(new_positions[i][0],
                                    2) + \
                std::pow(new_positions[i][1], 2);
            yfwd[i] = std::exp(-dist_squared / sigma);
        }

        return yfwd;
    }

    int len() { return N; }
    std::string str() { return "GridLayerHex"; }
    std::string repr() { return "GridLayerHex"; }
    std::array<std::array<float, 2>, 25> get_positions()
    { return positions; }
    std::array<std::array<float, 2>, 25> get_centers()
    { return basis; }
    std::array<float, 25> get_activation() { return y; }
    void reset(std::array<float, 2> v) {
        this->positions = basis;
        call(v);
    }
};


class GridNetworkHex {

    std::vector<GridLayerHex> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;
    Eigen::MatrixXf positions;

public:

    GridNetworkHex(std::vector<GridLayerHex> layers)
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
        positions = Eigen::MatrixXf::Zero(total_size, 2);
    }

    // CALL
    Eigen::VectorXf call(const std::array<float, 2> x) {

        for (int i = 0; i < num_layers; i++) {
            // Convert the output of layers[i].call(x) to
            // an Eigen::VectorXf
            Eigen::VectorXf layer_output = \
                Eigen::Map<const Eigen::VectorXf>(
                    layers[i].call(x).data(), 25);

            // Assign the converted vector to
            // the corresponding segment of y
            y.segment(i * 25, 25) = layer_output;
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
                    layers[i].simulate_one_step(v).data(), 25);

            // Assign the converted vector to
            // the corresponding segment of y
            yfwd.segment(i * 25, 25) = layer_output;
        }

        return yfwd;
    }

    int len() const { return N; }
    int get_num_layers() const { return num_layers; }
    std::string str() const { return "GridNetwork"; }
    std::string repr() const {
        return str() + "(" + full_repr + ", N=" + \
        std::to_string(N) + ")"; }
    Eigen::VectorXf get_activation() const { return y; }
    Eigen::MatrixXf get_centers() {
        for (int i = 0; i < num_layers; i++) {
            int layer_len = layers[i].len();
            std::array<std::array<float, 2>, 25> layer_basis = \
                layers[i].get_centers();

            for (int j = 1; j < (layer_len+1); j++) {
                 basis(i*layer_len+j, 0) = layer_basis[j][0];
                 basis(i*layer_len+j, 1) = layer_basis[j][1];
                 }
        }
        return basis;
    }

    Eigen::MatrixXf get_positions() {
        for (int i = 0; i < num_layers; i++) {
            int layer_len = layers[i].len();
            std::array<std::array<float, 2>, 25> layer_positions = \
                layers[i].get_positions();

            for (int j = 1; j < (layer_len+1); j++) {
                 positions(i*layer_len+j, 0) = layer_positions[j][0];
                 positions(i*layer_len+j, 1) = layer_positions[j][1];
                 }
        }
        return basis;
    }

    std::vector<std::array<std::array<float, 2>, 25>> get_positions_vec() {
        std::vector<std::array<std::array<float, 2>, 25>> positions_vec;
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


// === purely square grid network ===

class GridLayerSq {

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

    GridLayerSq(float sigma, float speed,
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
    std::string str() { return "GridLayerSq"; }
    std::string repr() { return "GridLayerSq"; }
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


class GridNetworkSq {

    std::vector<GridLayerSq> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;

public:

    GridNetworkSq(std::vector<GridLayerSq> layers)
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
    std::string str()  { return "GridNetworkSq"; }
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
            basis.block(i * layers[i].len(), 0, layers[i].len(), 2) = layer_positions;
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
            basis.block(i * layers[i].len(), 0, layers[i].len(), 2) = layer_positions;
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


class PCNN {

    // internal components
    VelocitySpace vspace;
    GCN_REF xfilter;

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

    // ~activation
    float rep_threshold;
    float min_rep_threshold;
    float threshold;
    float gain;

    // remappint
    const int remap_tag_frequency;
    int t_update = 0;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    Eigen::MatrixXf Wff_gcp;
    float delta_wff;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf connectivity;
    Eigen::VectorXf mask;
    std::vector<int> fixed_indexes;
    std::vector<int> free_indexes;
    int cell_count;
    Eigen::VectorXf u;
    Eigen::VectorXf traces;
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
        free_indexes.clear();
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
         GCN_REF xfilter, float tau_trace = 2.0f,
         int remap_tag_frequency = 1, int num_neighbors = 3,
         std::string name = "fine")
        : N(N), Nj(Nj), gain(gain), gain_const(gain),
        offset(offset), clip_min(clip_min),
        min_rep_threshold(min_rep_threshold), rep_threshold(rep_threshold),
        rep_threshold_const(rep_threshold), rec_threshold(rec_threshold),
        threshold_const(threshold), threshold(threshold),
        xfilter(xfilter), tau_trace(tau_trace), name(name),
        remap_tag_frequency(remap_tag_frequency),
        vspace(VelocitySpace(N, rec_threshold, num_neighbors)) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        traces = Eigen::VectorXf::Zero(N);
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        // make vector of free indexes
        for (int i = 0; i < N; i++) { free_indexes.push_back(i); }
        fixed_indexes = {};
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
                                                          0.01);

        u = generalized_sigmoid_vec(u, offset, gain, clip_min);
        traces = traces - traces / tau_trace + u;
        traces = traces.cwiseMin(1.0f);

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
        int idx = free_indexes[cell_count];

        // determine weight update
        /* Eigen::VectorXf dw = x_filtered - Wff.row(idx).transpose(); */
        std::vector<float> dw;
        float delta_wff = 0.0f;
        for (int i = 0; i < Nj; i++) {
            dw.push_back(x_filtered(i) - Wff(idx, i));
            delta_wff += dw[i] * dw[i];
        }

        // trim the weight update
        /* delta_wff = dw.norm(); */

        if (delta_wff > 0.0) {

            for (int i = 0; i < Nj; i++) {
                if (dw[i] < 0.99f) { Wff(idx, i) += dw[i]; }
            }

            // update weights
            /* Wff.row(idx) += dw.transpose(); */

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
            cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);

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
        u = generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);
        return u;
    }

    void remap(Eigen::VectorXf& block_weights,
               std::array<float, 2> velocity,
               float width, float magnitude) {

        if (magnitude < 0.00001f) { return; }

        float magnitude_i;;
        for (int i = 0; i < N; i++) {

            // check remap tag
            if (!remap_tag[i] || traces(i) < 0.1) { continue; }

            // consider the trace
            /* magnitude_i = magnitude * 1.0f;// trace(i); */

            /* if (magnitude_i < 0.001f) { continue; } */

            // skip blocked edges
            if (vspace.centers(i, 0) < -900.0f || block_weights(i) > 0.0f)
                { continue; }

            std::array<float, 2> displacement = \
                {vspace.position[0] - vspace.centers(i, 0),
                 vspace.position[1] - vspace.centers(i, 1)};

            // gaussian activation function centered at zero
            float dist = std::exp(-std::sqrt(displacement[0] * displacement[0] + \
                                    displacement[1] * displacement[1]) / width) * \
                                    magnitude;

            if (vspace.is_too_close(i, {displacement[0] * magnitude, displacement[1] * magnitude},
                                    min_rep_threshold)) { continue; }

            // cutoff
            /* if (dist < 0.1f) { continue; } */

            // weight the displacement
            std::array<float, 2> gc_displacement = {dist * magnitude,
                                                    dist * magnitude};

            // pass the input through the filter layer
            x_filtered = xfilter.simulate_one_step(gc_displacement);

            // update the weights & centers
            Wff.row(i) += x_filtered.transpose() - Wff.row(i).transpose();

            // check similarity
            /* float similarity = max_cosine_similarity_in_rows(Wff, i); */

            // check repulsion (similarity) level
             /* - displacement[0] */
            /* if (similarity > min_rep_threshold || std::isnan(similarity)) { */
            /* if (similarity > (dist * rep_threshold * 0. + \ */
            /*     (1 -0*dist) * min_rep_threshold) \ */
            /*     || std::isnan(similarity)) { */
            /*     /1* std::cout << "remap failed, too close" << std::endl; *1/ */
            /*     Wff.row(i) = Wffbackup.row(i); */
            /*     continue; */
            /* } */

            // update backup and vspace
            Wffbackup.row(i) = Wff.row(i);
            /* vspace.remap_center(i, {dist * magnitude, dist * magnitude}); */
            vspace.remap_center(i, {displacement[0] * magnitude, displacement[1] * magnitude});
        }
    }

    void remap(std::array<float, 2> velocity,
               float width, float magnitude) {


        if (magnitude < 0.0001f && magnitude > -0.0001f) { return; }
        LOG("remapping with magnitude: " + std::to_string(magnitude));

        float magnitude_i;;
        for (int i = 0; i < N; i++) {

            // check remap tag
            if (!remap_tag[i] || traces(i) < 0.1) { continue; }

            // consider the trace
            /* magnitude_i = magnitude * 1.0f;// trace(i); */

            /* if (magnitude_i < 0.001f) { continue; } */

            // skip blocked edges
            if (vspace.centers(i, 0) < -900.0f) { continue; }

            std::array<float, 2> displacement = \
                {vspace.position[0] - vspace.centers(i, 0),
                 vspace.position[1] - vspace.centers(i, 1)};

            // gaussian activation function centered at zero
            /* float dist = std::exp(-std::sqrt(displacement[0] * displacement[0] + \ */
            /*                         displacement[1] * displacement[1]) / width); */
            // Euclidean distance
            /* float dist = std::sqrt(displacement[0] * displacement[0] + \ */
            /*                        displacement[1] * displacement[1]); */
            float dist = std::exp(-std::sqrt(displacement[0] * displacement[0] + \
                                    displacement[1] * displacement[1]) / width) * \
                                    magnitude;

            if (vspace.is_too_close(i, {displacement[0] * magnitude, displacement[1] * magnitude},
                                    min_rep_threshold)) { continue; }

            // cutoff
            /* if (dist > width) { continue; } */
            /* if (dist < 0.1f) { continue; } */

            // weight the displacement
            std::array<float, 2> gc_displacement = {dist * magnitude,
                                                    dist * magnitude};

            // pass the input through the filter layer
            x_filtered = xfilter.simulate_one_step(gc_displacement);

            // update the weights & centers
            Wffbackup.row(i) = Wff.row(i);
            Wff.row(i) += x_filtered.transpose() - Wff.row(i).transpose();

            // check similarity
            /* float similarity = max_cosine_similarity_in_rows(Wff, i); */

            /* // check repulsion (similarity) level */
            /*  /1* - displacement[0] *1/ */
            /* if (similarity > min_rep_threshold || std::isnan(similarity)) { */
            /*     /1* std::cout << "remap failed, too close" << std::endl; *1/ */
            /*     Wff.row(i) = Wffbackup.row(i); */
            /*     continue; */
            /* } */

            /* LOG("remapping with magnitude: " + std::to_string(magnitude)); */
            /* std::cout << "remapping with similarity: " << similarity << "\n"; */

            // update backup and vspace
            Wffbackup.row(i) = Wff.row(i);
            vspace.remap_center(i, {displacement[0] * magnitude, displacement[1] * magnitude});
            /* vspace.remap_center(i, {dist * magnitude, */
            /*                         dist * magnitude}); */
        }
    }

    void single_remap(int idx, std::array<float, 2> displacement,
                      float magnitude) {

        // consider the trace
        /* magnitude = magnitude * trace(idx); */

        if (magnitude < 0.00001f) { return; }

        // weight the displacement
        std::array<float, 2> gc_displacement = {
                        displacement[0] * magnitude - displacement[0],
                        displacement[1] * magnitude - displacement[1]};

        // pass the input through the filter layer
        x_filtered = xfilter.simulate_one_step(gc_displacement);

        // update the weights & centers
        Wff.row(idx) += x_filtered.transpose() - Wff.row(idx).transpose();

        // check similarity
        float similarity = max_cosine_similarity_in_rows(Wff, idx);

        // check repulsion (similarity) level
        if (similarity > min_rep_threshold || std::isnan(similarity)) {
            Wff.row(idx) = Wffbackup.row(idx);
            return;
        }

        std::cout << "remapping with magnitude: " << magnitude << "\n";

        // update backup and vspace
        Wffbackup.row(idx) = Wff.row(idx);
        vspace.remap_center(idx, {displacement[0] * magnitude,
                                  displacement[1] * magnitude});
    }

    int calculate_closest_index(const std::array<float, 2>& c)
        { return vspace.calculate_closest_index(c); }

    int get_neighbourhood_node_degree(int idx) {
        int node_degree = vspace.node_degree(idx);
        int count = 1;

        for (int i = 0; i < N; i++) {
            if (vspace.connectivity(idx, i) > 0.0f) {
                node_degree += vspace.node_degree(i);
                count++;
            }
        }
        return node_degree / count;
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
        cell_count--;
        fixed_indexes.erase(std::remove(fixed_indexes.begin(),
                                        fixed_indexes.end(), idx),
                            fixed_indexes.end());
        free_indexes.push_back(idx);

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
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rec_threshold) + ")";
    }
    Eigen::VectorXf& get_activation() { return u; }
    Eigen::VectorXf get_activation_gcn()
        { return xfilter.get_activation(); }
    Eigen::MatrixXf& get_wff() { return Wff; }
    Eigen::MatrixXf& get_wrec() { return Wrec; }
    float get_trace_value(int idx) { return traces(idx); }
    float get_max_activation() { return u.maxCoeff(); }
    std::vector<std::array<std::array<float, 2>, 2>> make_edges()
        { return vspace.make_edges(); }
    std::vector<std::array<std::array<float, 2>, 3>> make_edges_value(
        Eigen::MatrixXf& values) { return vspace.make_edges_value(values); }
    Eigen::MatrixXf& get_connectivity() { return connectivity; }
    Eigen::MatrixXf get_centers(bool nonzero = false)
        { return vspace.get_centers(nonzero); }
    Eigen::VectorXf& get_node_degrees() { return vspace.node_degree; }
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
        if (!simulate) {
            for (int i = 0; i < size; i++) {

                // forward, delta from current input
                float ui = u[i] > threshold ? u[i] : 0.0;
                /* float dw = lr * v * ui; */

                // backward, prediction error
                float pred_err = lr_pred * (prediction[i] - v * ui);
                if (pred_err < 0.0) { pred_err = 0.0; }

                // clip the prediction error if within a certain range
                pred_err = pred_err > 0.001f && pred_err < 0.1 ? 0.1 : pred_err;

                if (pred_err > 0.01 && name == "DA") { LOG("[+] prediction error: " + std::to_string(pred_err));}

                // update weights
                weights[i] += lr * v * ui - pred_err;

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
                LOG("[+] update " + name + " target: [" + std::to_string(trg_idx) + "] " + std::to_string(maxValue));
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

        if (value > 0.00f) {
            LOG("[+] " + name + " prediction: [" + std::to_string(idx) + "] " + std::to_string(value));
        }
    }

    float get_output() { return output; }

    Eigen::VectorXf& get_weights() { return weights; }
    Eigen::VectorXf& make_mask() {

        // the nodes that are fresh in memory will affect action selection,
        // thus acting as a mask
        for (int i = 0; i < size; i++) { this->mask(i) = weights(i) > mask_threshold ? 0.0f : 1.0f; }
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
        LOG("[+] Circuits created, threshold=" + std::to_string(threshold));
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
            float bnd_value = bnd_weights(i);

            /* if (!strict) { */
            /*     value_mask(i) = bnd_value < 0.01f ? 1.0f : 0.0f; */
            /*     continue; */
            /* } */

            if (bnd_value < 0.01f) {
                value_mask(i) = 1.0f; }
            else if (bnd_value < threshold) {
                value_mask(i) = -4000.0f; }
            else {
                value_mask(i) = -10000.0f; }
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


/* ========================================== */
/* ============== BEHAVIOURS ================ */
/* ========================================== */


struct RewardObject {

    int trg_idx = -1;
    float trg_value = 0.0f;
    float min_weight_value;

    int update(Eigen::VectorXf& da_weights,
               PCNN_REF& space_fine,
               bool trigger = true) {

        // exit: no trigger
        if (!trigger) {
            return -1;

            // exit: weights not strong enough
            if (da_weights.maxCoeff() < min_weight_value) {
                return -1;
            }
        }

        // --- update the target representation ---

        // method 1: take the center of mass
        trg_idx = converge_to_trg_index(da_weights, space_fine);

        // try method 2
        if (trg_idx < 0) {

            // method 2: take the argmax of the weights
            Eigen::Index maxIndex;
            float maxValue = da_weights.maxCoeff(&maxIndex);
            trg_idx = static_cast<int>(maxIndex);
        }

        // exit: no trg index
        if (trg_idx < 0) {
            LOG("[RwO] no trg index");
            return -1; }

        // exit: low trg value
        if (da_weights(trg_idx) < 0.00001f) {
            LOG("[RwO] low trg value");
            return -1; }

        // update the target value
        this->trg_idx = trg_idx;
        this->trg_value = da_weights(trg_idx);

        LOG("[RwO] goal_idx=" + std::to_string(trg_idx) + \
            " | goal_value=" + std::to_string(trg_value));

        return trg_idx;
    }

    RewardObject(float min_weight_value = 0.01):
        min_weight_value(min_weight_value) {}

private:

    int converge_to_trg_index(Eigen::VectorXf& da_weights,
                              PCNN_REF& space_fine) {

        // weights for the centers
        float cx, cy;
        float sum = da_weights.sum();
        if (sum == 0.0f) {
            /* LOG("[-] sum is zero"); */
            return -1;
        }

        Eigen::MatrixXf centers = space_fine.get_centers();

        for (int i = 0; i < da_weights.size(); i++) {
            cx += da_weights(i) * centers(i, 0);
            cy += da_weights(i) * centers(i, 1);
        }

        // centers of mass
        cx /= sum;
        cy /= sum;

        // get closest center
        int closest_idx = space_fine.calculate_closest_index({cx, cy});

        return closest_idx;
    }
};


struct Plan {

    // external components
    PCNN_REF& space_fine;

    // parameters
    bool is_coarse;
    float speed;

    // variables
    std::vector<int> plan_idxs = {};
    std::array<float, 2> curr_position = {0.0f, 0.0f};
    std::array<float, 2> next_position = {0.0f, 0.0f};
    int size = -1;
    int counter = 0;
    int curr_idx = -1;
    int trg_idx = -1;


    std::pair<std::array<float, 2>, bool> step_plan() {

        float distance = calculate_distance();

        // same next position
        if (distance > 0.01f && counter > 0) {
            /* LOG("[plan] same next position"); */
            return std::make_pair(make_velocity(), true);
        }

        // check: end of the plan
        if (counter > size || curr_idx == trg_idx) {
            reset();
            /* LOG("[plan] end of the plan"); */
            return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
        }

        // retrieve next position
        this->next_position = {space_fine.get_centers()(plan_idxs[counter], 0),
                               space_fine.get_centers()(plan_idxs[counter], 1)};

        // check if it's the last point
        /* if (plan_idxs[counter] == trg_idx) { */
            /* LOG("[plan] last point | idx=" + std::to_string(trg_idx)); */
        /* } else if (plan_idxs[counter] < -10000 || plan_idxs[counter] > 10000) { */
        /*     LOG("[!plan] !!! possible memory leak?? " + \ */
        /*         std::to_string(plan_idxs[counter]) + ", counter=" + \ */
        /*         std::to_string(counter) + " | plan size=" + \ */
        /*         std::to_string(plan_idxs.size())); */
        /* } */

        counter++;
        return std::make_pair(make_velocity(), true);
    }

    std::array<float, 2> make_velocity() {

        std::array<float, 2> local_velocity;
        float dx = next_position[0] - curr_position[0];
        float dy = next_position[1] - curr_position[1];

        // determine velocity magnitude
        if (std::sqrt(dx * dx + dy * dy) < speed)
            {
            /* LOG("[plan] just a little bit"); */
            local_velocity = {dx, dy}; }
        else {
            float norm = std::sqrt(dx * dx + dy * dy);
            /* LOG("[plan] norm=" + std::to_string(norm) + \ */
            /*     " | speed=" + std::to_string(speed)); */
            local_velocity = {speed * dx / norm,
                              speed * dy / norm};
        }
        /* LOG("[plan] local_velocity=" + std::to_string(local_velocity[0]) + \ */
        /*     ", " + std::to_string(local_velocity[1])); */
        return local_velocity;
    }

    float calculate_distance() {

        curr_position = space_fine.get_position();
        curr_idx = space_fine.calculate_closest_index(curr_position);
        return euclidean_distance(curr_position, next_position);
    }

    void set_plan(std::vector<int> plan_idxs) {
        this->plan_idxs = plan_idxs;
        this->size = plan_idxs.size();
        this->counter = 0;
        this->curr_idx = -1;
        this->trg_idx = plan_idxs.back();
    }

    bool is_finished() { return counter > size; }
    std::array<float, 2> get_next_position() { return next_position; }

    void reset() {
        plan_idxs = {};
        size = -1;
        counter = 0;
        curr_idx = -1;
        trg_idx = -1;
    }

    Plan(PCNN_REF& space_fine, bool is_coarse, float speed):
        space_fine(space_fine), is_coarse(is_coarse), speed(speed) {}
};


class GoalModule {

    // external components
    PCNN_REF& space_fine;
    PCNN_REF& space_coarse;
    Circuits& circuits;
    Eigen::VectorXf flat_weights;

    // internal components
    Plan fine_plan;
    Plan coarse_plan;

    // variables
    bool using_coarse = true;
    bool is_fine_tuning = false;
    int fine_tuning_time = 0;
    int final_fine_idx = -1;

    std::pair<std::vector<int>, bool> make_plan(PCNN_REF& space_fine,
                    Eigen::VectorXf& space_weights,
                   int trg_idx, int curr_idx = -1) {

        // current index and value
        if (curr_idx == -1)
            { curr_idx = space_fine.calculate_closest_index(space_fine.get_position()); }

        // check: current position at the boundary
        if (space_weights(curr_idx) < -1000.0f) {
            return std::make_pair(std::vector<int>{}, false); }

        // make plan path
        std::vector<int> plan_idxs = \
            spatial_shortest_path(space_fine.get_connectivity(),
                                         space_fine.get_centers(),
                                         space_weights,
                                         curr_idx, trg_idx);

        // check if the plan is valid, ie size > 1
        if (plan_idxs.size() < 1) {
            return std::make_pair(std::vector<int>{}, false); }

        LOG("[goal] new plan ############################### return bnd.get_weights()(idx); :");
        LOG(" ");

        // check bnd value of each index
        for (int i = 0; i < plan_idxs.size(); i++) {
            LOG("[goal] idx=" + std::to_string(plan_idxs[i]) + \
                " | value=" + std::to_string(space_weights(plan_idxs[i])));
        }

        return std::make_pair(plan_idxs, true);
    }

public:

    GoalModule(PCNN_REF& space_fine, PCNN_REF& space_coarse,
               Circuits& circuits, float speed, float speed_coarse):
        space_fine(space_fine), space_coarse(space_coarse),
        circuits(circuits),
        flat_weights(Eigen::VectorXf::Ones(space_coarse.get_size())),
        coarse_plan(space_coarse, true, speed_coarse),
        fine_plan(space_fine, false, speed) {}

    bool update(int trg_idx_fine, bool goal_directed) {

        // -- proporse a coarse plan --

        // extract trg_idx in the coarse space
        int trg_idx_coarse = space_coarse.calculate_closest_index(
                {space_fine.get_centers()(trg_idx_fine, 0),
                 space_fine.get_centers()(trg_idx_fine, 1)});

        // plan from the current position
        std::pair<std::vector<int>, bool> res_coarse = \
            make_plan(space_coarse, flat_weights,
                      trg_idx_coarse);

        // check: failed coarse planning
        if (!res_coarse.second) {
            LOG("[Goal] failed coarse planning");
            return false;
        }

        // plan from the current position
        std::pair<std::vector<int>, bool> res_fine_prop = \
            make_plan(space_fine, circuits.make_value_mask(goal_directed),
                      trg_idx_fine);

        // check: failed fine planning
        if (!res_fine_prop.second) {
            LOG("[Goal] failed fine planning");
            return false;
        }

        // -- make a fine plan from the end of the coarse plan

        // extract the last index of the coarse plan
        int curr_idx_fine = space_fine.calculate_closest_index(
                {space_coarse.get_centers()(res_coarse.first.back(), 0),
                 space_coarse.get_centers()(res_coarse.first.back(), 1)});

        // plan from the last indegoal_directedx of the coarse plan
        std::pair<std::vector<int>, bool> res_fine = \
            make_plan(space_fine, circuits.make_value_mask(goal_directed),
                      trg_idx_fine, curr_idx_fine);

        // check: failed planning
        if (!res_fine.second) {
            LOG("[Goal] failed fine planning");
            return false; }

        // record
        coarse_plan.set_plan(res_coarse.first);
        final_fine_idx = trg_idx_fine;

        return true;
    }

    std::pair<std::array<float, 2>, bool> step_plan(bool obstacle = false) {

        // exit: active
        if (coarse_plan.is_finished() && fine_plan.is_finished())
            { return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false); }

        std::array<float, 2> local_velocity;

        // -- coarse plan
        if (using_coarse && !coarse_plan.is_finished() && \
            !obstacle && !is_fine_tuning) {
            std::pair<std::array<float, 2>, bool> coarse_progress = \
                        coarse_plan.step_plan();

            /* LOG("[Goal] coarse_progress=" + std::to_string(coarse_progress.second)); */

            // exit: coarse action
            if (coarse_progress.second) {
                /* LOG("[Goal] coarse action" + std::to_string(coarse_progress.second)); */
                return coarse_progress;
            }
        }
        /* LOG("[Goal] obstacle=" + std::to_string(obstacle)); */

        // -- fine plan

        // [] case 1: already fine tuning
        if (is_fine_tuning) {
            /* LOG("[Goal] fine tuning.."); */
            std::pair<std::array<float, 2>, bool> fine_progress = \
                fine_plan.step_plan();

            // fine plan finished
            if (fine_plan.is_finished()) {
                is_fine_tuning = false;

                // also the coarse plan is finished
                if (coarse_plan.is_finished()) { return fine_progress; }
            }

            return std::make_pair(fine_progress.first, true);
        }

        // [] case 2: start fine tuning
        int trg_idx_fine = -1;
        if (coarse_plan.is_finished()) { trg_idx_fine = final_fine_idx; }
        else {
            // make a new plan to the coarse plan next position
            trg_idx_fine = space_fine.calculate_closest_index(
                        coarse_plan.get_next_position());
        }
        /* LOG("[Goal] trg_idx_fine=" + std::to_string(trg_idx_fine)); */
        std::pair<std::vector<int>, bool> fine_progress = \
            make_plan(space_fine, circuits.make_value_mask(true),
                      trg_idx_fine);

        // check: failed planning
        if (!fine_progress.second) {
            /* LOG("[Goal] failed fine planning"); */

            // end coarse plan too
            coarse_plan.reset();
            return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
        }

        // record
        fine_plan.set_plan(fine_progress.first);
        is_fine_tuning = true;
        LOG("[Goal] start fine tuning");

        return fine_plan.step_plan();
    }

    void reset() {
        coarse_plan.reset();
        fine_plan.reset();
        is_fine_tuning = false;
        fine_tuning_time = 0;
        final_fine_idx = -1;
    }

    bool is_active() { return !coarse_plan.is_finished() || \
        !fine_plan.is_finished(); }
    std::vector<int> get_plan_idxs_fine() { return fine_plan.plan_idxs; }
    std::vector<int> get_plan_idxs_coarse() { return coarse_plan.plan_idxs; }
};


std::array<bool, 4> remapping_options(int flag) {

    switch (flag) {
        case 0:
            return {true, true, true, true};
        case 1:
            return {true, false, false, false};
        case 2:
            return {false, true, false, false};
        case 3:
            return {true, false, true, false};
        case 4:
            return {true, true, true, false};
        case 5:
            return {true, false, true, true};
        case 6:
            return {false, true, false, true};
        case 7:
            return {false, true, true, true};
        default:
            return {false, false, false, false};
    }
}


struct DensityPolicy {

    float rwd_weight;
    float rwd_sigma;
    float col_weight;;
    float col_sigma;
    float rwd_drive = 0.0f;
    float col_drive = 0.0f;

    bool remapping_flag;
    std::array<bool, 4> remapping_option;

    void call(PCNN_REF& space_fine,
              PCNN_REF& space_coarse,
              Circuits& circuits,
              GoalModule& goalmd,
              std::array<float, 2> displacement,
              float curr_da, float curr_bnd,
              float reward, float collision) {

        if (remapping_flag < 0) { return; }

        // +reward -collision
        if (reward > 0.1) {

            // update & remap
            Eigen::VectorXf bnd_weights = circuits.get_da_weights();
            rwd_drive = rwd_weight * curr_da;

            if (remapping_option[0]) {
                space_fine.remap(bnd_weights, displacement, rwd_sigma, rwd_drive);
            }
            if (remapping_option[1]) {
                space_coarse.remap(displacement, rwd_sigma, rwd_drive);
            }

            /* space_fine.remap(bnd_weights, displacement, rwd_sigma, rwd_drive); */
            /* space_coarse.remap(displacement, rwd_sigma, rwd_drive); */
            /* remap_space(bnd_weights, space, goalmd, displacement, */
            /*             rwd_sigma, rwd_drive); */

        } else if (collision > 0.1) {

            // udpate & remap
            Eigen::VectorXf da_weights = circuits.get_bnd_weights();
            col_drive = col_weight * curr_bnd;

            if (remapping_option[2]) {
                space_fine.remap(da_weights, {-1.0f*displacement[0], -1.0f*displacement[1]},
                                 col_sigma, col_drive);
            }
            if (remapping_option[3]) {
                LOG("[+] collision remap coarse");
                space_coarse.remap({-1.0f*displacement[0], -1.0f*displacement[1]},
                                   col_sigma, col_drive);
            }

            /* space_fine.remap(da_weights, displacement, col_sigma, col_drive); */
            /* space_coarse.remap(displacement, col_sigma, col_drive); */
        }
    }

    DensityPolicy(float rwd_weight, float rwd_sigma,
                  float col_weight, float col_sigma,
                  int remapping_flag = -1):
        rwd_weight(rwd_weight), rwd_sigma(rwd_sigma),
        col_sigma(col_sigma), col_weight(col_weight),
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


class ExplorationModule {

    // external components
    Circuits& circuits;
    PCNN_REF& space;

    // parameters
    float speed;
    float action_delay;
    const float open_threshold = 2.0f;  // radiants
    Eigen::VectorXf rejected_indexes;

    const int sparsity_threshold = 5;

    // plan
    int t = 0;
    std::array<float, 2> action = {0.0f, 0.0f};
    int edge_idx = -1;

    // alternate between random walk and edge exploration
    int edge_route_time = 0;
    int edge_route_interval = 100;

    // make new plan
    int make_plan(int rejected_idx) {

        // check: evaluate current position
        if (is_edge_position()) { return -1; }

        int rand_idx = sample_random_idx(0);

        // [+] new random walk plan at an open boundary
        if (rand_idx < 0 || rejected_idx == 404 || \
            edge_route_time < edge_route_interval) { return -1; }

        // [+] new trg plan to reach the open boundary
        LOG("[Exp] new trg plan to reach a random point");
        return rand_idx;
    }

    // make random plan
    void random_action_plan() {

        // sample a random angle
        float angle = random_float(0.0f, 2.0f * M_PI, SEED);

        // update plan
        this->action = {std::cos(angle) * speed,
                        std::sin(angle) * speed};
        this->t = 0;
    }

    std::pair<std::array<float, 2>, bool> step_random_plan() {

        // exit: the action have been provided #action_delay times
        if (t > (action_delay-1)) { return std::make_pair(action, true); }

        // provide the action
        this->t++;
        return std::make_pair(action, false);
    }

    int sample_random_idx(int num_attempts) {

        if (num_attempts > 20) { return -1; }

        // sample a random index
        int idx = random_int(0, space.get_size(), SEED);

        // check: the index is not in the rejected indexes
        if (rejected_indexes(idx) > 0.0f) { return sample_random_idx(num_attempts+1); }

        // check is it is on the boundary
        if (circuits.get_bnd_weights()(idx) < 0.01f || space.get_trace_value(idx) > 0.001f)
         { return idx; }

        return sample_random_idx(num_attempts+1);
    }

    bool is_edge_position() {
        // get idx of the current position
        int curr_idx = space.calculate_closest_index(space.get_position());

        int neighbourhood_degree = space.get_neighbourhood_node_degree(curr_idx);

        if (neighbourhood_degree < sparsity_threshold) { return true; }
        return false;
    }

public:
    bool new_plan = true;

    ExplorationModule(float speed, Circuits& circuits,
                     PCNN_REF& space, float action_delay = 1.0f,
                     int edge_route_interval = 100):
        speed(speed), circuits(circuits), space(space),
        action_delay(action_delay), edge_route_interval(edge_route_interval),
        rejected_indexes(Eigen::VectorXf::Zero(space.get_size())) {}

    // CALL
    std::pair<std::array<float, 2>, int>
    call(std::string directive, int rejected_idx = -1) {

        // check: previous plan ongoing
        std::pair<std::array<float, 2>, bool> next_step = step_random_plan();

        // -- new plan | for directive "new" or if the previous plan is finished
        if (directive == "new" || next_step.second) {

            // rejection : the attempt to make a trg plan to the boundary has failed
            int res = make_plan(rejected_idx);
            this->new_plan = true;

            // new plan to the open boundary
            if (res > -1) { return {{-1.0f, -1.0f}, res}; }
            else {
                // make random action
                random_action_plan();

                // step the plan | (action, done)
                std::pair<std::array<float, 2>, bool> next_step = step_random_plan();
                edge_route_time++;
                return {next_step.first, -1};
            }
        }

        // -- continue plan
        this->new_plan = false;
        return { next_step.first, -1 };
    }

    std::string str() { return "ExplorationModule"; }
    std::string repr() { return "ExplorationModule"; }
    void confirm_edge_walk() { edge_route_time = 0; }
    void reset_rejected_indexes()
        { rejected_indexes = Eigen::VectorXf::Zero(space.get_size()); }
    Eigen::VectorXf get_edge_representation() {

        Eigen::VectorXf edge_rep = Eigen::VectorXf::Zero(space.get_size());

        if (edge_idx < 0) { return edge_rep; }
        edge_rep(edge_idx) = 1.0f;
        return edge_rep;
    }
};


/* ========================================== */
/* ================= BRAIN ================== */
/* ========================================== */


class Brain {

    // external components
    BaseModulation da;
    BaseModulation bnd;
    std::vector<GCL_REF> gcn_layers;
    GCN_REF gcn;

    // external components
    Circuits circuits;
    PCNN_REF space_fine;
    PCNN_REF space_coarse;
    ExplorationModule expmd;
    StationarySensory ssry;
    DensityPolicy dpolicy;
    GoalModule goalmd;
    RewardObject rwobj;

    // variables
    Eigen::VectorXf curr_representation;
    Eigen::VectorXf curr_representation_coarse;
    std::string directive;
    int clock;
    int trg_plan_end = 0;
    int forced_exploration = -1;
    int forced_duration;

    // initialize
    std::pair<std::array<float, 2>, int> expmd_res = \
        std::make_pair(std::array<float, 2>{0.0f, 0.0f}, 0);
    int tmp_trg_idx = -1;
    std::array<float, 2> action = {0.0f, 0.0f};

    std::array<float, 2> attempt_boundary_plan(int idx) {

        // reset the set of rejected indexes
        expmd.reset_rejected_indexes();

        // attempt for a bunch of times | use 404 as final rejection
        for (int i = 0; i < 3; i++) {

            // attempt a plan
            goalmd.reset();
            bool valid_plan = goalmd.update(idx, false);

            // valid plan
            if (valid_plan) {
                this->directive = "trg ob";
                std::pair<std::array<float, 2>, bool> progress = \
                    goalmd.step_plan(false);

                // confirm the edge walk
                expmd.confirm_edge_walk();
                return progress.first;
            }

            // invalid plan -> try again
            std::pair<std::array<float, 2>, int> expmd_res = \
                expmd.call(directive, idx);
            idx = expmd_res.second;
        }

        // tried too many times, make a random walk plan instead
        std::pair<std::array<float, 2>, int> expmd_res = \
            expmd.call(directive, 404);

        return expmd_res.first;
    }

    void make_prediction() {

        // simulate a step
        Eigen::VectorXf& next_representation = space_fine.simulate_one_step(
            action);

        // make prediction
        circuits.make_prediction(next_representation);
    }

    void prune_bnd_edges() {

        // get the current idx
        int curr_idx = space_fine.calculate_closest_index(
            space_fine.get_position());

        if (curr_idx < 0) { return; }

        // go over the neighbourhood
        for (int j = 0; j < space_fine.get_size(); j++) {
            if (space_fine.check_edge(curr_idx, j)) {
                if (circuits.get_bnd_value(j) > 0.01f) {
                    space_fine.delete_edge(curr_idx, j);
                }
            }
        }
    }


public:

    Brain(float local_scale_fine,
          float local_scale_coarse,

          int N,
          int Nc,
          float rec_threshold_fine,
          float rec_threshold_coarse,
          float speed,
          float min_rep_threshold,
          int num_neighbors,

          float gain_fine,
          float offset_fine,
          float threshold_fine,
          float rep_threshold_fine,
          float tau_trace_fine,
          int remap_tag_frequency,

          float gain_coarse,
          float offset_coarse,
          float threshold_coarse,
          float rep_threshold_coarse,
          float tau_trace_coarse,

          float lr_da,
          float lr_pred,
          float threshold_da,
          float tau_v_da,

          float lr_bnd,
          float threshold_bnd,
          float tau_v_bnd,

          float tau_ssry,
          float threshold_ssry,

          float threshold_circuit,
          int remapping_flag,

          float rwd_weight,
          float rwd_sigma,
          float col_weight,
          float col_sigma,

          float action_delay,
          int edge_route_interval,

          int forced_duration,
          int fine_tuning_min_duration,
          float min_weight_value = 0.3):
        clock(0), forced_duration(forced_duration), directive("new"),
        da(BaseModulation("DA", N, lr_da, lr_pred, threshold_da, 1.0f,
                          tau_v_da, 0.0f, 0.4f, 0.1f)),
        bnd(BaseModulation("BND", N, lr_bnd, 0.0f, threshold_bnd, 1.0f,
                           tau_v_bnd, 0.0f, 0.1f)),
        ssry(StationarySensory(N, tau_ssry, threshold_ssry, 0.99)),
        circuits(Circuits(da, bnd, threshold_circuit)),

        // initialize with a set of GridLayerSq
        gcn_layers({GCL_REF(0.04, 1.0 * local_scale_fine),
                    GCL_REF(0.04, 0.8 * local_scale_fine),
                    GCL_REF(0.04, 0.7 * local_scale_fine),
                    GCL_REF(0.04, 0.5 * local_scale_fine),
                    GCL_REF(0.04, 0.3 * local_scale_fine),
                    GCL_REF(0.04, 0.2 * local_scale_fine),
                    GCL_REF(0.04, 0.1 * local_scale_fine)}),
        gcn(GCN_REF(gcn_layers)),
        space_fine(PCNN(N, gcn.len(), gain_fine, offset_fine,
                        0.01f, threshold_fine, rep_threshold_fine,
                        rec_threshold_fine, min_rep_threshold, gcn,
                        tau_trace_fine, remap_tag_frequency,
                        num_neighbors, "fine")),
        space_coarse(PCNN(Nc, gcn.len(), gain_coarse, offset_coarse,
                          0.01f, threshold_coarse, rep_threshold_coarse,
                          rec_threshold_coarse, min_rep_threshold,
                          gcn, tau_trace_coarse, 1,
                          num_neighbors, "coarse")),
        goalmd(GoalModule(space_fine, space_coarse, circuits, speed,
                          speed * local_scale_fine / local_scale_coarse)),
        rwobj(RewardObject(min_weight_value)),
        dpolicy(DensityPolicy(rwd_weight, rwd_sigma, col_weight,
                              col_sigma, remapping_flag)),
        expmd(ExplorationModule(speed * 2.0f, circuits, space_fine,
                                action_delay, edge_route_interval)) {}

    // CALL
    std::array<float, 2> call(
            const std::array<float, 2>& velocity,
            float collision, float reward,
            bool trigger) {

        clock++;

        if (collision > 0.0f) {
            prune_bnd_edges();
            LOG("[Brain] collision received");
        }
        if (reward > 0.0f) { LOG("[Brain] reward received"); }

        // === STATE UPDATE ==============================================

        // :space
        auto [u, _] = space_fine.call(velocity);
        auto [uc, __] = space_coarse.call(velocity);
        this->curr_representation = u;
        this->curr_representation_coarse = uc;

        // :circuits
        std::array<float, CIRCUIT_SIZE> internal_state = \
            circuits.call(u, collision, reward, false);

        // :dpolicy fine space
        dpolicy.call(space_fine,
                     space_coarse,
                     circuits,
                     goalmd,
                     velocity,
                     internal_state[1], internal_state[0],
                     reward, collision);
        space_fine.update();

        if (circuits.get_bnd_leaky_v() < 0.001 && space_fine.get_max_activation() > 0.45) {
            space_coarse.update(); 
        }// else if (collision > 0.01f) { space_coarse.update_upon_collision(); }

        // check: still-ness | wrt the fine space
        if (forced_exploration < forced_duration) {
            forced_exploration++;
            goalmd.reset();
            goto explore;
        } else if (ssry.call(curr_representation)) {
            /* LOG("[Brain] forced exploration : v=" + std::to_string(ssry.get_v())); */
            forced_exploration = 0;
            goto explore;
        }

        // === GOAL-DIRECTED BEHAVIOUR ====================================

        // --- current target plan

        // check: current trg plan
        if (goalmd.is_active()) {
            /* LOG("[Brain] active goal plan"); */
            trg_plan_end = 0;
            std::pair<std::array<float, 2>, bool> progress = \
                goalmd.step_plan(collision > 0.0f);

            // keep going
            if (progress.second) {
                this->action = progress.first;
                goto final;
            }
            // end or fail -> random walk
            forced_exploration = 0;
            /* LOG("[Brain] end or fail -> random walk"); */
        }

        // time since the last trg plan ended
        trg_plan_end++;

        // --- new target plan: REWARD

        // :reward object | new reward trg index wrt the fine space
        tmp_trg_idx = rwobj.update(circuits.get_da_weights(),
                                   space_fine, trigger);

        if (tmp_trg_idx > -1) {

            // check new reward trg plan
            bool valid_plan = goalmd.update(tmp_trg_idx, true);

            // [+] reward trg plan
            if (valid_plan) {
                LOG("[Brain] valid goal plan");
                this->directive = "trg";
                std::pair<std::array<float, 2>, bool> progress = \
                    goalmd.step_plan(collision > 0.0f);
                if (progress.second) {
                    this->action = progress.first;
                    goto final;
                }
                forced_exploration = 0;
            }
            LOG("[Brain] invalid goal plan");
        }

        // === EXPLORATIVE BEHAVIOUR =======================================
explore:

        // check: collision
        if (collision > 0.0f) { this->directive = "new"; }
        else { this->directive = "continue"; }

        // :experience module
        expmd_res = expmd.call(directive);

        // check: plan to go to the open boundary
        if (expmd_res.second > -1) {
            this->action = attempt_boundary_plan(expmd_res.second);
            goto final;
        }
        this->action = expmd_res.first;

final:

        // ================================================================

        // make prediction
        make_prediction();

        /* LOG("[brain] action=" + std::to_string(action[0]) + ", " + \ */
        /*     std::to_string(action[1])); */
        return action;
    }

    std::string str() { return "Brain"; }
    std::string repr() { return "Brain"; }
    int len() { return space_fine.get_size(); }
    Eigen::VectorXf get_trg_representation() {
        Eigen::VectorXf trg_representation = \
            Eigen::VectorXf::Zero(space_fine.get_size());
        trg_representation(rwobj.trg_idx) = 1.;
        return trg_representation;
    }
    int get_trg_idx() { return rwobj.trg_idx; }
    std::array<float, 2> get_leaky_v() { return circuits.get_leaky_v(); }
    Eigen::VectorXf& get_representation_fine() { return curr_representation; }
    Eigen::VectorXf& get_representation_coarse()
        { return curr_representation_coarse; }
    std::array<float, 2> get_trg_position() {
        Eigen::MatrixXf centers = space_fine.get_centers();
        return {centers(rwobj.trg_idx, 0), centers(rwobj.trg_idx, 1)};
    }
    ExplorationModule& get_expmd() { return expmd; }
    std::string get_directive() { return directive; }
    int get_space_fine_size() { return space_fine.get_size(); }
    int get_space_coarse_size() { return space_coarse.get_size(); }
    int get_space_fine_count() { return space_fine.len(); }
    int get_space_coarse_count() { return space_coarse.len(); }
    std::vector<int> get_plan_idxs_fine() { return goalmd.get_plan_idxs_fine(); }
    std::vector<int> get_plan_idxs_coarse() { return goalmd.get_plan_idxs_coarse(); }
    std::array<float, 2> get_space_fine_position()
        { return space_fine.get_position(); }
    std::array<float, 2> get_space_coarse_position()
        { return space_coarse.get_position(); }
    Eigen::MatrixXf get_space_fine_centers() { return space_fine.get_centers(); }
    Eigen::MatrixXf get_space_coarse_centers() { return space_coarse.get_centers(); }
    Eigen::VectorXf get_da_weights() { return circuits.get_da_weights(); }
    Eigen::VectorXf get_bnd_weights() { return circuits.get_bnd_weights(); }
    std::vector<std::array<std::array<float, 2>, 2>> make_space_fine_edges()
        { return space_fine.make_edges(); }
    std::vector<std::array<std::array<float, 2>, 2>> make_space_coarse_edges()
        { return space_coarse.make_edges(); }
    Eigen::VectorXf get_edge_representation()
        { return expmd.get_edge_representation(); }
    void reset() {
        goalmd.reset();
        circuits.reset();
        space_fine.reset();
        space_coarse.reset();
    }

};


/* ========================================== */
/* ========================================== */


namespace pcl {



struct Reward {
    float x, y, sigma;
    int t, count, pause;
    bool available;

    Reward(float x, float y, float sigma, int pause):
        x(x), y(y), sigma(sigma), t(0), count(0), pause(pause),
        available(true) {}
    ~Reward() {}

    float call(float x, float y) {

        if (t != 0) {
            // max between 0 and t-1
            this->t = std::max(0, t-1);
            this->available = false;
            /* return 0.0f; */
        } else {
            this->available = true;
        }

        // calculate distance
        float dx = x - this->x;
        float dy = y - this->y;
        float dist = std::exp(-std::sqrt(dx * dx + dy * dy) / sigma);

        if (dist > 0.1) {
            this->t = pause;
            this->count++;
            /* LOG("[+] Reward: " + std::to_string(dist)); */
            return dist;
        }
        return 0.0f;
    }

    bool is_available() { return available; }
};


void local_log(const std::string& msg) {
    std::cout << msg << std::endl;
}

int simple_env(int pause = 20, int duration = 3000, float bnd_w = 0.0f) {

    // settings
    float SPEED = 1.0f;
    std::array<float, 2> BOUNDS = {0.0f, 50.0f};
    int N = std::pow(25, 2);

    // SPACE
    std::vector<GCL_REF> gcn_layers;
    gcn_layers.push_back(GCL_REF(0.04, 0.1));
    gcn_layers.push_back(GCL_REF(0.04, 0.07));
    gcn_layers.push_back(GCL_REF(0.04, 0.03));
    gcn_layers.push_back(GCL_REF(0.04, 0.005));
    GCN_REF gcn = GCN_REF(gcn_layers);
    PCNN space = PCNN(N, gcn.len(), 10.0f, 1.4f, 0.01f, 0.2f, 0.7f,
                              5.0f, 5, gcn, 10.0f, 1, 3, "fine");
    PCNN space_coarse = PCNN(N, gcn.len(),
                                     10.0f, 1.4f, 0.01f, 0.2f, 0.7f,
                                     5.0f, 0.95, gcn, 20.0f, 1, 3, "coarse");

    // MODULATION
    // name size lr threshold maxw tauv eqv minv
    BaseModulation da = BaseModulation("DA", N, 0.5f, 0.1f, 0.0f, 1.0f,
                                       2.0f, 0.0f, 0.0f, 0.01f);
    BaseModulation bnd = BaseModulation("BND", N, 0.9f, 0.0f, 0.0f, 1.0f,
                                        2.0f, 0.0f, 0.0f, 0.01f);
    StationarySensory ssry = StationarySensory(N, 100.0f, 0.2f, 0.5f);
    Circuits circuits = Circuits(da, bnd, 0.5f);

    // EXPERIENCE MODULE & BRAIN
    DensityPolicy dpolicy = DensityPolicy(0.5f, 40.0f, 0.5f, 20.0f, 1);
    ExplorationModule expmd = ExplorationModule(SPEED, circuits, space, 1.0f);
    /* Brain brain = Brain(circuits, space, space_coarse, expmd, ssry, dpolicy, */
    /*                     SPEED, SPEED * 2.0f, 5); */
    Brain brain = Brain(0.1f, 0.1f, N, N, 0.01f, 0.01f, SPEED, 0.01f,
                            0.5f, 0.0f, 0.2f, 0.7f, 10.0f, 1, 3,
                            0.5f, 0.0f, 0.2f, 0.7f, 20.0f,
                            0.5f, 0.1f, 0.0f, 0.2f,
                            0.5f, 0.0f, 0.2f,
                            100.0f, 0.2f,
                            0.5f, 1,
                            0.5f, 40.0f, 0.5f, 20.0f,
                            1.0f, 100,
                            100, 50, 0.3f);

    // simulation settigns
    Reward reward = Reward(10.0f, 10.0f, 1.0f, pause);
    std::array<float, 2> position = {5.0f, 5.0f};
    std::array<float, 2> velocity = {0.0f, 0.0f};
    float collision_ = 0.0f;
    float reward_ = 0.0f;
    bool trigger = false;

    std::vector<std::array<float, 2>> trajectory = {{0.0f}};

    int num_collision = 0;

    // log
    local_log("%Simple environment simulation");
    local_log("%Duration: " + std::to_string(duration));
    local_log("%Pause: " + std::to_string(pause));

    // loop
    local_log("<start>");
    for (int i=0; i < duration; i++) {

        /* std::cout << "[" << i << " | " << duration << "]\r"; */

        // move
        position[0] += velocity[0];
        position[1] += velocity[1];

        // check: collision
        if (position[0] < BOUNDS[0] || position[0] > BOUNDS[1]) {
            velocity[0] = -velocity[0];
            position[0] += 2.0f * velocity[0];
            collision_ = 1.0f;
            /* LOG("------------>>> collision"); */
            num_collision++;
        } else if (position[1] < BOUNDS[0] || position[1] > BOUNDS[1]) {
            velocity[1] = -velocity[1];
            position[1] += 2.0f * velocity[1];
            collision_ = 1.0f;
            /* LOG("------------>>> collision"); */
            num_collision++;
        } else {
            collision_ = 0.0f;
        }

        // check: reward
        reward_ = reward.call(position[0], position[1]);
        trigger = reward.is_available();

        // step
        velocity = brain.call(velocity, collision_, reward_, trigger);
    }


    local_log("<end>");

    local_log("\n------------");
    local_log("#count: " + std::to_string(space.len()));
    local_log("#rw_count: " + std::to_string(reward.count));
    local_log("#collision: " + std::to_string(num_collision));
    local_log("------------\n");

    return reward.count;

};


void test_env(int duration = 3000) {

    int pause1 = 4000;
    int count1 = simple_env(pause1, duration);

    int pause2 = 20;
    int count2 = simple_env(pause2, duration);

    LOG(" ");
    LOG("----------------");
    LOG(" ");
    LOG("Test 1, pause=" + std::to_string(pause1) + " | count: " + std::to_string(count1));
    LOG("Test 2, pause=" + std::to_string(pause2) + " | count: " + std::to_string(count2));
};


void test_bnd(int duration = 3000) {

    std::array<float, 3> bw = {0.0f, -1.0f, 1.0f};

    for (auto& w : bw) {
        LOG("##############################################");
        LOG("Test boundary sensor: " + std::to_string(w));
        LOG(" ");
        int count1 = simple_env(100000, duration, w);
        LOG(" ");
    }
};

};


/* #endif // MAINLIB_HPP */
