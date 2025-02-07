#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
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

#define SPACE utils::logging.space
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

// blank log function
void LOG(const std::string& msg) {
    std::cout << msg << std::endl;
}


// DEBUGGING logs

bool DEBUGGING = false;

void set_debug(bool flag) {
    DEBUGGING = flag;
}

void DEBUG(const std::string& msg) {
    if (DEBUGGING) {
        std::cout << "[DEBUG] " << msg << std::endl;
    }
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

    // center: (0, 0)
    // side: 1
    std::array<std::array<float, 2>, 6> centers;
    std::array<size_t, 6> index = {0, 1, 2, 3, 4, 5};

    // @brief check whether p is within the inner circle
    bool apothem_checkpoint(float x, float y) {
        float dist = std::sqrt(std::pow(x, 2) + \
                               std::pow(y, 2));
        bool within = dist < 0.86602540378f;
        if (within) {
            return true;
        } else {
            return false;
        }
    }

    // @brief wrap the point to the boundary
    int wrap(float x, float y, float *p_new_x, float *p_new_y) {

        // reflect the point p to r wrt the center (0, 0)
        float rx = x * -1;
        float ry = y * -1;

        /* printf("Reflection: rx %f, ry %f\n", rx, ry); */

        // calculate and sort the distances to the centers
        std::array<float, 6> distances;
        for (int i = 0; i < 6; i++) {
            distances[i] = std::sqrt(std::pow(centers[i][0] - rx,
                                              2) + \
                                     std::pow(centers[i][1] - ry,
                                              2));
        }

        // Sort the index array based on the
        // values in the original array
        std::sort(index.begin(), index.end(),
            [&distances](const size_t& a, const size_t& b) {
                return distances[a] < distances[b];
            }
        );

        float ax = centers[index[0]][0];
        float ay = centers[index[0]][1];
        float bx = centers[index[1]][0];
        float by = centers[index[1]][1];
        float mx = (ax + bx) / 2.0f;
        float my = (ay + by) / 2.0f;

        //
        /* printf("A: %f, %f\n", ax, ay); */
        /* printf("B: %f, %f\n", bx, by); */
        // calculate the intersection s between ab and ro
        float sx, sy;
        if (utils::get_segments_intersection(
            ax, ay, bx, by, rx, ry,
            0.0f, 0.0f, &sx, &sy)) {
        } else {
            // checkpoint: no intersection,
            // point is inside the hexagon
            /* LOG("[+] no intersection"); */
            return 0;
        }

        // reflect the point r wrt the intersection s
        /* *p_new_x = 2 * sx - rx; */
        /* *p_new_y = 2 * sy - ry; */
        rx = 2 * sx - rx;
        ry = 2 * sy - ry;

        // reflect wrt the line s-center
        std::array<float, 2> z;
        if (sy > 0) {
             z = utils::reflect_point_over_segment(
                rx, ry, 0.0f, 0.0f, mx, my);
        } else {
            z = utils::reflect_point_over_segment(
                rx, ry, mx, my, 0.0f, 0.0f);
        }

        *p_new_x = z[0];
        *p_new_y = z[1];

        return 1;
    }

public:

    Hexagon() {
        centers[0] = {-0.5f, -0.86602540378f};
        centers[1] = {0.5f, -0.86602540378f};
        centers[2] = {1.0f, 0.0f};
        centers[3] = {0.5f, 0.86602540378f};
        centers[4] = {-0.5, 0.86602540378f};
        centers[5] = {-1.0f, 0.0f};

        /* LOG("[+] hexagon created"); */
    }

    ~Hexagon() {} // LOG("[-] hexagon destroyed"); }

    // @brief call: apply the boundary conditions
    std::array<float, 2> call(float x, float y) {

        float new_x, new_y;

        if (!apothem_checkpoint(x, y)) {
            if (wrap(x, y, &new_x, &new_y)) {
                /* LOG("[+] point wrapped to boundary"); */
                return {new_x, new_y};
            } else {
                /* LOG("[+] point within the hexagon"); */
                return {x, y};
            }
        } else {
            /* LOG("[+] within the apothem"); */
            return {x, y};
        }
    }

    std::string str() { return "hexagon"; }
    std::string repr() { return str(); }
    std::array<std::array<float, 2>, 6> get_centers() {
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


class VelocitySpace {

    /* void update_node_angles(int idx) { */

    /*     // update the neighbors */
    /*     for (int j = 0; j < size; j++) { */
    /*         if ( j == idx ) { continue; } */
    /*         if (connectivity(idx, j) == 1.0) { */
    /*             nodes_max_angle(j) = utils::calculate_angle_gap( */
    /*                 j, centers, connectivity); */
    /*         } */
    /*     } */

    /*     // update the node itself */
    /*     nodes_max_angle(idx) = utils::calculate_angle_gap( */
    /*         idx, centers, connectivity); */
    /* } */

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
    std::array<float, 2> position;
    int size;
    /* std::vector<std::array<float, 2>> trajectory; */
    float threshold;

    std::vector<std::array<int, 2>> blocked_edges;

    VelocitySpace(int size, float threshold)
        : size(size), threshold(threshold) {
        centers = Eigen::MatrixXf::Constant(size, 2, -9999.0f);
        connectivity = Eigen::MatrixXf::Zero(size, size);
        weights = Eigen::MatrixXf::Zero(size, size);
        position = {0.00124789f, 0.00147891f};
        blocked_edges = {};
        nodes_max_angle = Eigen::VectorXf::Zero(size);
        node_degree = Eigen::VectorXf::Zero(size);
    }

    // CALL
    std::array<float, 2> call(const std::array<float, 2>& v) {
        position[0] += v[0];
        position[1] += v[1];
    }

    void update(int idx, int current_size, bool update_center = true) {

        // update the centers
        if (update_center) {
            centers.row(idx) = Eigen::Vector2f(
                position[0], position[1]);
        }

        // add recurrent connections
        for (int j = 0; j < current_size; j++) {

            // check if the edge is blocked
            bool blocked = false;
            for (auto& edge : blocked_edges) {
                if (edge[0] == idx && edge[1] == j) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) { continue; }

            // check if the nodes exist
            if (centers(idx, 0) < -999.0f || \
                centers(j, 0) < -999.0f || \
                idx == j) {
                continue;
            }
            float dist = std::sqrt(
                (centers(idx, 0) - centers(j, 0)) *
                (centers(idx, 0) - centers(j, 0)) +
                (centers(idx, 1) - centers(j, 1)) *
                (centers(idx, 1) - centers(j, 1))
            );
            if (dist < threshold) {
                this->weights(idx, j) = dist;
                this->weights(j, idx) = dist;
                this->connectivity(idx, j) = 1.0;
                this->connectivity(j, idx) = 1.0;
            }
        }

        // update the node angles
        /* update_node_angles(idx); */
        update_node_degree(idx, current_size);
    }

    void remap_center(int idx, int size, std::array<float, 2> displacement) {
        centers(idx, 0) += displacement[0];
        centers(idx, 1) += displacement[1];

        update(idx, size, false);
    }

    void add_blocked_edge(int i, int j) {

        // check if the edge is already blocked
        for (auto& edge : blocked_edges) {
            if (edge[0] == i && edge[1] == j) {
                LOG("[VS] edge already blocked: " + std::to_string(i) + ", " + \
                    std::to_string(j));

                // show the blocked edges in the connectivity matrix
                LOG("\t::->" + std::to_string(connectivity(i, j)));
                return;
            }
        }

        blocked_edges.push_back({i, j});
        blocked_edges.push_back({j, i});

        // remove the edge from the connectivity matrix
        this->connectivity(i, j) = 0.0;
        this->connectivity(j, i) = 0.0;
        this->weights(i, j) = 0.0;
        this->weights(j, i) = 0.0;

        LOG("[VS] added blocked edge: " + std::to_string(i) + ", " + \
            std::to_string(j));
    }

    /* std::vector<std::array<float, 2>> get_trajectory() { */
    /*     return trajectory; } */
    Eigen::MatrixXf& get_centers(bool nonzero=false) {

        if (!nonzero) { return centers; }

        Eigen::MatrixXf centers_nonzero = Eigen::MatrixXf::Zero(size, 2);
        for (int i = 0; i < size; i++) {
            if (centers(i, 0) > -999.0f) {
                centers_nonzero.row(i) = centers.row(i);
            }
        }
        return centers_nonzero;
    }
    /* Eigen::MatrixXf& get_connectivity() { return connectivity; } */
    /* Eigen::MatrixXf& get_connections() { return weights; } */
    /* Eigen::VectorXf& get_nodes_max_angle() { return nodes_max_angle; } */
    /* std::array<float, 2> get_position() { return position; } */

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
        return edges;  // Add this line to return the edges vector
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

    int calculate_closest_index(const std::array<float, 2>& c) {
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

};


/* ========================================== */
/* ============== INPUT LAYER =============== */
/* ========================================== */

// === purely hexagonal grid network ===

class GridHexLayer {

public:

    GridHexLayer(float sigma, float speed,
                 float offset_dx = 0.0f,
                 float offset_dy = 0.0f):
        sigma(sigma), speed(speed), hexagon(Hexagon()){

        // make matrix
        /* LOG("[+] GridHexLayer created"); */

        // apply the offset by stepping
        if (offset_dx != 0.0f && offset_dy != 0.0f) {
            call({offset_dx, offset_dy});
        }
    }

    ~GridHexLayer() {} //LOG("[-] GridHexLayer destroyed"); }

    // @brief call the GridLayer with a 2D input
    Eigen::VectorXf \
    call(const std::array<float, 2>& v) {

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

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& v) {

        /* Eigen::MatrixXf new_positions = Eigen::MatrixXf::Zero(N, 2); */
        std::array<std::array<float, 2>, 25> new_positions;
        /* new_positions.col(0) = positions.col(0) + speed * \ */
            /* v[0]; */
        /* new_positions.col(1) = positions.col(1) + speed * \ */
            /* v[1]; */
        for (int i = 0; i < N; i++) {
            new_positions[i][0] = positions[i][0] + \
                speed * (v[0] - positions[i][0]);
            new_positions[i][1] = positions[i][1] + \
                speed * (v[1] - positions[i][1]);
        }

        // check boundary conditions
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = hexagon.call(
                new_positions[i][0], new_positions[i][1]);
            new_positions[i][0] = new_position[0];
            new_positions[i][1] = new_position[1];
            /* std::array<float, 2> new_position = hexagon.call( */
            /*     new_positions(i, 0), new_positions(i, 1)); */
            /* new_positions(i, 0) = new_position[0]; */
            /* new_positions(i, 1) = new_position[1]; */
        }

        // compute the activation
        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        float dist_squared;
        for (int i = 0; i < N; i++) {
            /* dist_squared = std::pow(new_positions(i, 0), 2) + \ */
            /*     std::pow(new_positions(i, 1), 2); */
            dist_squared = std::pow(new_positions[i][0], 2) + \
                std::pow(new_positions[i][1], 2);
            yfwd(i) = std::exp(-dist_squared / sigma);
        }

        return yfwd;
    }

    int len() const { return N; }
    std::string str() const { return "GridHexLayer"; }
    std::string repr() const { return "GridHexLayer"; }
    std::array<std::array<float, 2>, 25> get_positions()
    { return positions; }
    std::array<std::array<float, 2>, 25> get_centers()
    { return basis; }
    void reset(std::array<float, 2> v) {
        this->positions = basis;
        call(v);
    }


private:

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

    /* std::array<float, 7> y; */
    Eigen::VectorXf y = Eigen::VectorXf::Zero(25);
    float sigma;
    float speed;
    Hexagon hexagon;

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
            y(i) = std::exp(-dist_squared / sigma);
        }
    }

};


class GridHexNetwork {

public:

    GridHexNetwork(std::vector<GridHexLayer> layers)
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

        /* LOG("[+] GridHexNetwork created"); */
    }

    ~GridHexNetwork() {} // LOG("[-] GridHexNetwork destroyed"); }

    Eigen::VectorXf call(const std::array<float, 2>& x) {
        for (int i = 0; i < num_layers; i++) {
            y.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].call(x);
        }

        return y;
    }

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& x) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {
            yfwd.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].fwd_position(x);
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

    void reset(std::array<float, 2> v) {
        for (int i = 0; i < num_layers; i++) {
            layers[i].reset(v);
        }
    }

private:
    std::vector<GridHexLayer> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;
    Eigen::MatrixXf positions;
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
        std::vector<float> linex = utils::linspace_vec(
                        bounds[0], bounds[1], GCL_SIZE_SQRT,
                        true, false);
        std::vector<float> liney = utils::linspace_vec(
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

    ~GridLayerSq() {} // LOG("[-] GridLayer destroyed"); }

    // CALL
    std::array<float, GCL_SIZE> call(
        const std::array<float, 2>& v) {

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

    std::array<float, GCL_SIZE> simulate(
        const std::array<float, 2>& v,
        std::array<std::array<float, 2>, GCL_SIZE>& sim_gc_positions) {

        std::array<std::array<float, 2>, GCL_SIZE> new_positions;
        for (int i = 0; i < GCL_SIZE; i++) {
            new_positions[i][0] = sim_gc_positions[i][0] + \
                speed * v[0];
            new_positions[i][1] = sim_gc_positions[i][1] + \
                speed * v[1];
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

    // SIMULATE
    std::array<float, GCL_SIZE> simulate_one_step(
        const std::array<float, 2>& v) {

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

    int len() const { return GCL_SIZE; }
    std::string str() const { return "GridLayerSq"; }
    std::string repr() const { return "GridLayerSq"; }
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

        LOG("[+] GridNetwork created");
    }

    ~GridNetworkSq() {} // LOG("[-] GridNetworkSq destroyed"); }

    Eigen::VectorXf& call(const std::array<float, 2>& x) {

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

    Eigen::VectorXf& simulate(
        const std::array<float, 2>& v,
        std::vector<std::array<std::array<float, 2>, GCL_SIZE>>& sim_gc_positions) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {

            // Convert the output of layers[i].call(x) to
            // an Eigen::VectorXf
            Eigen::VectorXf layer_output = \
                Eigen::Map<const Eigen::VectorXf>(
                    /* layers[i].fwd_position(v).data(), */
                    layers[i].simulate(v, sim_gc_positions[i]).data(),
                    GCL_SIZE);

            // Assign the converted vector to
            // the corresponding segment of y
            yfwd.segment(i * GCL_SIZE, GCL_SIZE) = layer_output;
        }

        return yfwd;
    }

    Eigen::VectorXf& simulate_one_step(const std::array<float, 2>& v) {

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

    int len() const { return N; }
    int get_num_layers() const { return num_layers; }
    std::string str() const { return "GridNetworkSq"; }
    std::string repr() const {
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

private:
    std::vector<GridLayerSq> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;
};


/* ========================================== */
/* ================= PCNN =================== */
/* ========================================== */


class PCNNgridhex {
public:
    PCNNgridhex(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         GridHexNetwork xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(std::move(xfilter)),
        name(name) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        centers = Eigen::MatrixXf::Ones(N, 2) * -1000.0f;
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        ach = 1.0f;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};

        LOG("[+] PCNNgridhex created");
    }

    ~PCNNgridhex() { LOG("[-] PCNNgridhex destroyed"); }

    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v,
                          const bool frozen = false,
                          const bool traced = false) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        /* u = Wff * x_filtered + pre_x; */
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = \
            Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        /* u = utils::gaussian_distance(x_filtered, Wff, sigma); */

        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);

        // update the trace
        if (traced) {
            trace = (1 - trace_tau) * trace + trace_tau * u;
        }

        // update model
        /* if (!frozen) { */
        /*     update(x_filtered); */
        /* } */

        /* return u; */
        return std::make_pair(u, x_filtered);
    }

    // @brief update the model
    void update(float x = -1.0, float y = -1.0) {

        make_indexes();

        // exit: a fixed neuron is above threshold
        if (check_fixed_indexes() != -1) {
            DEBUG("!Fixed index above threshold");
            /* printf("(-)Fixed index above threshold\n"); */
           return void();
        };

        // exit: there are no free neurons
        if (free_indexes.size() == 0) {
            DEBUG("!No free neurons");
            return void();
        };

        // pick new index
        int idx = utils::random.get_random_element_vec(
                                        free_indexes);
        DEBUG("Picked index: " + std::to_string(idx));

        // determine weight update
        Eigen::VectorXf dw = x_filtered - \
            Wff.row(idx).transpose();

        // trim the weight update
        /* delta_wff = (dw.array() > 0.01).select(dw, 0.0f); */

        delta_wff = dw.norm();

        if (delta_wff > 0.0) {

            DEBUG("delta_wff: " + std::to_string(delta_wff));

            // update weights
            Wff.row(idx) += dw.transpose();

            // calculate the similarity among the rows
            float similarity = \
                utils::max_cosine_similarity_in_rows(
                    Wff, idx);

            // check repulsion (similarity) level
            if (similarity > (rep_threshold * ach)) {
                Wff.row(idx) = Wffbackup.row(idx);
                /* printf("(-)Repulsion [%f]{%f}\n", similarity, */
                       /* rep_threshold); */
                return void();
            }

            // update count and backup
            cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);
            /* printf("(:)cell_count: %d [%f]\n", cell_count, */
                   /* similarity); */

            // update recurrent connections
            update_recurrent();

            // record new center
            centers.row(idx) = Eigen::Vector2f(x, y);
        }

    }

   /* void fwd_ext(const std::array<float, 2>& x) {} */
   /* /1* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { *1/ */
   /* /1*      return call(x, true, false); *1/ */
   /* /1*  } *1/ */
   /* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { */
   /*      std::pair<Eigen::VectorXf, Eigen::VectorXf> ans = call(x); */
   Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) {

        // pass the input through the filter layer
        x_filtered = xfilter.fwd_position(x);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = \
            Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        return utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);
    }

   Eigen::VectorXf fwd_int(const Eigen::VectorXf& a) {
        return Wrec * a + pre_x;
    }

    void reset() {
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
    }

    // Getters
    int len() const { return cell_count; }
    int get_size() const { return N; }
    std::string str() const { return "PCNNgridhex." + name; }
    std::string repr() const {
        return "PCNNgridhex(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rep_threshold) + \
            std::to_string(rec_threshold) + \
            std::to_string(num_neighbors) + \
            std::to_string(trace_tau) + ")";
    }
    Eigen::VectorXf get_activation() const { return u; }
    Eigen::VectorXf get_activation_gcn() const {
        return xfilter.get_activation(); }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf get_wrec() const { return Wrec; }
    Eigen::MatrixXf get_connectivity() const { return connectivity; }
    Eigen::MatrixXf get_centers(bool nonzero = false) const {

        if (!nonzero) {
            return centers;
        }

        // filter out the non-zero centers
        std::vector<int> idxs;
        for (int i = 0; i < N; i++) {
            if (centers.row(i).sum() != 0.0) {
                idxs.push_back(i);
            }
        }
        Eigen::MatrixXf centers = Eigen::MatrixXf::Zero(idxs.size(), 2);
        for (int i = 0; i < idxs.size(); i++) {
            centers.row(i) = this->centers.row(idxs[i]);
        }
        return centers;
    }
    Eigen::VectorXf get_trace() const { return trace; }
    float get_delta_update() const { return delta_wff; }
    Eigen::MatrixXf get_positions_gcn() {
        return xfilter.get_positions();
    }
    void reset_gcn(std::array<float, 2> v) {
        xfilter.reset(v);
    }
    // @brief modulate the density of new PCs
    void ach_modulation(float ach) {
        this->ach = ach;
    }

private:
    // parameters
    const int N;
    const int Nj;
    const float gain;
    const float offset;
    const float clip_min;
    const float threshold;
    const float rep_threshold;
    const float rec_threshold;
    const int num_neighbors;
    const float trace_tau;
    const std::string name;

    float ach;

    GridHexNetwork xfilter;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    float delta_wff;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf connectivity;
    /* Eigen::MatrixXf centers; */
    Eigen::VectorXf mask;
    std::vector<int> fixed_indexes;
    std::vector<int> free_indexes;
    int cell_count;
    Eigen::VectorXf u;
    Eigen::VectorXf x_filtered;
    Eigen::VectorXf trace;
    Eigen::VectorXf pre_x;
    Eigen::MatrixXf centers;

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

        if (max_u < (threshold * ach) ) { return -1; }
        else {
            DEBUG("Fixed index above threshold: " + \
                std::to_string(max_u) + \
                " [" + std::to_string(threshold) + "]");
            /* printf("(-)Fixed index above threshold [u=%f]{%f}\n", */
                   /* max_u, threshold); */
            return max_idx; };
    }

    // @brief Quantify the indexes.
    void make_indexes() {

        free_indexes.clear();
        for (int i = 0; i < N; i++) {
            if (Wff.row(i).sum() > threshold) {
                fixed_indexes.push_back(i);
            } else {
                free_indexes.push_back(i);
            }
        }
    }

    // @brief calculate the recurrent connections
    void update_recurrent() {
        // connectivity matrix
        connectivity = utils::connectivity_matrix(
            Wff, rec_threshold
        );

        // similarity
        Wrec = utils::cosine_similarity_matrix(Wff);

        // weights
        Wrec = Wrec.cwiseProduct(connectivity);
    }

};


class PCNNbase {
public:
    PCNNbase(int N, int Nj, float gain, float offset,
             float clip_min, float threshold,
             float rep_threshold,
             float rec_threshold,
             int num_neighbors,
             int length,
             std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        length(length),
        name(name) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        centers = Eigen::MatrixXf::Zero(N, 2);
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        xfilter = {0.0f};


        ach = 1.0;

        this->fixed_centers = utils::generate_lattice(N, length);

        // make recurrent connections based on the fixed centers
        // 1. loop through the fixed centers
        // 2. calculate the distance between the fixed centers
        // 3. if the distance is less than the threshold, make a connection
        // 4. make the connection symmetric
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    float dist = std::sqrt(
                        (fixed_centers(i, 0) - fixed_centers(j, 0)) *
                        (fixed_centers(i, 0) - fixed_centers(j, 0)) +
                        (fixed_centers(i, 1) - fixed_centers(j, 1)) *
                        (fixed_centers(i, 1) - fixed_centers(j, 1))
                    );
                    if (dist < rec_threshold) {
                        Wrec(i, j) = dist;
                        Wrec(j, i) = dist;
                    }
                }
            }
        }
    }

    ~PCNNbase() { LOG("[-] PCNN destroyed"); }

    // CALL
    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v,
                          const bool frozen = false,
                          const bool traced = true) {

        xfilter[0] = v[0];
        xfilter[1] = v[1];

        // give a position v, calculate the activation
        // as a gaussian distance
        Eigen::VectorXf y = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < N; i++) {
            y(i) = std::exp(-((xfilter[0] - fixed_centers(i, 0)) *
                              (xfilter[0] - fixed_centers(i, 0)) +
                              (xfilter[1] - fixed_centers(i, 1)) *
                              (xfilter[1] - fixed_centers(i, 1))));
        }

        y = utils::generalized_sigmoid_vec(y, offset,
                                           gain, clip_min);

        return std::make_pair(y, x_filtered);
    }

    // @brief update the model
    void update(float x = -1.0, float y = -1.0) {};

    Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) {

        Eigen::VectorXf z = Eigen::VectorXf::Zero(N);

        for (int i = 0; i < N; i++) {
            z(i) = std::exp(-((x[0] - fixed_centers(i, 0)) *
                              (x[0] - fixed_centers(i, 0)) +
                              (x[1] - fixed_centers(i, 1)) *
                              (x[1] - fixed_centers(i, 1))));
        }

        return utils::generalized_sigmoid_vec(z, offset,
                                             gain, clip_min);

        // maybe use cosine similarity?
        /* std::pair<Eigen::VectorXf, Eigen::VectorXf> res = \ */
        /*     call(v, true, false); */

        /* return z; */
    }

   Eigen::VectorXf fwd_int(const Eigen::VectorXf& a) {
        return Wrec * a + pre_x;
    }

    void reset() {
        u = Eigen::VectorXf::Zero(N);
        trace = Eigen::VectorXf::Zero(N);
        pre_x = Eigen::VectorXf::Zero(N);
    }

    // Getters
    int len() const { return N; }
    int get_size() const { return N; }
    std::string str() const { return "PCNNbase." + name; }
    std::string repr() const {
        return "PCNNbase(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rec_threshold) + \
            std::to_string(num_neighbors) + ")";
    }
    void set_xfilter(std::array<float, 2> x) {
        xfilter[0] = x[0];
        xfilter[1] = x[1];
    }
    Eigen::VectorXf get_activation() const { return u; }
    std::array<float, 2> get_activation_gcn() const {
        return xfilter; }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf get_wrec() const { return Wrec; }
    Eigen::MatrixXf get_connectivity() const { return connectivity; }
    Eigen::MatrixXf get_centers(bool nonzero = false) const {
        return fixed_centers;
    }
    Eigen::VectorXf get_trace() const { return trace; }
    float get_delta_update() const { return delta_wff; }
    std::array<float, 2> get_positions_gcn() {
        return xfilter;
    }
    Eigen::MatrixXf get_basis() {
        return fixed_centers;
    }
    void reset_gcn(std::array<float, 2> v) {
        xfilter = v;
    }

    // @brief modulate the density of new PCs
    void ach_modulation(float ach) {
        this->ach = ach;
    }

private:
    // parameters
    const int N;
    const int Nj;
    const int length;
    const float gain;
    const float offset;
    const float clip_min;
    const float threshold;
    const float rep_threshold;
    const float rec_threshold;
    const int num_neighbors;
    const std::string name;
    Eigen::MatrixXf fixed_centers;

    float ach;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    float delta_wff;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf connectivity;
    /* Eigen::MatrixXf centers; */
    Eigen::VectorXf mask;
    std::vector<int> fixed_indexes;
    std::vector<int> free_indexes;
    int cell_count;
    Eigen::VectorXf u;
    Eigen::VectorXf x_filtered;
    Eigen::VectorXf trace;
    Eigen::VectorXf pre_x;
    Eigen::MatrixXf centers;
    std::array<float, 2> xfilter;

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

        if (max_u < (threshold * ach) ) { return -1; }
        else {
            DEBUG("Fixed index above threshold: " + \
                std::to_string(max_u) + \
                " [" + std::to_string(threshold) + "]");
            return max_idx; };
    }

    // @brief Quantify the indexes.
    void make_indexes() {

        free_indexes.clear();
        for (int i = 0; i < N; i++) {
            if (Wff.row(i).sum() > threshold) {
                fixed_indexes.push_back(i);
            } else {
                free_indexes.push_back(i);
            }
        }
    }

    // @brief calculate the recurrent connections
    void update_recurrent() {
        // connectivity matrix
        connectivity = utils::connectivity_matrix(
            Wff, rec_threshold
        );

        // similarity
        Wrec = utils::cosine_similarity_matrix(Wff);

        // weights
        Wrec = Wrec.cwiseProduct(connectivity);
    }

};


class PCNNsqv2 {
    // parameters
    const int N;
    const int Nj;
    const float offset;
    const float clip_min;
    const int num_neighbors;
    const std::string name;
    const float threshold_const;
    const float rep_threshold_const;
    const float gain_const;

    float rep_threshold;
    float min_rep_threshold = 0.94f;
    float threshold;
    float gain;

    /* float ach; */

    GridNetworkSq xfilter;

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
    Eigen::VectorXf x_filtered;
    /* Eigen::MatrixXf gc_positions; */

    VelocitySpace vspace;

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

        /* if (max_u < (threshold * ach) ) { return -1; } */
        /* else { */
        /*     DEBUG("Fixed index above threshold: " + \ */
        /*         std::to_string(max_u) + \ */
        /*         " [" + std::to_string(threshold) + "]"); */
        /*     return max_idx; }; */
        return max_idx;
    }

    // @brief Quantify the indexes.
    void make_indexes() {

        free_indexes.clear();
        for (int i = 0; i < N; i++) {
            if (Wff.row(i).sum() > 0.0f) {
                fixed_indexes.push_back(i);
            } else {
                free_indexes.push_back(i);
            }
        }
    }

public:
    const float rec_threshold;

    PCNNsqv2(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors,
         GridNetworkSq& xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), gain_const(gain),
        offset(offset),
        clip_min(clip_min),
        rep_threshold(rep_threshold),
        rep_threshold_const(rep_threshold),
        rec_threshold(rec_threshold),
        threshold_const(threshold),
        threshold(threshold),
        num_neighbors(num_neighbors),
        xfilter(xfilter), name(name),
        vspace(VelocitySpace(N, rec_threshold)) {

        // Initialize the variables
        Wff = Eigen::MatrixXf::Zero(N, Nj);
        Wffbackup = Eigen::MatrixXf::Zero(N, Nj);
        Wrec = Eigen::MatrixXf::Zero(N, N);
        connectivity = Eigen::MatrixXf::Zero(N, N);
        /* centers = Eigen::MatrixXf::Zero(N, 2); */
        mask = Eigen::VectorXf::Zero(N);
        u = Eigen::VectorXf::Zero(N);
        /* trace = Eigen::VectorXf::Zero(N); */
        delta_wff = 0.0;
        x_filtered = Eigen::VectorXf::Zero(N);
        cell_count = 0;

        /* ach = 1.0; */

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};

        LOG("PCNN gain: " + std::to_string(gain));
        LOG("PCNN  rep: " + std::to_string(rep_threshold));
    }

    // CALL
    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v) {

        vspace.call(v);

        // pass the input through the filter layer
        x_filtered = xfilter.call(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered;
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(),
                                                          0.01);

        // maybe use cosine similarity?
        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);

        /* return u; */
        return std::make_pair(u, x_filtered);
    }

    // UPDATE
    void update() {

        make_indexes();

        // exit: a fixed neuron is above threshold
        if (check_fixed_indexes() != -1) {
            DEBUG("!Fixed index above threshold");
           return void();
        };

        // exit: there are no free neurons
        /* if (free_indexes.size() == 0) { */
        if (cell_count == N) {
            DEBUG("!No free neurons");
            return void();
        };

        /* // pick new index */
        /* int idx = utils::random.get_random_element_vec( */
        /*                                 free_indexes); */
        int idx = free_indexes[cell_count];
        DEBUG("Picked index: " + std::to_string(idx));

        // determine weight update
        Eigen::VectorXf dw = x_filtered - Wff.row(idx).transpose();

        // trim the weight update
        delta_wff = dw.norm();

        if (delta_wff > 0.0) {

            DEBUG("delta_wff: " + std::to_string(delta_wff));

            // update weights
            Wff.row(idx) += dw.transpose();

            // calculate the similarity among the rows
            float similarity = \
                utils::max_cosine_similarity_in_rows(
                    Wff, idx);

            // check repulsion (similarity) level
            if (similarity > rep_threshold) {
                Wff.row(idx) = Wffbackup.row(idx);
                return void();
            }

            // update count and backup
            cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);
            /* LOG("(:)cell_count: " + std::to_string(cell_count) + \ */
            /*     " [" + std::to_string(similarity) + ", idx=" + \ */
            /*     std::to_string(idx) + "]"); */

            // record new center
            vspace.update(idx, get_size());
            this->Wrec = vspace.weights;
            this->connectivity = vspace.connectivity;
        }
    }

    std::pair<Eigen::VectorXf, std::vector<std::array<std::array<float, 2>, GCL_SIZE>>>
    simulate(const std::array<float, 2>& v,
             std::vector<std::array<std::array<float, 2>, \
                GCL_SIZE>>& sim_gc_positions) {

        // pass the input through the filter layer
        x_filtered = xfilter.simulate(v, sim_gc_positions);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(),
                                                          0.01);

        // maybe use cosine similarity?
        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);
        return std::make_pair(u, xfilter.get_positions_vec());
    }

    Eigen::VectorXf& simulate_one_step(const std::array<float, 2>& v) {

        // pass the input through the filter layer
        x_filtered = xfilter.simulate_one_step(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        u = utils::generalized_sigmoid_vec(u, offset,
                                           gain, clip_min);
        return u;
    }

    void add_blocked_edge(int idx, int idx2) {
        vspace.add_blocked_edge(idx, idx2);
        this->Wrec = vspace.weights;
        this->connectivity = vspace.connectivity;
    }

    void remap(Eigen::VectorXf& block_weights,
               std::array<float, 2> velocity,
               float width, float magnitude) {

        if (magnitude < 0.00001f) { return; }

        /* LOG("--- remapping ---"); */
        /* LOG("magnitude: " + std::to_string(magnitude)); */
        /* LOG("displacement: " + std::to_string(displacement[0]) + ", " + std::to_string(displacement[1])); */
        /* LOG("curr position: " + std::to_string(vspace.position[0]) + ", " + std::to_string(vspace.position[1])); */

        float max_dist = 0.0f;
        for (int i = 0; i < N; i++) {

            if (vspace.centers(i, 0) < -900.0f || block_weights(i) > 0.0f) { continue; }

            std::array<float, 2> displacement = \
                {vspace.position[0] - vspace.centers(i, 0),
                 vspace.position[1] - vspace.centers(i, 1)};

            // gaussian activation function centered at zero
            float dist = std::exp(-std::sqrt(displacement[0] * displacement[0] + \
                                             displacement[1] * displacement[1]) / width);

            // cutoff
            if (dist < 0.1f) { continue; }

            max_dist = max_dist < dist ? dist : max_dist;

            // weight the displacement
            std::array<float, 2> gc_displacement = {displacement[0] * dist * magnitude - displacement[0],
                                                    displacement[1] * dist * magnitude - displacement[1]};

            // pass the input through the filter layer
            x_filtered = xfilter.simulate_one_step(gc_displacement);

            // update the weights & centers
            Wff.row(i) += x_filtered.transpose() - Wff.row(i).transpose();

            // check similarity
            float similarity = \
                utils::max_cosine_similarity_in_rows(
                    Wff, i);

            // check repulsion (similarity) level
            if (similarity > (dist * min_rep_threshold + (1 - dist) * min_rep_threshold) || \
                    std::isnan(similarity)) {
                Wff.row(i) = Wffbackup.row(i);
                continue;
            }

            /* LOG("remapping " + std::to_string(i) + " [" + std::to_string(dist) + "]" + \ */
            /*     " block=" + std::to_string(block_weights(i)) + \ */
            /*     " sim=" + std::to_string(similarity) + \ */
            /*     " [M=" + std::to_string(magnitude) + "]"); */

            // update backup and vspace
            Wffbackup.row(i) = Wff.row(i);
            vspace.remap_center(i, N, {displacement[0] * dist * magnitude,
                                       displacement[1] * dist * magnitude});
        }
        /* LOG("max_dist: " + std::to_string(max_dist)); */
    }

    int calculate_closest_index(const std::array<float, 2>& c)
    { return vspace.calculate_closest_index(c); }

    void reset() { u = Eigen::VectorXf::Zero(N); }

    // Getters
    int len() const { return cell_count; }
    int get_size() const { return N; }
    std::string str() const { return "PCNNsqv2." + name; }
    std::string repr() const {
        return "PCNNsqv2(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
            std::to_string(rec_threshold) + \
            std::to_string(num_neighbors) + ")";
    }
    Eigen::VectorXf get_activation() const { return u; }
    Eigen::VectorXf get_activation_gcn() const
        { return xfilter.get_activation(); }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf& get_wrec() { return Wrec; }
    std::vector<std::array<std::array<float, 2>, 2>> make_edges()
        { return vspace.make_edges(); }
    std::vector<std::array<std::array<float, 2>, 3>> make_edges_value(
        Eigen::MatrixXf& values) { return vspace.make_edges_value(values); }
    /* std::vector<std::array<float, 2>> get_trajectory() { */
    /*     return vspace.get_trajectory(); } */
    Eigen::MatrixXf& get_connectivity() { return connectivity; }
    Eigen::MatrixXf& get_centers(bool nonzero = false)
        { return vspace.get_centers(nonzero); }
    Eigen::VectorXf& get_node_degrees() { return vspace.node_degree; }
    float get_delta_update() const { return delta_wff; }
    /* Eigen::VectorXf& get_nodes_max_angle() */
    /*     {  return vspace.get_nodes_max_angle(); } */
    Eigen::MatrixXf get_positions_gcn()
        { return xfilter.get_positions(); }
    std::array<float, 2> get_position()
        { return vspace.position; }
    std::vector<std::array<std::array<float, 2>, GCL_SIZE>> get_gc_positions_vec()
        { return xfilter.get_positions_vec(); }
    void reset_gcn(std::array<float, 2> v) { xfilter.reset(v); }

    // @brief modulate the density of new PCs
    void modulate_gain(float mod) { this->gain = gain_const * mod; }
    void modulate_rep(float mod) {
        this->rep_threshold = rep_threshold_const * mod; 
        rep_threshold = rep_threshold < 1.0 ? rep_threshold : 0.99;
    }
    void modulate_threshold(float mod) {
        this->threshold = threshold_const * mod;
        threshold = threshold < 1.0 ? threshold : 0.99;
    }
    float get_gain() { return gain; }
    float get_rep() { return rep_threshold; }
    float get_threshold() { return threshold; }
    float calculate_angle_gap(int idx, Eigen::MatrixXf& centers,
                            Eigen::MatrixXf& connectivity)
        { return utils::calculate_angle_gap(idx, centers, connectivity); }

};


/* ========================================== */
/* ========= MODULATION & PROGRAMS ========== */
/* ========================================== */


class LeakyVariable1D {
public:
    std::string name;

    /* @brief Call the LeakyVariable with an input
     * @param input The input to the LeakyVariable
     * with `ndim` dimensions
    */
    float call(float x = 0.0,
               bool simulate = false) {

        // simulate
        if (simulate) {
            float z = v + (eq - v) * tau + x;
            if (z < min_v) {
                z = min_v;
            }
            return z;
        }

        v = v + (eq - v) * tau + x;

        /* if (v < min_v) { */
        /*     v = min_v; */
        /* } */
        if (v < min_v) {
            v = 0.0f;
        }
        return v;
    }

    LeakyVariable1D(std::string name, float eq,
                    float tau, float min_v = 0.0)
        : name(std::move(name)), eq(eq), tau(1.0/tau),
        v(eq), min_v(min_v){

        /* LOG("[+] LeakyVariable1D created with name: " + \ */
        /*     this->name); */
    }

    ~LeakyVariable1D() {
        /* LOG("[-] LeakyVariable1D destroyed with name: " + name); */
    }

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
    float output;
    std::string name;
    int size;
    float lr;
    float lt_p = 1.0f;
    float threshold;
    float mask_threshold;
    Eigen::VectorXf weights;
    Eigen::VectorXf mask;
    LeakyVariable1D leaky;

    // prediction
    Eigen::VectorXf prediction;
    float lr_pred;

public:
    float max_w;

    BaseModulation(std::string name, int size, float lr,
                   float threshold, float max_w = 1.0f,
                   float tau_v = 1.0f, float eq_v = 1.0f,
                   float min_v = 0.0f, float mask_threshold = 0.01f,
                   float lr_pred = 0.01f):
        name(name), size(size), lr(lr), threshold(threshold),
        max_w(max_w), mask_threshold(mask_threshold),
        leaky(LeakyVariable1D(name, eq_v, tau_v, min_v)),
        lr_pred(lr_pred) {
        weights = Eigen::VectorXf::Zero(size);
        mask = Eigen::VectorXf::Zero(size);
        prediction = Eigen::VectorXf::Zero(size);
    }

    ~BaseModulation() {}

    // CALL
    float call(const Eigen::VectorXf& u,
               float x = 0.0f, bool simulate = false) {

        // forward to the leaky variable
        float v = leaky.call(x, simulate);

        // update the weights
        if (!simulate) {
            for (int i = 0; i < size; i++) {

                // forward, delta from current input
                float ui = u[i] > threshold ? u[i] : 0.0;
                float dw = lr * v * ui;

                // backward, prediction error
                float pred_err = lr_pred * (prediction[i] - v * ui);

                // update weights
                weights[i] += lr * v * ui + pred_err;

                // clip the weights in (0, max_w)
                if (weights[i] < 0.0) {
                    weights[i] = 0.0;
                } else if (weights[i] > max_w) {
                    weights[i] = max_w;
                }
            }
        }

        // compute the output
        output = 0.0;
        for (int i = 0; i < size; i++) {
            output += weights[i] * u[i];
        }
        return output;
    }

    void make_prediction(const Eigen::VectorXf& u) {

        // update the prediction
        for (int i = 0; i < size; i++) {
            prediction[i] = weights[i] * u[i];
        }
    }

    float get_output() { return output; }
    /* std::vector<float> get_weights() { return weights; } */
    Eigen::VectorXf& get_weights() { return weights; }
    Eigen::VectorXf& make_mask() {

        // the nodes that are fresh in memory will affect action selection,
        // thus acting as a mask
        for (int i = 0; i < size; i++) {
            mask(i) = weights(i) > mask_threshold ? 0.0f : 1.0f;
        }
        return mask;
    }
    float get_leaky_v() { return leaky.get_v(); }
    std::string str() { return name; }
    std::string repr() { return name + "(1D)"; }
    int len() { return size; }
    void reset() { leaky.reset(); }
};


// === Memory ===

struct MemoryRepresentation {

    Eigen::VectorXf tape;
    Eigen::VectorXf mask;
    float decay;
    float mask_threshold;

    // call
    float call(Eigen::VectorXf& representation, bool simulate = false) {

        // evaluate without updating
        if (simulate) {
            // check if the norm is zero
            if (representation.norm() == 0.0f) { return 0.0f; }

            // dot product
            return tape.dot(representation) / representation.norm();
        }

        // update the memory
        update(representation);
        return 1.0f;
    }

    /* Eigen::VectorXf& make_mask() { */

    /*     // the nodes that are fresh in memory will affect action selection, */
    /*     // thus acting as a mask */
    /*     for (int i = 0; i < tape.size(); i++) { */
    /*         mask(i) = tape(i) > mask_threshold ? 0.0f : 1.0f; */
    /*     } */
    /*     return mask; */
    /* } */

    float get_max_value() { return tape.maxCoeff(); }

    MemoryRepresentation(int size, float decay, float mask_threshold):
        tape(Eigen::VectorXf::Zero(size)), decay(decay),
        mask_threshold(mask_threshold),
        mask(Eigen::VectorXf::Zero(size)) {}
    std::string str() { return "MemoryRepresentation"; }
    std::string repr() { return "MemoryRepresentation"; }

private:

    void update(Eigen::VectorXf& representation) {

        Eigen::Index maxIndex;
        representation.maxCoeff(&maxIndex);
        int max_idx = static_cast<int>(maxIndex);

        // decay the memory
        tape -= tape / decay;
        tape(max_idx) = 1.0f;
        /* LOG("[+] MemoryRepresentation: updated | max_idx: " + std::to_string(max_idx) + \ */
        /*     " | max_value: " + std::to_string(max_value)); */
    }

};


struct MemoryAction {

    std::array<float, ACTION_SPACE_SIZE> tape;
    float decay;

    // CALL
    float call(int idx, bool simulate=false) {

        // evaluate without updating
        if (simulate) {
            return tape[idx];
        }

        // decay the memory
        for (int i = 0; i < ACTION_SPACE_SIZE; i++) {
            tape[i] -= tape[i] / decay;
        }

        // update the action
        tape[idx] += (1.0f - tape[idx]) / decay;


        return tape[idx];
    }

    float get_max_value() {
        return *std::max_element(tape.begin(), tape.end());
    }

    MemoryAction(float decay): decay(decay) {}
    std::string str() { return "MemoryAction"; }
    std::string repr() { return "MemoryAction"; }
};


struct EpisodicMemory {

    Eigen::MatrixXf value_weights;
    std::vector<int> path = {-1};
    float v = 0.0f;
    float tau_decay;
    float tau_rise;
    float threshold = 0.2f;

    // CALL
    void call(float reward) {

        // update the value
        v += (reward - v) / tau_rise - v / tau_decay;

        /* LOG("[+] reward=" + std::to_string(reward)); */
        /* LOG("[+] threshold=" + std::to_string(threshold)); */
        /* LOG("[+] updating memory connections, v=" + std::to_string(v)); */

        // record the edges between the nodes in the path
        if (v > threshold && path[0] != -1) {
            // one way though
            for (int i = 0; i < path.size() - 2; i++) {
                value_weights(path[i], path[i+1]) += \
                    (1 - value_weights(path[i], path[i+1])) * v;
            }
            /* LOG("[+] EpisodicMemory: updated | size: " + std::to_string(path.size())); */
        }
    }

    void set_path(std::vector<int> path) { this->path = path; }
        /* LOG("## set path, size=" + std::to_string(path.size()));; */
    Eigen::MatrixXf& get_value_weights() { return value_weights; }
    void reset() { v = 0.0f; }

    EpisodicMemory(int size, float tau_rise = 3.0f, float tau_decay = 4.0f,
                      float threshold = 0.3f):
        value_weights(Eigen::MatrixXf::Zero(size, size)), tau_rise(tau_rise),
        tau_decay(tau_decay), threshold(threshold) {}
    std::string str() { return "EpisodicMemory"; }
    std::string repr() { return "EpisodicMemory"; }
};


struct StationarySensory {

    Eigen::VectorXf prev_representation;
    float v = 0.0f;
    float tau;
    float threshold;
    float min_cosine = 0.5f;

    bool call(Eigen::VectorXf representation) {

        float cosim = utils::cosine_similarity_vec(representation, prev_representation);

        // if cosine similarity is not nan
        if (std::isnan(cosim) || cosim < min_cosine) {
            /* LOG("[-] cosine similarity is nan"); */
            /* float sum_1 = representation.sum(); */
            /* float sum_2 = prev_representation.sum(); */
            /* LOG("[-] curr repr sum: " + std::to_string(sum_1)); */
            /* LOG("[-] prev repr sum: " + std::to_string(sum_2)); */
            cosim = 0.0f;
        }// else {
            /* LOG("[+] cosine similarity: " + std::to_string(cosim)); */
        /* } */
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


// === Programs ===

class PopulationMaxProgram {

    float output;

public:

    PopulationMaxProgram() {}
    ~PopulationMaxProgram() {}

    float call(const Eigen::VectorXf& u) {
        // compute the maximum activity
        output = *std::max_element(u.data(), u.data() + u.size());
        return output;
    }

    float get_value() { return output; }
    std::string str() { return "PopulationMaxProgram"; }
    std::string repr() { return "PopulationMaxProgram"; }
    int len() { return 1; }
};


struct ConsecutiveLocationsHandler {
    int start_point = -1;
    int end_point = -1;
    int counter = 0;
    int max_attempts = 1;

    void set_points(int start_point, int end_point) {
        this->start_point = start_point;
        this->end_point = end_point;
        this->counter = 0;
        /* LOG("[+] setting ConsecutiveLocationsHandler: " + \ */
        /*     std::to_string(start_point) + " -> " + \ */
        /*     std::to_string(end_point)); */
    }
    bool update() {

        if (start_point == end_point || start_point < 0 || end_point < 0) {
            /* LOG("[-] ConsecutiveLocationsHandler: invalid points or equal points"); */
            return true;
        }
        counter++;
        /* LOG("[-] ConsecutiveLocationsHandler: " + \ */
        /*     std::to_string(start_point) + " -> " + \ */
        /*     std::to_string(end_point) + " [" + \ */
        /*     std::to_string(counter) + "]"); */
        return false;
    }

    bool is_valid() { return counter < max_attempts; }
    void reset() {
        start_point = -1;
        end_point = -1;
        counter = 0;
    }

    ConsecutiveLocationsHandler(int max_attemps): max_attempts(max_attemps) {}
};


struct RewardObject {

    int trg_idx = -1;
    float trg_value = 0.0f;

    int update(Eigen::VectorXf& da_weights,
               PCNN_REF& space,
               bool trigger = true) {

        // exit: no trigge
        if (!trigger) {
            return -1;
        }

        // --- update the target representation ---

        // method 1: take the center of mass
        trg_idx = converge_to_trg_index(da_weights, space);

        // try method 2
        if (trg_idx < 0) {

            // method 2: take the argmax of the weights
            Eigen::Index maxIndex;
            float maxValue = da_weights.maxCoeff(&maxIndex);
            trg_idx = static_cast<int>(maxIndex);
        }

        // exit: no trg index
        if (trg_idx < 0) {
            /* LOG("[-] no trg index"); */
            return -1; }

        // exit: low trg value
        if (da_weights(trg_idx) < 0.00001f) {
            /* LOG("[-] low trg value"); */
            return -1; }

        // update the target value
        this->trg_idx = trg_idx;
        this->trg_value = da_weights(trg_idx);

        LOG("[+] RewardObject: trg_idx=" + std::to_string(trg_idx) + \
            " | trg_value=" + std::to_string(trg_value));

        return trg_idx;
    }

private:

    int converge_to_trg_index(Eigen::VectorXf& da_weights,
                              PCNN_REF& space) {

        // weights for the centers
        float cx, cy;
        float sum = da_weights.sum();
        if (sum == 0.0f) {
            /* LOG("[-] sum is zero"); */
            return -1;
        }

        Eigen::MatrixXf centers = space.get_centers();

        for (int i = 0; i < da_weights.size(); i++) {
            cx += da_weights(i) * centers(i, 0);
            cy += da_weights(i) * centers(i, 1);
        }

        // centers of mass
        cx /= sum;
        cy /= sum;

        // get closest center
        int closest_idx = space.calculate_closest_index({cx, cy});

        return closest_idx;
    }
};


class TargetProgram {

    // external variables
    Eigen::MatrixXf& wrec;
    Eigen::MatrixXf& connectivity;
    Eigen::MatrixXf& centers;
    PCNN_REF& space;
    /* Eigen::VectorXf& da_weights; */

    // internal variables
    ConsecutiveLocationsHandler conlochandler;
    EpisodicMemory episodic_memory;
    float speed;
    bool active;
    /* Eigen::VectorXf trg_representation; */
    int tmp_trg_idx;
    /* float trg_value; */
    int curr_idx;
    std::vector<int> plan_idxs;
    std::array<float, 2> next_position;
    std::array<float, 2> curr_position;
    int depth;
    int size;
    int counter;

    bool make_plan(Eigen::VectorXf& curr_representation,
                   Eigen::VectorXf& space_weights,
                   int tmp_trg_idx) {
                   /* bool use_episodic_memory = true) { */

        // calculate the current and target nodes
        Eigen::Index maxIndex_curr;
        curr_representation.maxCoeff(&maxIndex_curr);
        curr_idx = static_cast<int>(maxIndex_curr);
        /* LOG("[+] curr_idx: " + std::to_string(curr_idx)); */

        // calculate path
        /* std::vector<int> plan_idxs = \ */
        /*     utils::shortest_path_bfs(wrec, start_idx, end_idx); */
        /* LOG("use episodic memory: " + std::to_string(use_episodic_memory)); */
        /* if (use_episodic_memory == true) { */
            /* LOG("[+] using episodic memory"); */
            /* LOG("[+] curr_idx: " + std::to_string(curr_idx)); */
            /* std::vector<int> plan_idxs = \ */
            /*     utils::weighted_en_shortest_path(connectivity, */
            /*                                      episodic_memory.get_value_weights(), */
            /*                                      space_weights, */
            /*                                      curr_idx, tmp_trg_idx); */
        /* } else { */
        /*     LOG("[-] not using episodic memory"); */
        /* } */
        /* LOG("[+] plan_idxs size: " + std::to_string(plan_idxs.size())); */
        /* std::vector<int> plan_idxs = \ */
            /* utils::weighted_shortest_path(connectivity, */
            /*                               space_weights, */
            /*                               curr_idx, tmp_trg_idx); */

        std::vector<int> plan_idxs = \
            utils::spatial_shortest_path(connectivity,
                                         space.get_centers(),
                                         space_weights,
                                         curr_idx, tmp_trg_idx);

        /* } */

        /* float max_trg_value = trg_representation.maxCoeff(); */
        float max_curr_value = curr_representation.maxCoeff();

        // check if the plan is valid, ie size > 1
        if (plan_idxs.size() < 3) {
            /* LOG("[-] short plan"); */
            return false;
        } else if (max_curr_value < 0.00001f) {
            /* LOG("[-] low max curr value"); */
            return false;
        }

        // log plan
        /* LOG("\n[new plan]"); */
        /* LOG("start: " + std::to_string(curr_idx) + \ */
        /*     " | end: " + std::to_string(tmp_trg_idx)); */
        /* for (int i = 0; i < (plan_idxs.size()-1); i++) { */
        /*     LOG(std::to_string(plan_idxs[i]) + "->" + \ */
        /*         std::to_string(plan_idxs[i+1]) + " | " + \ */
        /*         std::to_string(connectivity(plan_idxs[i], plan_idxs[i+1]))); */
        /* } */
        /* LOG("#size=" + std::to_string(plan_idxs.size()) + "\n"); */

        // log idx centers
        /* LOG("centers:"); */
        /* for (int i = 0; i < plan_idxs.size(); i++) { */
        /*     LOG("[" + std::to_string(i) + "] " + \ */
        /*         std::to_string(centers(plan_idxs[i], 0)) + " | " + \ */
        /*         std::to_string(centers(plan_idxs[i], 1))); */
        /* } */

        // next position as the center corresponding to the
        // the next index in the plan
        this->active = true;
        this->curr_position = {centers(plan_idxs[0], 0),
                               centers(plan_idxs[0], 1)};
        this->next_position = {centers(plan_idxs[1], 0),
                               centers(plan_idxs[1], 1)};
        this->counter = 1;
        this->depth = plan_idxs.size();
        this->plan_idxs = plan_idxs;

        conlochandler.set_points(curr_idx, plan_idxs[1]);
        return true;
    }

public:

    TargetProgram(PCNN_REF& space, float speed, int max_attempts = 2):
        active(false), space(space), wrec(space.get_wrec()),
        connectivity(space.get_connectivity()),
        centers(space.get_centers()),
        episodic_memory(EpisodicMemory(space.get_size())),
        speed(speed), depth(0), conlochandler(max_attempts) {

        size = wrec.rows();
        plan_idxs = {};
        next_position = {0.0f, 0.0f};
        curr_position = {0.0f, 0.0f};
    }

    // UPDATE

    bool update(Eigen::VectorXf& curr_representation,
                Eigen::VectorXf& space_weights,
                int tmp_trg_idx) {

        // update graph
        wrec = space.get_wrec();
        connectivity = space.get_connectivity();

        // attempt planning
        bool is_valid = make_plan(curr_representation, space_weights,
                                  tmp_trg_idx);

        // exit: no plan
        if (!is_valid) {
            active = false;
            /* LOG("[-] no plan, it was invalid"); */
            return false;
        }

        active = true;
        episodic_memory.set_path(plan_idxs);
        return true;
    }

    // if there's a plan, follow it
    std::pair<std::array<float, 2>, bool> step_plan(Eigen::VectorXf& curr_representation) {

        // exit: active
        if (!active) {
            /* LOG("[-] not active"); */
            return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
    }

        std::array<float, 2> local_velocity;

        // check: that move has been tried a lot
        if (!conlochandler.is_valid()) {
            /* LOG("[-] max attempts reached <<<<<<<<<<<<<<"); */
            space.add_blocked_edge(conlochandler.start_point,
                                   conlochandler.end_point);
            conlochandler.reset();

            // check edges
            /* LOG("blocked edges conn: " + std::to_string(connectivity(conlochandler.start_point, */
            /*                                                     conlochandler.end_point))); */
            return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
        }

        // define current position from the representation argmax
        /* Eigen::Index maxIndex; */
        /* curr_representation.maxCoeff(&maxIndex); */
        /* int curr_idx = static_cast<int>(maxIndex); */
        /* this->curr_position = {centers(curr_idx, 0), */
        /*                        centers(curr_idx, 1)}; */

        // use the vspace instead
        this->curr_position = space.get_position();
        int curr_idx = space.calculate_closest_index(curr_position);

        /* LOG(" "); */

        /* LOG("next idx: " + std::to_string(plan_idxs[counter]) + \ */
        /*     ", curr idx: " + std::to_string(curr_idx)); */
        /* LOG("[+] curr position: " + std::to_string(curr_idx) + \ */
        /*     " at " + std::to_string(curr_position[0]) + ", " + std::to_string(curr_position[1])); */
        /* LOG("[+] next position: " + std::to_string(plan_idxs[counter]) + \ */
        /*     " at " + std::to_string(next_position[0]) + ", " + std::to_string(next_position[1])); */

        // distance netween the current and next position
        float dist = utils::euclidean_distance(curr_position, next_position);
        /* LOG("[+] dist: " + std::to_string(dist)); */


        // check | next position not reached yet
        if (dist > 0.01f && counter > 0) {
            float dx = next_position[0] - curr_position[0];
            float dy = next_position[1] - curr_position[1];
            /* LOG("[+] dx: " + std::to_string(dx) + " | dy: " + std::to_string(dy)); */

            // cover the last bit of distance
            if (dist < speed) {
                /* LOG("[-] last bit of distance b4 next position"); */
                local_velocity = {dx, dy};
            } else {
                float norm = sqrt(dx * dx + dy * dy);

                /* LOG("[+] next speed, norm: " + std::to_string(norm)); */
                /* // return the action vector of length speed */
                local_velocity = {speed * dx / norm, speed * dy / norm};
            }
 
            /* LOG("\tloc vel: " + std::to_string(local_velocity[0]) + \ */
            /*     " | " + std::to_string(local_velocity[1])); */

            /* std::array<float, 2> spos = space.get_position(); */
            /* LOG("space position: " + std::to_string(spos[0]) + ", " + std::to_string(spos[1])); */

            // step the consecutive locations handler
            /* bool is_identitcal = conlochandler.update(); */
            /* if (is_identitcal) { */
            /*     /1* LOG("identical idx in conlochandler, blocking"); *1/ */
            /*     this->active = false; */
            /*     this->counter = 0; */
            /*     this->depth = 0; */
            /*     conlochandler.reset(); */

            /*     return std::make_pair({0.0f, 0.0f}, false); */
            /* } */
        } else {

            /* LOG("[+] next position reached, distance=" + std::to_string(dist) + \ */
            /*     " [" + std::to_string(counter) + "]"); */

            if (counter > (plan_idxs.size()-1)) {
                this->active = false;
                this->counter = 0;
                this->depth = 0;
                /* LOG("[-] end of plan (hit plan size) ..active=" + std::to_string(active) + \ */
                /*     " [" + std::to_string(counter) + "]"); */
                conlochandler.reset();
                return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
            }

            // next position reached, move to the next
            this->next_position = {centers(plan_idxs[counter], 0),
                                   centers(plan_idxs[counter], 1)};
            /* LOG("[+] moving to the next position: " + std::to_string(plan_idxs[counter]) + \ */
            /*     " at " + std::to_string(next_position[0]) + " | " + \ */
            /*     std::to_string(next_position[1])); */

            // check if it's the last point
            if (plan_idxs[counter] == tmp_trg_idx) {
                /* LOG("target point set as next position, the completion is soon ##"); */
            } else if (plan_idxs[counter] < -10000 || plan_idxs[counter] > 10000) {
                LOG("[!] !!! possible memory leak?? " + std::to_string(plan_idxs[counter]) + ", counter=" + \
                    std::to_string(counter) + " | plan size=" + std::to_string(plan_idxs.size()));
            }

            float dx = next_position[0] - curr_position[0];
            float dy = next_position[1] - curr_position[1];

            // distance netween the current and next position
            float dist = utils::euclidean_distance(curr_position, next_position);

            // cover the last bit of distance
            if (dist < speed) {
                /* LOG("[-] last bit of distance b4 next position, \tloc vel: " + \ */
                    /* std::to_string(local_velocity[0]) + " | " + std::to_string(local_velocity[1])); */
                local_velocity = {dx, dy};
                /* LOG("\tloc vel: " + std::to_string(local_velocity[0]) + " | " + std::to_string(local_velocity[1])); */
            } else {
                float norm = sqrt(dx * dx + dy * dy);

                // return the action vector of length speed
                local_velocity = {speed * dx / norm, speed * dy / norm};
                /* LOG("\tloc vel: " + std::to_string(local_velocity[0]) + " | " + std::to_string(local_velocity[1])); */
            }

            // set the new points
            conlochandler.set_points(plan_idxs[counter-1], plan_idxs[counter]);

            counter++;
            /* LOG("\tcounter: " + std::to_string(counter)); */
        }

        // update the current position
        this->curr_position = {curr_position[0] + local_velocity[0],
                               curr_position[1] + local_velocity[1]};

        /* LOG("out velocity: " + std::to_string(local_velocity[0]) + \ */
        /*     " | " + std::to_string(local_velocity[1])); */
        return std::make_pair(local_velocity, true);
    }

    void step_episodic_memory(float reward) { episodic_memory.call(reward); }

    std::vector<int> make_shortest_path(Eigen::MatrixXf wrec,
                                        int start_idx, int end_idx) {
        return utils::shortest_path_bfs(wrec, start_idx, end_idx);
    }

    std::string str() { return "TargetProgram"; }
    std::string repr() { return "TargetProgram"; }
    int len() { return 1; }
    bool is_active() { return active; }
    bool is_plan_finished() { return counter == depth; }
    void set_wrec(Eigen::MatrixXf wrec) { this->wrec = wrec; }
    void set_centers(Eigen::MatrixXf centers) { this->centers = centers; }
    std::vector<int> get_plan() { return plan_idxs; }
    Eigen::MatrixXf& get_episodic_memory()
        { return episodic_memory.get_value_weights(); }
    void reset() {
        /* LOG("[-] resetting TargetProgram"); */
        active = false;
        counter = 0;
        depth = 0;
        episodic_memory.reset();
    }
};


class Circuits {

    BaseModulation& da;
    BaseModulation& bnd;
    /* MemoryRepresentation& memrepr; */
    /* MemoryAction& memact; */
    /* PopulationMaxProgram pmax; */

    std::array<float, CIRCUIT_SIZE> output;
    Eigen::VectorXf value_mask;
    int space_size;

public:

    Circuits(BaseModulation& da, BaseModulation& bnd):
             /* MemoryRepresentation& memrepr, */
             /* MemoryAction& memact): */
        da(da), bnd(bnd), //pmax(PopulationMaxProgram()),
        /* memrepr(memrepr), memact(memact), */
        value_mask(Eigen::VectorXf::Ones(da.len())),
        space_size(da.len()) {}

    // CALL
    std::array<float, CIRCUIT_SIZE> call(Eigen::VectorXf& representation,
                              float collision,
                              float reward,
                              bool simulate = false) {

        output[0] = bnd.call(representation, collision, simulate);
        output[1] = da.call(representation, reward, simulate);
        /* output[2] = pmax.call(representation); */
        /* output[3] = memrepr.call(representation, simulate); */
        /* output[4] = memact.call(action_idx, simulate); */

        return output;
    }

    Eigen::VectorXf& make_value_mask(bool strict = false) {

        Eigen::VectorXf& bnd_mask = bnd.make_mask();

        for (int i = 0; i < space_size; i++) {
            float bnd_value = bnd_mask(i);

            if (!strict) {
                value_mask(i) = bnd_value < 0.01f ? 1.0f : 0.0f;
                continue;
            }

            if (bnd_value < 0.01f) { value_mask(i) = 1.0f; }
            else if (bnd_value < 0.5f) { value_mask(i) = 0.00f; }
            else { value_mask(i) = -1000.0; }
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
    Eigen::VectorXf& get_da_weights() { return da.get_weights(); }
    Eigen::VectorXf& get_bnd_weights() { return bnd.get_weights(); }
    /* Eigen::VectorXf get_memory_mask() { return memrepr.make_mask(); } */
    Eigen::VectorXf get_bnd_mask() { return bnd.make_mask(); }
    /* Eigen::VectorXf get_memory_representation() { return memrepr.tape; } */
    /* float get_memory_representation_max() { return memrepr.get_max_value(); } */
    /* float get_memory_action_max() { return memact.get_max_value(); } */
    /* std::array<float, ACTION_SPACE_SIZE> get_memory_action() { return memact.tape; } */
    void reset() {
        da.reset();
        bnd.reset();
    }
};


/* ========================================== */
/* =========== EXPERIENCE MODULE ============ */
/* ========================================== */



struct DensityPolicy {

    PCNN_REF& space;
    float rwd_weight;
    float rwd_sigma;
    float col_weight;;
    float col_sigma;
    float rwd_drive = 0.0f;
    float col_drive = 0.0f;

    void call(Eigen::VectorXf& da_weights, Eigen::VectorXf& bnd_weights,
              std::array<float, 2> displacement,
              float curr_da, float curr_bnd, float reward, float collision) {

        // +reward -collision
        if (reward > 0.1 || curr_bnd < 0.01) {

            // update & remap
            rwd_drive = rwd_weight * curr_da;
            space.remap(bnd_weights, displacement, rwd_sigma, rwd_drive);

        } else if (collision > 0.1 || curr_da < 0.01) {

            // udpate & remap
            col_drive = col_weight * curr_bnd;
            space.remap(da_weights, displacement, col_sigma, col_drive);
        }
    }

    DensityPolicy(PCNN_REF& space,
                  float rwd_weight,
                  float rwd_sigma,
                  float col_weight,
                  float col_sigma):
        space(space), rwd_weight(rwd_weight),
        rwd_sigma(rwd_sigma), col_sigma(col_sigma),
        col_weight(col_weight) {}
    std::string str() { return "DensityPolicy"; }
    std::string repr() { return "DensityPolicy"; }
    float get_rwd_mod() { return rwd_drive; }
    float get_col_mod() { return col_drive; }
};


struct Plan {

    std::array<float, 2> action = {0.0f, 0.0f};
    int t = 0;
    int action_delay;

    // CALL
    std::pair<std::array<float, 2>, bool> call() {

        // exit: the action have been provided #action_delay times
        if (t > (action_delay-1)) { return std::make_pair(action, true); }

        // provide the action
        this->t++;
        return std::make_pair(action, false);
    }

    void reset() {
        this->action = {0.0f, 0.0f};
        this->t = 0;
    }

    Plan(float action_delay = 1.0f)
        { this->action_delay = static_cast<int>(action_delay); }
};


class ExperienceModule {

    // external components
    Circuits& circuits;
    PCNN_REF& space;

    // internal components
    Plan plan;
    /* ActionSampler2D action_sampler; */

    // parameters
    /* std::array<float, CIRCUIT_SIZE> weights; */
    float speed;
    float action_delay;
    const float open_threshold = 2.0f;  // radiants
    Eigen::VectorXf rejected_indexes;

    // alternate between random walk and edge exploration
    int edge_route_time = 0;
    int edge_route_interval = 100;

    // make new plan
    int make_plan(Eigen::VectorXf& curr_representation,
                  int rejected_idx) {

        /* random_action_plan(); */
        /* return -1; */
        /* LOG("[+] making new plan.."); */

        // check: the current position is at an open boundary
        Eigen::Index maxIndex;
        curr_representation.maxCoeff(&maxIndex);
        int start_idx = static_cast<int>(maxIndex);
        /* LOG("[+] start_idx: " + std::to_string(start_idx)); */
        /* float value = open_boundary_value(start_idx, circuits.get_bnd_weights(), */
        /*                                   space.get_nodes_max_angle()); */
        /* Eigen::VectorXf& node_degrees = space.get_node_degrees(); */
        float value = open_boundary_value(start_idx, circuits.get_bnd_weights(),
                                          space.get_node_degrees());

        // [+] new random walk plan at an open boundary
        if (value < open_threshold || rejected_idx == 404 || \
            edge_route_time < edge_route_interval) {
            /* LOG("[ ] at the open boundary | rejected | not edge time, value=" + std::to_string(value)); */
            /* one_step_look_ahead(curr_representation); */
            return -1;
        }// else {
            /* LOG("[-] not at the open boundary, value=" + std::to_string(value)); */
        /* } */

        // check: are there are points at the open boundary
        int open_boundary_idx = get_open_boundary_idx(rejected_idx);

        // [+] new random walk plan at an open boundary
        if (open_boundary_idx < 1) {
            /* LOG("[-] no open boundary"); */
            /* one_step_look_ahead(curr_representation); */
            return -1;
        }

        // [+] new trg plan to reach the open boundary
        /* LOG("[+] new trg plan to reach the open boundary"); */
        return open_boundary_idx;
        /* return -1; */
    }

    void random_action_plan() {

        std::array<float, 2> action = sample_random_action();
        this->plan.action = action;
        this->plan.t = 0;
    }

    // check map open-ness
    int get_open_boundary_idx(int rejected_idx) {

        Eigen::VectorXf& bnd_weights = circuits.get_bnd_weights();
        Eigen::VectorXf& node_degrees = space.get_node_degrees();

        if (rejected_idx > -1) { rejected_indexes(rejected_idx) = 1.0f; }

        // check each neuron
        int idx = -1;
        float min_value = 1000.0f;

        for (int i = 1; i < space.get_size(); i++) {
            float value = open_boundary_value(
                    i, bnd_weights, space.get_node_degrees());
            if (value < min_value && value > 0) {
                idx = i;
                min_value = value;
            }
        }
        return idx;
    }

    float open_boundary_value(int idx, Eigen::VectorXf& bnd_weights,
                              Eigen::VectorXf& node_degrees) {

        if (bnd_weights(idx) > 0.0f) { return 10000.0f; }
        return node_degrees(idx);
    }

    std::array<float, 2> sample_random_action() {
        // sample a random angle 0-2pi
        float angle = utils::random.get_random_float(0.0f, 2.0f * M_PI);
        return {std::cos(angle) * speed, std::sin(angle) * speed};
    }

public:
    bool new_plan = true;

    ExperienceModule(float speed,
                     Circuits& circuits,
                     PCNN_REF& space,
                     float action_delay = 1.0f,
                     int edge_route_interval = 100):
            speed(speed), circuits(circuits),
            space(space),
            plan(Plan(action_delay)),
            action_delay(action_delay),
            edge_route_interval(edge_route_interval),
            rejected_indexes(Eigen::VectorXf::Zero(space.get_size())) {}

    // CALL
    std::pair<std::array<float, 2>, int>
    call(std::string directive, Eigen::VectorXf& curr_representation,
         int rejected_idx = -1) {

        // --- check previous plan real quick
        std::pair<std::array<float, 2>, bool> next_step = plan.call();

        // --- new plan | for directive "new" or if the previous plan is finished
        if (directive == "new" || next_step.second) {

            // rejection : the attempt to make a trg plan to the boundary has failed
            int res = make_plan(curr_representation, rejected_idx);
            this->new_plan = true;

            // new plan to the open boundary
            if (res > -1) {
                /* LOG("[+] query new plan to the open boundary" + std::to_string(res)); */
                return {{-1.0f, -1.0f}, res}; }
            else {
                // make random action
                random_action_plan();
                // step the plan | (action, done)
                std::pair<std::array<float, 2>, bool> next_step = plan.call();
                edge_route_time++;
                /* LOG("[+] make random walk plan | edt: " + std::to_string(edge_route_time)); */
                return {next_step.first, -1};
            }
        }

        // --- continue the plan ---
        this->new_plan = false;
        /* LOG("[+] continue the random walk plan"); */
        return { next_step.first, -1 };
    }

    std::string str() { return "ExperienceModule"; }
    std::string repr() { return "ExperienceModule"; }
    void confirm_edge_walk() { edge_route_time = 0; }
    void reset_rejected_indexes()
        { rejected_indexes = Eigen::VectorXf::Zero(space.get_size()); }
};


/* ========================================== */
/* ================= BRAIN ================== */
/* ========================================== */


class Brain {

    // external components
    Circuits& circuits;
    PCNN_REF& space;
    ExperienceModule& expmd;
    StationarySensory& ssry;
    DensityPolicy& dpolicy;
    TargetProgram trgp;
    RewardObject rwobj = RewardObject();

    // variables
    Eigen::VectorXf curr_representation;
    Eigen::VectorXf value_representation;
    std::string directive;
    int clock;
    int trg_plan_end = 0;
    int forced_exploration = -1;
    int forced_duration = 7;

    // initialize
    std::pair<std::array<float, 2>, int> expmd_res = \
        std::make_pair(std::array<float, 2>{0.0f, 0.0f}, 0);
    int tmp_trg_idx = -1;
    std::array<float, 2> action = {0.0f, 0.0f};

    std::array<float, 2> attempt_boundary_plan(int idx) {

        // reset the set of rejected indexes
        expmd.reset_rejected_indexes();

        /* LOG("---"); */
        /* LOG("[+] attempting to plan to the open boundary"); */

        // attempt for a bunch of times | use 404 as final rejection
        for (int i = 0; i < 3; i++) {

            // attempt a plan
            trgp.reset();
            bool valid_plan = trgp.update(curr_representation,
                                          circuits.make_value_mask(false),
                                          idx);

            // valid plan
            if (valid_plan) {
                this->directive = "trg ob";
                /* LOG("[+] plan to the open boundary SUCCESSFUL"); */
                /* LOG("---"); */
                std::pair<std::array<float, 2>, bool> progress = trgp.step_plan(curr_representation);

                // confirm the edge walk
                expmd.confirm_edge_walk();
                return progress.first;
            } //{ LOG("trying again"); }

            // invalid plan -> try again
            std::pair<std::array<float, 2>, int> expmd_res = \
                expmd.call(directive, curr_representation, idx);
            idx = expmd_res.second;
        }

        // tried too many times, make a random walk plan instead
        /* LOG("[-] plan to the open boundary FAILED"); */
        std::pair<std::array<float, 2>, int> expmd_res = \
            expmd.call(directive, curr_representation, 404);
        /* LOG("---"); */

        return expmd_res.first;
    }

    void make_prediction() {

        // simulate a step
        Eigen::VectorXf& next_representation = space.simulate_one_step(action);

        // make prediction
        circuits.make_prediction(next_representation);
    }


public:

    Brain(Circuits& circuits,
          PCNN_REF& space,
          /* TargetProgram& trgp, */
          ExperienceModule& expmd,
          StationarySensory& ssry,
          DensityPolicy& dpolicy,
          float speed, int max_attempts = 3):
        circuits(circuits), space(space),
        expmd(expmd), ssry(ssry), dpolicy(dpolicy),
        trgp(TargetProgram(space, speed, max_attempts)),
        directive("new"), clock(0) {}

    // CALL
    std::array<float, 2> call(
            const std::array<float, 2>& velocity,
            float collision, float reward,
            bool trigger) {

        clock++;

        // === STATE UPDATE ===

        // :space
        auto [u, _] = space.call(velocity);
        this->curr_representation = u;

        // :circuits
        std::array<float, CIRCUIT_SIZE> state_int = \
            circuits.call(u, collision, reward, false);

        // :dpolicy
        dpolicy.call(circuits.get_da_weights(),
                     circuits.get_bnd_weights(),
                     velocity,
                     state_int[1], state_int[0], reward,
                     collision);
        space.update();

        // check still-ness
        if (forced_exploration < forced_duration) {
            forced_exploration++;
            trgp.reset();
            goto explore; }

        if (ssry.call(curr_representation)) {
            LOG("<forced exploration> : v=" + std::to_string(ssry.get_v()));
            forced_exploration = 0;
            goto explore;
        } //else { LOG("[+] repr trace below: " + std::to_string(rdtrace.get_v())); }


        // === GOAL-DIRECTED BEHAVIOUR ====================================

        // :target program
        trgp.set_wrec(space.get_connectivity());
        trgp.set_centers(space.get_centers());
        /* trgp.set_da_weights(circuits.get_da_weights()); */
        trgp.step_episodic_memory(reward);

        // --- current target plan

        // check: current trg plan
        /* if (trgp.is_active() && !trgp.is_plan_finished()) { */
        if (trgp.is_active()) {
            /* LOG("continuing trg plan..."); */
            /* this->directive = "trg"; */
            trg_plan_end = 0;
            std::pair<std::array<float, 2>, bool> progress = \
                trgp.step_plan(curr_representation);

            // keep going
            if (progress.second) {
                /* LOG("continuing trg plan... " + std::to_string(progress.first[0]) + ", " + std::to_string(progress.first[1])); */
                this->action = progress.first; 
                goto final;
            }
            // end or fail -> random walk
            forced_exploration = 0;
            /* LOG("plan failed or ended, walk"); */
        } //else if (!trgp.is_active()) {
            /* LOG("[-] trg plan finished"); */
            /* this->directive = "new"; */
        /* } //else { LOG("[-] trg plan not active");} */
        /* this->directive = "new"; */

        // time since the last trg plan ended
        trg_plan_end++;

        // update trg memory in case of reward
        /* if (reward > 0.0f && trg_plan_end < 4) */
        /*     { */
        /*     LOG("########### +++++ trg memory " + std::to_string(trg_plan_end) + \ */
        /*         " | reward: " + std::to_string(reward)); */
        /* } */
        /*     /1* trgp.update_episodic_memory(); } *1/ */
        /* else if (reward) { LOG("########### no trg memory? " + std::to_string(trg_plan_end) + \ */
        /*     " | reward: " + std::to_string(reward)); } */
        /* else { LOG("#### no reward " + std::to_string(reward));} */

        // --- new target plan: REWARD

        // :reward object | new reward trg index
        tmp_trg_idx = rwobj.update(circuits.get_da_weights(),
                                       space, trigger);

        if (tmp_trg_idx > -1) {

            /* LOG("[+] new reward trg index: " + std::to_string(tmp_trg_idx)); */

            // trg position
            Eigen::VectorXf trg_position = space.get_centers().row(tmp_trg_idx);

            // argmax of the current representation
            Eigen::Index maxIndex;
            curr_representation.maxCoeff(&maxIndex);
            int curr_idx = static_cast<int>(maxIndex);
            Eigen::VectorXf curr_position = space.get_centers().row(curr_idx);

            // check new reward trg plan
            bool valid_plan = trgp.update(curr_representation,
                                          circuits.make_value_mask(),
                                          tmp_trg_idx);

            // [+] reward trg plan
            if (valid_plan) {
                this->directive = "trg";
                std::pair<std::array<float, 2>, bool> progress = trgp.step_plan(curr_representation);
                if (progress.second) {
                    this->action = progress.first; 
                    goto final;
                }
                forced_exploration = 0;
                /* LOG("planning failed, walk"); */
            }
        }

        // === EXPLORATIVE BEHAVIOUR =======================================
explore:

        // check: collision
        if (collision > 0.0f) { this->directive = "new"; }
        else { this->directive = "continue"; }

        // :experience module
        /* return expmd.call(directive, curr_representation); */
        expmd_res = expmd.call(directive, curr_representation);

        // check: plan to go to the open boundary
        if (expmd_res.second > -1) {
            this->action = attempt_boundary_plan(expmd_res.second);
            goto final;
        }
        this->action = expmd_res.first;

final:
        // make prediction
        make_prediction();
        return action;
    }

    std::string str() { return "Brain"; }
    std::string repr() { return "Brain"; }
    int len() { return space.get_size(); }
    Eigen::VectorXf get_trg_representation() {
        Eigen::VectorXf trg_representation = Eigen::VectorXf::Zero(space.get_size());
        /* trg_representation(trgp.get_trg_idx()) = trgp.get_trg_value(); */
        trg_representation(rwobj.trg_idx) = 1.; // rwobj.trg_value;
        return trg_representation;
    }
    /* int get_trg_idx() { return trgp.get_trg_idx(); } */
    int get_trg_idx() { return rwobj.trg_idx; }
    std::array<float, 2> get_leaky_v() { return circuits.get_leaky_v(); }
    Eigen::VectorXf get_representation()
        { return curr_representation; }
    std::array<float, 2> get_trg_position() {
        Eigen::MatrixXf centers = space.get_centers();
        return {centers(rwobj.trg_idx, 0), centers(rwobj.trg_idx, 1)};
    }
    ExperienceModule& get_expmd() { return expmd; }
    PCNN_REF& get_space() { return space; }
    Eigen::MatrixXf& get_episodic_memory()
        { return trgp.get_episodic_memory(); }
    std::vector<std::array<std::array<float, 2>, 3>> make_edges_value()
        { return space.make_edges_value(trgp.get_episodic_memory()); }
    std::string get_directive() { return directive; }
    std::vector<int> get_trg_plan() { return trgp.get_plan(); }
    std::vector<int> get_plan_idxs() { return trgp.get_plan(); }
    std::array<float, 2> get_space_position() { return space.get_position(); }
    /* void set_plan_positions(std::array<float, 2> position) */
    /*     { expmd.set_plan_positions(position); } */
    /* std::vector<std::array<float, 2>> get_plan_positions(std::array<float, 2> position) { */
    /*     if (expmd.new_plan) { expmd.set_plan_positions(position); } */
    /*     return expmd.get_plan_positions(); */
    /* } */
    /* float get_plan_score() */
    /*     { return expmd.get_plan_score(); } */
    /* std::vector<float> get_plan_scores() */
    /*     { return expmd.get_plan_scores(); } */
    /* std::array<float, CIRCUIT_SIZE>  get_plan_values() */
    /*     { return expmd.get_plan_values(); } */
    void reset() {
        trgp.reset();
        circuits.reset();
    }
};



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


int simple_env(int pause = 20, int duration = 3000, float bnd_w = 0.0f) {

    // settings
    float SPEED = 1.0f;
    std::array<float, 2> BOUNDS = {0.0f, 50.0f};
    int N = std::pow(25, 2);

    // SPACE
    std::vector<GridLayerSq> gcn_layers;
    gcn_layers.push_back(GridLayerSq(0.04, 0.1, {-1.0f, 1.0f, -1.0f, 1.0f}));
    gcn_layers.push_back(GridLayerSq(0.04, 0.07, {-1.0f, 1.0f, -1.0f, 1.0f}));
    gcn_layers.push_back(GridLayerSq(0.04, 0.03, {-1.0f, 1.0f, -1.0f, 1.0f}));
    gcn_layers.push_back(GridLayerSq(0.04, 0.005, {-1.0f, 1.0f, -1.0f, 1.0f}));
    GridNetworkSq gcn = GridNetworkSq(gcn_layers);
    PCNNsqv2 space = PCNNsqv2(N, gcn.len(), 10.0f, 1.4f, 0.01f, 0.2f, 0.7f, 5.0f, 5, gcn, "2D");

    // MODULATION
    // name size lr threshold maxw tauv eqv minv
    BaseModulation da = BaseModulation("DA", N, 0.5f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.01f);
    BaseModulation bnd = BaseModulation("BND", N, 0.9f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.01f);
    /* MemoryRepresentation memrepr = MemoryRepresentation(N, 2.0f, 0.1f); */
    /* MemoryAction memact = MemoryAction(2.0f); */
    StationarySensory ssry = StationarySensory(N, 100.0f, 0.2f, 0.5f);
    Circuits circuits = Circuits(da, bnd);//, memrepr, memact);

    // TARGET PROGRAM
    /* TargetProgram trgp = TargetProgram(space.get_connectivity(), space.get_centers(), */
    /*                                    da.get_weights(), SPEED); */
    // EXPERIENCE MODULE & BRAIN
    DensityPolicy dpolicy = DensityPolicy(space, 0.5f, 40.0f, 0.5f, 20.0f);
    ExperienceModule expmd = ExperienceModule(SPEED, circuits, space, 1.0f);
                                              /* {bnd_w, 0.0f, 0.0f, 0.0f, 0.0f}, 1.0f); */
    Brain brain = Brain(circuits, space, expmd, ssry, dpolicy, SPEED, 5);

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
    LOG("%Simple environment simulation");
    LOG("%Duration: " + std::to_string(duration));
    LOG("%Pause: " + std::to_string(pause));

    // loop
    LOG("<start>");
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


    LOG("<end>");

    LOG("\n------------");
    LOG("#count: " + std::to_string(space.len()));
    LOG("#rw_count: " + std::to_string(reward.count));
    LOG("#collision: " + std::to_string(num_collision));
    LOG("------------\n");

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
