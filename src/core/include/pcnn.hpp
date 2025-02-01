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

#define SPACE utils::logging.space
#define GCL_SIZE 36
#define GCL_SIZE_SQRT 6
#define PCNN_REF PCNNsqv2
#define GCN_REF GridNetworkSq
#define CIRCUIT_SIZE 5
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


struct VelocitySpace {

    Eigen::MatrixXf centers;
    Eigen::MatrixXf connectivity;
    Eigen::MatrixXf weights;
    std::array<float, 2> position;
    int size;
    /* std::vector<std::array<float, 2>> trajectory; */
    float threshold;

    VelocitySpace(int size, float threshold)
        : size(size), threshold(threshold) {
        centers = Eigen::MatrixXf::Constant(size, 2, -1000.0f);
        connectivity = Eigen::MatrixXf::Zero(size, size);
        weights = Eigen::MatrixXf::Zero(size, size);
        position = {1.9479814f, 0.9479814f};
    }

    ~VelocitySpace() {}

    // CALL
    std::array<float, 2> call(const std::array<float, 2>& v) {
        position[0] += v[0];
        position[1] += v[1];
        /* LOG("[VS] position: " + std::to_string(position[0]) + ", " + \ */
        /*     std::to_string(position[1])); */
        /* trajectory.push_back({position[0], position[1]}); */
    }

    void update(int idx) {

        // update the centers
        centers.row(idx) = Eigen::Vector2f(
            position[0], position[1]);

        // add recurrent connections
        for (int j = 0; j < size; j++) {
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
    Eigen::MatrixXf& get_connectivity() { return connectivity; }
    Eigen::MatrixXf& get_connections() { return weights; }
    std::array<float, 2> get_position() { return position; }

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

    Eigen::VectorXf call(const std::array<float, 2>& x) {

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

    Eigen::VectorXf simulate(
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

    Eigen::VectorXf simulate_one_step(const std::array<float, 2>& v) {

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
    const float gain;
    const float offset;
    const float clip_min;
    const float threshold;
    const float rep_threshold;
    const int num_neighbors;
    const std::string name;

    float ach;

    GridNetworkSq xfilter;

    // variables
    Eigen::MatrixXf Wff;
    Eigen::MatrixXf Wffbackup;
    float delta_wff;
    Eigen::MatrixXf Wrec;
    Eigen::MatrixXf connectivity;
    Eigen::VectorXf mask;
    std::vector<int> fixed_indexes;
    std::vector<int> free_indexes;
    int cell_count;
    Eigen::VectorXf u;
    Eigen::VectorXf x_filtered;

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
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
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

        ach = 1.0;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};
    }

    ~PCNNsqv2() {}; //LOG("[-] PCNN destroyed"); }

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
            if (similarity > (rep_threshold * ach)) {
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
            vspace.update(idx);
            this->Wrec = vspace.get_connections();
            this->connectivity = vspace.get_connectivity();
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

    Eigen::VectorXf simulate_one_step(const std::array<float, 2>& v) {

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

    int calculate_closest_index(const std::array<float, 2>& c) {
        return vspace.calculate_closest_index(c);
    }

    void reset() {
        u = Eigen::VectorXf::Zero(N);
        /* trace = Eigen::VectorXf::Zero(N); */
        /* pre_x = Eigen::VectorXf::Zero(N); */
    }

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
    Eigen::VectorXf get_activation_gcn() const {
        return xfilter.get_activation(); }
    Eigen::MatrixXf get_wff() const { return Wff; }
    Eigen::MatrixXf& get_wrec() { return Wrec; }
    std::vector<std::array<std::array<float, 2>, 2>> make_edges() \
    { return vspace.make_edges(); }
    /* std::vector<std::array<float, 2>> get_trajectory() { */
    /*     return vspace.get_trajectory(); } */
    Eigen::MatrixXf& get_connectivity() { return connectivity; }
    Eigen::MatrixXf& get_centers(bool nonzero = false) {
        return vspace.get_centers(nonzero); }
    float get_delta_update() const { return delta_wff; }
    Eigen::MatrixXf get_positions_gcn() {
        return xfilter.get_positions();
    }
    std::array<float, 2> get_position() {
        return vspace.get_position();
    }
    std::vector<std::array<std::array<float, 2>, GCL_SIZE>> get_gc_positions_vec() {
        return xfilter.get_positions_vec();
    }
    void reset_gcn(std::array<float, 2> v) {
        xfilter.reset(v);
    }

    // @brief modulate the density of new PCs
    void ach_modulation(float ach) {
        this->ach = ach;
    }

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
    float threshold;
    float max_w;
    Eigen::VectorXf weights;
    LeakyVariable1D leaky;


public:

    BaseModulation(std::string name, int size, float lr,
                   float threshold, float max_w = 1.0f,
                   float tau_v = 1.0f, float eq_v = 1.0f,
                   float min_v = 0.0f):
        name(name), size(size), lr(lr), threshold(threshold),
        max_w(max_w), leaky(LeakyVariable1D(name, eq_v, tau_v, min_v))
    { weights = Eigen::VectorXf::Zero(size); }

    ~BaseModulation() {}

    // CALL
    float call(const Eigen::VectorXf& u,
               float x = 0.0f, bool simulate = false) {

        // forward to the leaky variable
        float v = leaky.call(x, simulate);

        // update the weights
        if (!simulate) {
            for (int i = 0; i < size; i++) {
                float ui = u[i] > threshold ? u[i] : 0.0;
                /* weights[i] += lr * v * ui; */
                float dw = lr * v * ui;
                weights[i] += dw;

                // clip the weights in (0, max_w)
                if (weights[i] < 0.0) {
                    weights[i] = 0.0;
                } else if (weights[i] > max_w) {
                    weights[i] = max_w;
                }

                if (dw > 0.0f) {
                    LOG(name + " | +LTP");
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

    float get_output() { return output; }
    /* std::vector<float> get_weights() { return weights; } */
    Eigen::VectorXf& get_weights() { return weights; }
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
    float threshold;

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

    Eigen::VectorXf get_memory_as_mask() {

        // the nodes that are fresh in memory will affect action selection,
        // thus acting as a mask
        for (int i = 0; i < tape.size(); i++) {
            mask(i) = tape(i) > threshold ? 0.0f : 1.0f;
        }
        return mask;
    }

    MemoryRepresentation(int size, float decay, float threshold):
        tape(Eigen::VectorXf::Zero(size)), decay(decay),
        threshold(threshold), mask(Eigen::VectorXf::Zero(size)) {}
    ~MemoryRepresentation() {}
    std::string str() { return "MemoryRepresentation"; }
    std::string repr() { return "MemoryRepresentation"; }

private:

    void update(Eigen::VectorXf& representation) {

        Eigen::Index maxIndex;
        representation.maxCoeff(&maxIndex);
        int max_idx = static_cast<int>(maxIndex);
        float max_value = representation.maxCoeff();

        // decay the memory
        tape -= tape / decay;
        tape(maxIndex) = max_value;
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
        tape[idx] = 1.0f;
        return tape[idx];
    }

    MemoryAction(float decay): decay(decay) {}
    ~MemoryAction() {}
    std::string str() { return "MemoryAction"; }
    std::string repr() { return "MemoryAction"; }
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


class TargetProgram {

    // external variables
    Eigen::MatrixXf& wrec;
    Eigen::MatrixXf& centers;
    PCNN_REF& space;
    Eigen::VectorXf& da_weights;

    // internal variables
    float speed;
    bool active;
    /* Eigen::VectorXf trg_representation; */
    int trg_idx;
    float trg_value;
    std::vector<int> plan_idxs;
    std::array<float, 2> next_position;
    std::array<float, 2> curr_position;
    int depth;
    int size;
    int counter;

    bool make_plan(Eigen::VectorXf& curr_representation,
                   Eigen::VectorXf& space_weights) {

        // calculate the current and target nodes
        /* Eigen::Index maxIndex_trg; */
        Eigen::Index maxIndex_curr;
        /* trg_representation.maxCoeff(&maxIndex_trg); */
        curr_representation.maxCoeff(&maxIndex_curr);
        int curr_idx = static_cast<int>(maxIndex_curr);
        /* int end_idx = static_cast<int>(maxIndex_trg); */

        // calculate path
        /* std::vector<int> plan_idxs = \ */
        /*     utils::shortest_path_bfs(wrec, start_idx, end_idx); */
        std::vector<int> plan_idxs = \
            utils::weighted_shortest_path(wrec, space_weights,
                                          curr_idx, trg_idx);

        /* float max_trg_value = trg_representation.maxCoeff(); */
        float max_curr_value = curr_representation.maxCoeff();

        // check if the plan is valid, ie size > 1
        if (plan_idxs.size() < 3) {
            /* LOG("[-] short plan"); */
            return false;
        } else if (max_curr_value < 0.00001f || trg_value < 0.00001f) {
            return false;
        }

        // define next position as the center corresponding to the
        // the next index in the plan
        this->active = true;
        this->curr_position = {centers(curr_idx, 0),
                               centers(curr_idx, 1)};
        this->next_position = {centers(plan_idxs[1], 0),
                               centers(plan_idxs[1], 1)};
        this->counter = 1;
        this->depth = plan_idxs.size();
        this->plan_idxs = plan_idxs;
        return true;
    }

    int converge_to_trg_index() {

        // weights for the centers
        float cx, cy;
        float sum = da_weights.sum();
        if (sum == 0.0f) {
            /* LOG("[-] sum is zero"); */
            return -1;
        }

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

    // !unused for now
    std::array<float, 2> converge_to_location(
        Eigen::VectorXf& representation) {

        // weights for the centers
        float cx, cy;
        float sum = representation.sum();
        if (sum == 0.0f) {
            /* LOG("[-] sum is zero"); */
            return {-1000.0f, 0.0f};
        }

        for (int i = 0; i < representation.size(); i++) {
            cx += representation(i) * centers(i, 0);
            cy += representation(i) * centers(i, 1);
        }

        cx /= sum;
        cy /= sum;
        /* LOG("[+] cx: " + std::to_string(cx) + ", cy: " + std::to_string(cy)); */

        return {cx, cy};
    }

public:

    TargetProgram(Eigen::VectorXf& da_weights,
                  PCNN_REF& space,
                  float speed):
        da_weights(da_weights), active(false), space(space),
        wrec(space.get_wrec()), centers(space.get_centers()),
        speed(speed), depth(0) {

        size = wrec.rows();
        /* trg_representation = Eigen::VectorXf::Zero(size); */
        plan_idxs = {};
        next_position = {0.0f, 0.0f};
        curr_position = {0.0f, 0.0f};
        trg_idx = -1;
        trg_value = 0.0f;

        /* LOG("[+] TargetProgramV2 created"); */
    }

    ~TargetProgram() {} //LOG("[-] TargetProgramV2 destroyed"); }

    // UPDATE
    bool update(Eigen::VectorXf& curr_representation,
                Eigen::VectorXf& space_weights,
                bool trigger = true) {

        active = false;

        // exit: no trigge
        if (!trigger) {
            return false;
        }

        // define a target representation by taking an argmax
        /* trg_representation = \ */
        /*     Eigen::VectorXf::Zero(size); */

        // method 1: take the argmax of the weights
        /* Eigen::Index maxIndex; */
        /* float maxValue = da_weights.maxCoeff(&maxIndex); */
        /* trg_idx = static_cast<int>(maxIndex); */

        // method 2: take the center of mass
        trg_idx = converge_to_trg_index();

        // exit: no trg index
        if (trg_idx < 0) { return false; }

        // exit: low trg value
        if (da_weights(trg_idx) < 0.00001f) { return false; }
        trg_value = da_weights(trg_idx);

        // set the target representation index
        // to the maximum value
        /* trg_representation(maxIndex) = 1.0f; */

        // make plan
        bool is_valid = make_plan(curr_representation, space_weights);

        // exit: no plan
        if (!is_valid) { return false; }
            /* LOG("[-] invalid plan!!!"); */

        /* LOG("[+] plan_idxs: " + std::to_string(plan_idxs.size())); */
        return true;
    }

    // if there's a plan, follow it
    std::array<float, 2> step_plan(Eigen::VectorXf& curr_representation) {

        // exit: active
        if (!active) {
            /* LOG("[-] not active"); */
            return {0.0f, 0.0f}; }

        // exit: end of plan
        /* if (counter == depth && active) { */
        /*     this->active = false; */
        /*     /1* LOG("[-] end of plan ..active=" + std::to_string(active)); *1/ */
        /*     counter = 0; */
        /*     depth = 0; */
        /*     LOG("[-] active but counter==depth"); */
        /*     return {0.0f, 0.0f}; */
        /* } */

        /* LOG("counter: " + std::to_string(counter)); */

        // retrieve center of the current pc
        /* std::array<float, 2> curr_position = converge_to_location( */
        /*     curr_representation); */

        std::array<float, 2> local_velocity;

        // exit: failed convergence
        /* if (curr_position[0] == 0.0f && curr_position[1] == 0.0f) { */
        /*     this->active = false; */
        /*     counter = 0; */
        /*     depth = 0; */
        /*     LOG("[-] failed convergence"); */
        /*     // log positions */
        /*     /1* LOG("[-] curr_position: " + std::to_string(curr_position[0]) + \ *1/ */
        /*     /1*     ", " + std::to_string(curr_position[1])); *1/ */
        /*     return {0.0f, 0.0f}; */
        /* } */

        /* LOG("[+] curr_position: " + std::to_string(curr_position[0]) + \ */
        /*     ", " + std::to_string(curr_position[1])); */
        /* LOG("[+] next_position: " + std::to_string(next_position[0]) + \ */
        /*     ", " + std::to_string(next_position[1])); */

        // distance netween the current and next position
        float dist = utils::euclidean_distance(curr_position, next_position);
        /* LOG("[+] dist: " + std::to_string(dist)); */

        // check: next position not reached
        if (dist > 0.01f && counter > 0) {
            float dx = next_position[0] - curr_position[0];
            float dy = next_position[1] - curr_position[1];
            /* LOG("[+] dx: " + std::to_string(dx) + ", dy: " + std::to_string(dy)); */
            /* LOG("[+] speed: " + std::to_string(speed)); */

            // cover the last bit of distance
            if (dist < speed) {
                /* LOG("[-] last bit of distance b4 next position"); */
                local_velocity = {dx, dy};
            } else {
                float norm = sqrt(dx * dx + dy * dy);

                /* LOG("[+] next speed, norm: " + std::to_string(norm)); */
                // return the action vector of length speed
                local_velocity = {speed * dx / norm, speed * dy / norm};
            }
        } else {

            /* LOG("\n[+] next position reached, move to the next"); */
            counter++;

            if (counter > (plan_idxs.size()-1)) {
                /* LOG("[ BUG? ] counter > plan_idxs.size() | !!"); */
                this->active = false;
                this->counter = 0;
                this->depth = 0;
                return {0.0f, 0.0f};
            }

            // next position reached, move to the next
            this->next_position = {centers(plan_idxs[counter], 0),
                                   centers(plan_idxs[counter], 1)};
            /* LOG("[+] next position: " + std::to_string(next_position[0]) + \ */
            /*     ", " + std::to_string(next_position[1])); */

            float dx = next_position[0] - curr_position[0];
            float dy = next_position[1] - curr_position[1];

            // distance netween the current and next position
            float dist = utils::euclidean_distance(curr_position, next_position);

            // cover the last bit of distance
            if (dist < speed) {
                /* LOG("[-] last bit of distance b4 next position"); */
                local_velocity = {dx, dy};
            } else {
                float norm = sqrt(dx * dx + dy * dy);

                /* LOG("[+] next speed, norm: " + std::to_string(norm)); */
                // return the action vector of length speed
                local_velocity = {speed * dx / norm, speed * dy / norm};
            }
            /* float norm = sqrt(dx * dx + dy * dy); */

            /* LOG("[+] next position, norm: " + std::to_string(norm)); */

            // return the action vector of length speed
            /* local_velocity = {speed * dx / norm, speed * dy / norm}; */
        }

        // update the current position
        this->curr_position = {curr_position[0] + local_velocity[0],
                               curr_position[1] + local_velocity[1]};

        // check: end of plan
        if (counter == depth) {
            this->active = false;
            counter = 0;
            depth = 0;
            /* LOG("[-] end of plan ..active=" + std::to_string(active)); */
        }
        return local_velocity;
    }

    int get_trg_idx() { return trg_idx; }
    int get_trg_value() { return trg_value; }
    std::vector<int> make_shortest_path(Eigen::MatrixXf wrec,
                                        int start_idx, int end_idx) {
        return utils::shortest_path_bfs(wrec, start_idx, end_idx);
    }

    std::string str() { return "TargetProgram"; }
    std::string repr() { return "TargetProgram"; }
    int len() { return 1; }
    bool is_active() { return active; }
    bool is_plan_finished() { return counter == depth; }
    void set_da_weights(Eigen::VectorXf da_weights) {
        this->da_weights = da_weights; }
    void set_wrec(Eigen::MatrixXf wrec) {
        this->wrec = wrec; }
    void set_centers(Eigen::MatrixXf centers) {
        this->centers = centers; }
    std::vector<int> get_plan() { return plan_idxs; }
    void reset() {
        active = false;
        counter = 0;
        depth = 0;
        trg_idx = -1;
        trg_value = 0.0f;
    }
};


class Circuits {

    BaseModulation& da;
    BaseModulation& bnd;
    MemoryRepresentation& memrepr;
    MemoryAction& memact;
    PopulationMaxProgram pmax;

    std::array<float, CIRCUIT_SIZE> output;

public:

    Circuits(BaseModulation& da, BaseModulation& bnd,
             MemoryRepresentation& memrepr,
             MemoryAction& memact):
        da(da), bnd(bnd), pmax(PopulationMaxProgram()),
        memrepr(memrepr), memact(memact) {}

    ~Circuits() {}

    // CALL
    std::array<float, CIRCUIT_SIZE> call(Eigen::VectorXf& representation,
                              float collision,
                              float reward,
                              int action_idx = -1,
                              bool simulate = false) {

        output[0] = bnd.call(representation, collision, simulate);
        output[1] = da.call(representation, reward, simulate);
        output[2] = pmax.call(representation);
        output[3] = memrepr.call(representation, simulate);
        output[4] = memact.call(action_idx, simulate);

        return output;
    }

    std::string str() { return "Circuits"; }
    std::string repr() { return "Circuits"; }
    int len() { return CIRCUIT_SIZE; }
    std::array<float, CIRCUIT_SIZE> get_output() { return output; }
    std::array<float, 2> get_leaky_v() {
        return {da.get_leaky_v(), bnd.get_leaky_v()}; }
    Eigen::VectorXf& get_da_weights() { return da.get_weights(); }
    Eigen::VectorXf& get_bnd_weights() { return bnd.get_weights(); }
    Eigen::VectorXf get_memory_representation_mask() {
        return memrepr.get_memory_as_mask();
    }
    Eigen::VectorXf get_memory_representation() { return memrepr.tape; }
    std::array<float, ACTION_SPACE_SIZE> get_memory_action() { return memact.tape; }
    void reset() {
        da.reset();
        bnd.reset();
    }
};


/* ========================================== */
/* =========== EXPERIENCE MODULE ============ */
/* ========================================== */


struct PolicyRNN {

    std::array<std::array<float, POLICY_HIDDEN>, POLICY_HIDDEN> weights;
    std::array<float, POLICY_HIDDEN> x = {0.0f};
    std::array<float, POLICY_OUTPUT> y = {0.0f};
    std::array<std::array<float, POLICY_HIDDEN>, POLICY_HIDDEN> in_weights = {{1.0f}};
    std::array<std::array<float, POLICY_HIDDEN>, POLICY_HIDDEN> out_weights = {{1.0f}};
    int num_steps;

    PolicyRNN(std::array<std::array<float, POLICY_HIDDEN>, POLICY_HIDDEN> weights):
        weights(weights), num_steps(num_steps) {}

    ~PolicyRNN() {}

    // CALL
    std::array<float, POLICY_OUTPUT> call(std::array<float, POLICY_INPUT>& input) {

        // set input
        for (int i = 0; i < POLICY_INPUT; i++) {
            x[i] = input[i];
        }

        // internal recurrent steps
        for (int t = 0; t < num_steps; t++) {
            std::array<float, POLICY_HIDDEN> new_x = {0.0f};
            for (int i = 0; i < POLICY_HIDDEN; i++) {
                for (int j = 0; j < POLICY_HIDDEN; j++) {
                    if (i == j) { continue; }
                    new_x[i] += weights[i][j] * x[j];
                }
            }
            x = new_x;
        }

        // output
        for (int i = 0; i < POLICY_OUTPUT; i++) {
            y[i] = 0.0f;
            for (int j = 0; j < POLICY_HIDDEN; j++) {
                y[i] += out_weights[i][j] * x[j];
            }
        }
    }

    std::string str() { return "DecisionMakingRNN"; }
    std::string repr() { return "DecisionMakingRNN"; }
};


struct ActionSampler2D {

    std::array<std::array<float, 2>, ACTION_SPACE_SIZE> action_set = {{0.0f}};
    int counter = 0;

    // CALL
    std::array<float, 2> call() {

        if (counter == ACTION_SPACE_SIZE) {
           counter = ACTION_SPACE_SIZE - 1;
           std::cerr << ">>> ! Action space complete, using the last action" << std::endl;
        }

        // Sample a new index
        std::array<float, 2> velocity = action_set[counter];
        counter++;

        return velocity;
    }

    int get_counter() { return counter-1; }
    bool is_done() { return counter == ACTION_SPACE_SIZE; }
    void reset() { counter = 0; }

    ActionSampler2D(float speed) {
        make_action_space();
        update_actions(speed);
    }
    ~ActionSampler2D() {}

private:

    // @brief update the actions given a speed
    void update_actions(float speed) {

        LOG("[updating actions in ActionSampler2D]");
        LOG("[+] speed: " + std::to_string(speed));

        for (size_t i = 0; i < ACTION_SPACE_SIZE; i++) {
            float dx = action_set[i][0];
            float dy = action_set[i][1];
            float scale = speed / std::sqrt(2.0f);
            if (dx == 0.0 && dy == 0.0) {
                continue;
            } else if (dx == 0.0) {
                dy *= speed;
            } else if (dy == 0.0) {
                dx *= speed;
            } else {
                // speed / sqrt(2)
                dx *= scale;
                dy *= scale;
            }
            action_set[i] = {dx, dy};
        }
    }

    // @brief divide the circle into #ACTION_SPACE_SIZE actions
    void make_action_space() {
        // Calculate angle increment based on ACTION_SPACE_SIZE
        float angle_increment = 2.0f * M_PI / ACTION_SPACE_SIZE;

        // Generate points around the circle
        for (size_t i = 0; i < ACTION_SPACE_SIZE; i++) {
            // Calculate angle for current action
            float angle = i * angle_increment;

            // Convert polar coordinates to Cartesian coordinates
            // Using unit circle (radius = 1.0)
            float dx = std::cos(angle);
            float dy = std::sin(angle);

            // Store the normalized direction vector
            action_set[i] = {dx, dy};
        }
    }

};


struct Plan {

    std::array<float, 2> action = {0.0f, 0.0f};
    int idx = 0;
    std::array<std::array<float, CIRCUIT_SIZE>, ACTION_SPACE_SIZE> all_values = {{0.0f}};
    std::array<float, ACTION_SPACE_SIZE> all_scores = {0.0f};
    std::vector<std::array<float, 2>> position_seq = {{0.0f}};  // wrt delay
    std::vector<float> score_seq = {0.0f};  // wrt delay
    int t = 0;
    int action_delay;

    // CALL
    std::pair<std::array<float, 2>, bool> call() {

        // exit: the action have been provided #action_delay times
        if (t == action_delay) {
            /* LOG("[-] Plan finished"); */
            return std::make_pair(action, true);
        }
        /* LOG("[+] Plan step: " + std::to_string(t) + ", action: " + \ */
        /*     std::to_string(action[0]) + ", " + std::to_string(action[1])); */

        // provide the action
        this->t++;
        return std::make_pair(action, false);
    }

    void make_plan_positions(std::array<float, 2> position) {
        // calculate the position sequence

        this->position_seq = {position};
        this->score_seq = {score_seq[0]}; // previous score
        for (int j = 0; j < action_delay; j++) {

            // step
            position[0] += action[0];
            position[1] += action[1];
            this->position_seq.push_back(position);
            this->score_seq.push_back(all_scores[idx]);
        }
    }
    void reset() {
        this->action = {0.0f, 0.0f};
        this->all_values = {{0.0f}};
        this->all_scores = {0.0f};
        this->position_seq = {{0.0f}};
        this->score_seq = {0.0f};
        this->idx = 0;
        this->t = 0;
    }

    Plan(float action_delay = 1.0f): action_delay(action_delay) {}
    ~Plan() {}
};


class ExperienceModule {

    // external components
    Circuits& circuits;
    PCNN_REF& space;

    // internal components
    Plan plan;
    ActionSampler2D action_sampler;

    // parameters
    std::array<float, CIRCUIT_SIZE> weights;
    float speed;
    float action_delay;

    // one rollout
    void one_step_look_ahead(Eigen::VectorXf& curr_representation) {

        // initialize the plan
        action_sampler.reset();
        plan.reset();
        float best_score = -10000.0f;
        float max_neuron_value = 0.0f;

        // loop over the actions
        while (!action_sampler.is_done()) {

            // sample the one action | (action, done)
            std::array<float, 2> new_action = action_sampler.call();

            // obtain pc representation
            Eigen::VectorXf next_representation = space.simulate_one_step(new_action);

            // evaluate: (values, score)
            std::pair<std::array<float, CIRCUIT_SIZE>, float> \
                evaluation = evaluate_action(curr_representation,
                                             next_representation,
                                             action_sampler.get_counter());

            // check score
            if (evaluation.second > best_score) {
                best_score = evaluation.second;
                this->plan.action = {new_action[0] / action_delay,
                                     new_action[1] / action_delay};
                this->plan.idx = action_sampler.get_counter();
            }

            // record evaluation
            this->plan.all_values[action_sampler.get_counter()] = evaluation.first;
            this->plan.all_scores[action_sampler.get_counter()] = evaluation.second;
        }

        /* LOG("[+] best action: " + std::to_string(plan.action[0]) + ", " + \ */
        /*     std::to_string(plan.action[1]) + " | score: " + std::to_string(best_score)); */
    }

    // evaluate
    std::pair<std::array<float, CIRCUIT_SIZE>, float> \
        evaluate_action(Eigen::VectorXf& curr_representation,
                        Eigen::VectorXf& next_representation,
                        int action_idx) {

        // bnd, da, pmax
        std::array<float, CIRCUIT_SIZE> values_mod = circuits.call(
            next_representation, 0.0, 0.0, action_idx, true);

        // check for nans
        int i = 0;
        for (auto& v : values_mod) {
            if (std::isnan(v)) {
                LOG("Error: nan values | " + std::to_string(i));
            };
            i++;
        }

        // array
        std::array<float, CIRCUIT_SIZE> z = {weights[0] * values_mod[0],
                                             weights[1] * values_mod[1],
                                             weights[2] * values_mod[2],
                                             weights[3] * values_mod[3],
                                             weights[4] * values_mod[4]};

        // sum
        float output = 0.0f;
        for (auto& v : z) { output += v; }

        // add a bit of noise
        output += utils::random.get_random_float(0.0f, 0.01f);

        /* LOG("[+] output: " + std::to_string(output)); */

        return std::make_pair(z, output);
    }

public:
    bool new_plan = true;

    ExperienceModule(float speed,
                     Circuits& circuits,
                     PCNN_REF& space,
                     std::array<float, CIRCUIT_SIZE> weights,
                     float action_delay = 1.0f):
            action_sampler(ActionSampler2D(speed * action_delay)),
            speed(speed), circuits(circuits),
            space(space), weights(weights),
            plan(Plan(action_delay)),
            action_delay(action_delay) {}

    ~ExperienceModule() {}

    // CALL
    std::array<float, 2> call(std::string directive, Eigen::VectorXf& curr_representation) {

        if (directive == "new") {
            one_step_look_ahead(curr_representation);
            this->new_plan = true;
        } else {
            this->new_plan = false;
        }

        // step the plan | (action, done)
        std::pair<std::array<float, 2>, bool> next_step = plan.call();

        // check: plan finished -> new plan
        if (next_step.second) {
            one_step_look_ahead(curr_representation);
            this->new_plan = true;
            std::pair<std::array<float, 2>, bool> next_step = plan.call();
            return next_step.first;
        }

        // return the next action
        return next_step.first;
    }

    std::string str() { return "ExperienceModule"; }
    std::string repr() { return "ExperienceModule"; }
    std::array<std::array<float, 2>, ACTION_SPACE_SIZE> get_actions()
        { return action_sampler.action_set; }
    void set_plan_positions(std::array<float, 2> position)
        { plan.make_plan_positions(position); }
    std::vector<std::array<float, 2>> get_plan_positions()
        { return plan.position_seq; }
    std::array<float, ACTION_SPACE_SIZE> get_all_plan_scores()
        { return plan.all_scores; }
    std::array<std::array<float, CIRCUIT_SIZE>, ACTION_SPACE_SIZE> get_all_plan_values()
        { return plan.all_values; }
    std::pair<std::array<float, CIRCUIT_SIZE>, float> get_plan()
        { return {plan.all_values[plan.idx], plan.all_scores[plan.idx]}; }
    std::array<float, CIRCUIT_SIZE> get_plan_values()
        { return plan.all_values[plan.idx]; }
    float get_plan_score() { return plan.all_scores[plan.idx]; }
    std::vector<float> get_plan_scores() { return plan.score_seq; }
    int get_last_action_idx() { return plan.idx; }
};


/* ========================================== */
/* ================= BRAIN ================== */
/* ========================================== */


class Brain {

    // external components
    Circuits& circuits;
    PCNN_REF& space;
    ExperienceModule& expmd;
    TargetProgram trgp;

    // variables
    Eigen::VectorXf curr_representation;
    Eigen::VectorXf value_representation;
    std::string directive;

public:

    Brain(Circuits& circuits,
          PCNN_REF& space,
          /* TargetProgram& trgp, */
          ExperienceModule& expmd,
          float speed):
        circuits(circuits), space(space),
        expmd(expmd),
        trgp(TargetProgram(circuits.get_da_weights(), space, speed)),
        directive("new") {}

    ~Brain() {}

    // CALL
    std::array<float, 2> call(
            const std::array<float, 2>& velocity,
            float collision, float reward,
            bool trigger) {

        // === updates ===
        // :space
        auto [u, _] = space.call(velocity);
        space.update();
        this->curr_representation = u;

        // state
        LOG("---");
        LOG("rw: " + std::to_string(reward));
        LOG("cl: " + std::to_string(collision));
        LOG("tr: " + std::to_string(trigger));

        // :circuits
        std::array<float, CIRCUIT_SIZE> state_int = \
            circuits.call(u, collision, reward,
                          expmd.get_last_action_idx(), false);

        // :decision makin'
        value_representation = circuits.get_bnd_weights() * \
            circuits.get_memory_representation_mask();

        // === TRG PROGRAM ===

        // :target program
        trgp.set_wrec(space.get_connectivity());
        trgp.set_centers(space.get_centers());
        /* trgp.set_da_weights(circuits.get_da_weights()); */

        // === TRG PLAN ===
        if (trgp.is_active() && !trgp.is_plan_finished()) {
            /* LOG("Continue trg plan..."); */
            this->directive = "trg";
            return trgp.step_plan(curr_representation);
        } else if (trgp.is_active() && trgp.is_plan_finished()) {
            /* LOG("[-] trg plan finished"); */
            this->directive = "new";
        } else {
            // new trg plan?
            /* Eigen::VectorXf space_weights = Eigen::VectorXf::Ones(space.len()); */
            bool valid_plan = trgp.update(curr_representation,
                                          value_representation,
                                          trigger);
            if (valid_plan) {
                /* LOG("New trg plan..." + std::to_string(valid_plan)); */
                this->directive = "trg";
                return trgp.step_plan(curr_representation);
            }
        }

        // === DRIFT PLAN ===

        // check if the trg plan is done
        // :experience module
        if (collision > 0.0f) {
            this->directive = "new";
        } else {
            this->directive = "continue";
        }
        return expmd.call(directive, curr_representation);
    }

    std::string str() { return "Brain"; }
    std::string repr() { return "Brain"; }
    Eigen::VectorXf get_trg_representation() {
        Eigen::VectorXf trg_representation = Eigen::VectorXf::Zero(space.get_size());
        trg_representation(trgp.get_trg_idx()) = trgp.get_trg_value();
        return trg_representation;
    }
    Eigen::VectorXf get_representation()
        { return curr_representation; }
    ExperienceModule& get_expmd() { return expmd; }
    PCNN_REF& get_space() { return space; }

    std::string get_directive() { return directive; }
    std::vector<int> get_trg_plan() { return trgp.get_plan(); }
    std::array<float, 2> get_space_position() { return space.get_position(); }
    void set_plan_positions(std::array<float, 2> position)
        { expmd.set_plan_positions(position); }
    std::vector<std::array<float, 2>> get_plan_positions(std::array<float, 2> position) {
        if (expmd.new_plan) { expmd.set_plan_positions(position); }
        return expmd.get_plan_positions();
    }
    float get_plan_score()
        { return expmd.get_plan_score(); }
    std::vector<float> get_plan_scores()
        { return expmd.get_plan_scores(); }
    std::array<float, CIRCUIT_SIZE>  get_plan_values()
        { return expmd.get_plan_values(); }
    void reset() {
        trgp.reset();
        circuits.reset();
    }
};



/* ========================================== */

namespace pcl {


void testSampling() {

    ActionSampler2D sm = ActionSampler2D(2.0f);

    LOG("[Sampling test]");

    sm.reset();
    std::array<float, 2> action = {0.0f, 0.0f};
    for (int i = 0; i < 28; i++) {
        action = sm.call();

        if (i == (8 + 3)) {
            LOG("resetting...");
            sm.reset();
        };

        LOG("[" + std::to_string(sm.get_counter()) + "] " + \
            std::to_string(action[0]) + ", " + std::to_string(action[1]));
    };
};


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
    BaseModulation da = BaseModulation("DA", N, 0.5f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f);
    BaseModulation bnd = BaseModulation("BND", N, 0.9f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f);
    MemoryRepresentation memrepr = MemoryRepresentation(N, 2.0f, 0.1f);
    MemoryAction memact = MemoryAction(2.0f);
    Circuits circuits = Circuits(da, bnd, memrepr, memact);

    // TARGET PROGRAM
    /* TargetProgram trgp = TargetProgram(space.get_connectivity(), space.get_centers(), */
    /*                                    da.get_weights(), SPEED); */
    // EXPERIENCE MODULE & BRAIN
    ExperienceModule expmd = ExperienceModule(SPEED, circuits, space, 
                                              {bnd_w, 0.0f, 0.0f, 0.0f, 0.0f}, 1.0f);
    Brain brain = Brain(circuits, space, expmd, SPEED);

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
