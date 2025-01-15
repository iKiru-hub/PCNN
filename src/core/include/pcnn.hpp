#include <iostream>
#include <Eigen/Dense>
#include "utils.hpp"
#include <unordered_map>
#include <memory>
#include <array>
#include <algorithm>
#include <cstdio>
#include <tuple>
#include <cassert>


/* ========================================== */

/* #define LOG(msg) utils::logging.log(msg, "PCLIB") */
#define SPACE utils::logging.space
#define GCL_SIZE 9
#define GCL_SIZE_SQRT 3
#define PCNN_REF PCNNbase
#define CIRCUIT_SIZE 3

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
    std::array<float, 2> bounds;  // it's square

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
        std::vector<float> line = utils::linspace_vec(
                        bounds[0], bounds[1], GCL_SIZE_SQRT,
                        true, false);

        for (std::size_t i=0; i<GCL_SIZE; i++) {
            float xi = line[i / GCL_SIZE_SQRT];
            float yi = line[i % GCL_SIZE_SQRT];
            basis[i][0] = xi;
            basis[i][1] = yi;
        }
    }

    // define boundary type
    void boundary_conditions(std::array<std::array<float, 2>,
                             GCL_SIZE>& _positions) {
        for (int i = 0; i < GCL_SIZE; i++) {
            std::array<float, 2> new_position = \
                apply_boundary(
                            _positions[i][0],
                            _positions[i][1]);
            _positions[i][0] = new_position[0];
            _positions[i][1] = new_position[1];
        }
    }

    // define boundary type
    void boundary_conditions() {
        for (int i = 0; i < GCL_SIZE; i++) {
            std::array<float, 2> new_position = \
                apply_boundary(
                            positions[i][0],
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
        if (y < bounds[0]) { y += 2.0f*std::abs(bounds[0]); }
        else if (y > bounds[1]) { y -= 2.0f*bounds[1]; }

        return {x, y};
    }

public:

    GridLayerSq(float sigma, float speed,
                std::array<float, 2> bounds = \
                    {-1.0, 1.0}):
        sigma(sigma), speed(speed), bounds(bounds) {

        // record positions
        square_basis();

        // record initial positions in the basis
        // pass by value
        positions = basis;
    }

    ~GridLayerSq() {} // LOG("[-] GridLayer destroyed"); }

    // @brief call the GridLayer with a 2D input
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

    std::array<float, GCL_SIZE> fwd_position(
        const std::array<float, 2>& v) {

        std::array<std::array<float, 2>, GCL_SIZE> new_positions;
        for (int i = 0; i < GCL_SIZE; i++) {
            new_positions[i][0] = positions[i][0] + \
                speed * (v[0] - positions[i][0]);
            new_positions[i][1] = positions[i][1] + \
                speed * (v[1] - positions[i][1]);
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

    ~GridNetworkSq() { LOG("[-] GridNetworkSq destroyed"); }

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

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& v) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {

            // Convert the output of layers[i].call(x) to
            // an Eigen::VectorXf
            Eigen::VectorXf layer_output = \
                Eigen::Map<const Eigen::VectorXf>(
                    layers[i].fwd_position(v).data(), GCL_SIZE);

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
        centers = Eigen::MatrixXf::Zero(N, 2);
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


class PCNNsq {
public:
    PCNNsq(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         GridNetworkSq& xfilter, std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(xfilter),
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

        ach = 1.0;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};
    }

    ~PCNNsq() { LOG("[-] PCNN destroyed"); }

    // CALL
    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v,
                          const bool frozen = false,
                          const bool traced = true) {

        // pass the input through the filter layer
        x_filtered = xfilter.call(v);

        // forward it to the network by doing a dot product
        // with the feedforward weights
        u = Wff * x_filtered + pre_x;
        u = utils::cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = Eigen::VectorXf::Constant(u.size(),
                                                          0.01);

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
        Eigen::VectorXf dw = x_filtered - Wff.row(idx).transpose();

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
                return void();
            }

            // update count and backup
            cell_count++;
            Wffbackup.row(idx) = Wff.row(idx);
            LOG("(:)cell_count: " + std::to_string(cell_count) + \
                " [" + std::to_string(similarity) + ", idx=" + \
                std::to_string(idx) + "]");

            // update recurrent connections
            update_recurrent();

            // record new center
            centers.row(idx) = Eigen::Vector2f(x, y);
        }

    }

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
    std::string str() const { return "PCNN." + name; }
    std::string repr() const {
        return "PCNN(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
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

    GridNetworkSq xfilter;

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
             int num_neighbors, float trace_tau,
             GridNetworkSq xfilter,
             int length,
             std::string name = "2D")
        : N(N), Nj(Nj), gain(gain), offset(offset),
        clip_min(clip_min), threshold(threshold),
        rep_threshold(rep_threshold),
        rec_threshold(rec_threshold),
        num_neighbors(num_neighbors),
        trace_tau(trace_tau),
        xfilter(std::move(xfilter)),
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

        ach = 1.0;

        this->fixed_centers = utils::generate_lattice(N, length);

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};
    }

    ~PCNNbase() { LOG("[-] PCNN destroyed"); }

    // CALL
    std::pair<Eigen::VectorXf,
    Eigen::VectorXf> call(const std::array<float, 2>& v,
                          const bool frozen = false,
                          const bool traced = true) {

        // give a position v, calculate the activation
        // as a gaussian distance
        Eigen::VectorXf y = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < N; i++) {
            y(i) = std::exp(-((v[0] - fixed_centers(i, 0)) *
                              (v[0] - fixed_centers(i, 0)) +
                              (v[1] - fixed_centers(i, 1)) *
                              (v[1] - fixed_centers(i, 1))));
        }

        y = utils::generalized_sigmoid_vec(y, offset,
                                           gain, clip_min);

        return std::make_pair(y, x_filtered);
    }

    // @brief update the model
    void update(float x = -1.0, float y = -1.0) {};

    Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) {

        // maybe use cosine similarity?
        std::pair<Eigen::VectorXf, Eigen::VectorXf> res = \
            call(x, true, false);

        return res.first;
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
    std::string str() const { return "PCNN." + name; }
    std::string repr() const {
        return "PCNN(" + name + std::to_string(N) + \
            std::to_string(Nj) + std::to_string(gain) + \
            std::to_string(offset) + \
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
    Eigen::MatrixXf get_basis() {
        return fixed_centers;
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
    const int length;
    const float gain;
    const float offset;
    const float clip_min;
    const float threshold;
    const float rep_threshold;
    const float rec_threshold;
    const int num_neighbors;
    const float trace_tau;
    const std::string name;
    Eigen::MatrixXf fixed_centers;

    float ach;

    GridNetworkSq xfilter;

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


/* ========================================== */
/* =========== MODULATION MODULES =========== */
/* ========================================== */

// leaky variable

class LeakyVariableND {
public:

    LeakyVariableND(std::string name, float eq, float tau,
                    int ndim, float min_v = 0.0)
        : name(std::move(name)), tau(1.0f / tau), eq_base(eq),
        ndim(ndim), min_v(min_v),
        v(Eigen::VectorXf::Constant(ndim, eq)),
        eq(Eigen::VectorXf::Constant(ndim, eq)){

        LOG("[+] LeakyVariableND created with name: " + this->name);
    }

    ~LeakyVariableND() {
        LOG("[-] LeakyVariableND destroyed with name: " + name);
    }

    /* @brief Call the LeakyVariableND with a 2D input
     * @param x A 2D input to the LeakyVariable */
    Eigen::VectorXf call(const Eigen::VectorXf x,
                         const bool simulate = false) {

        // simulate
        if (simulate) {
            Eigen::VectorXf z = v + (eq - v) * tau + x;
            for (int i = 0; i < ndim; i++) {
                if (z(i) < min_v) {
                    z(i) = min_v;
                }
            }
            return z;
        }

        // Compute dv and update v
        v += (eq - v) * tau + x;
        return v;
    }

    Eigen::VectorXf get_v() const { return v; }
    void print_v() const {
        std::cout << "v: " << v.transpose() << std::endl; }
    std::string str() const { return "LeakyVariableND." + name; }
    int len() const { return ndim; }
    std::string repr() const {
        return "LeakyVariableND." + name + "(eq=" + \
            std::to_string(eq_base) + ", tau=" + std::to_string(tau) + \
            ", ndim=" + std::to_string(ndim) + ")";
    }
    std::string get_name() { return name; }
    void set_eq(const Eigen::VectorXf& eq) { this->eq = eq; }
    void reset() {
        for (int i = 0; i < ndim; i++) {
            v(i) = eq(i);
        }
    }

private:
    const float tau;
    const int ndim;
    const float min_v;
    const float eq_base;
    std::string name;
    Eigen::VectorXf v;
    Eigen::VectorXf eq;
};


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

        LOG("[+] LeakyVariable1D created with name: " + \
            this->name);
    }

    ~LeakyVariable1D() {
        LOG("[-] LeakyVariable1D destroyed with name: " + name);
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


// density modulation

class DensityMod {

public:

    DensityMod(std::array<float, 5> weights,
               float theta):
        weights(weights), theta(theta), baseline(theta) {}

    ~DensityMod() {}

    float call(const std::array<float, 5>& x) {
        dtheta = 0.0;
        for (size_t i = 0; i < 5; i++) {
            dtheta += x[i] * weights[i];
        }
        theta = baseline + utils::generalized_tanh(
            dtheta, 0.0, 1.0);
        return theta;
    }
    std::string str() { return "DensityMod"; }
    float get_value() { return theta; }

private:
    std::array<float, 5> weights;
    float baseline;
    float theta;
    float dtheta;
};


// base modulation | Dopamine & Boundary
class BaseModulation{
    float output;
    std::string name;
    int size;
    float lr;
    float threshold;
    /* std::vector<float> weights; */
    Eigen::VectorXf weights;
    LeakyVariable1D leaky;
    GSparams gsparams;

public:

    BaseModulation(std::string name, int size, float lr,
             float threshold, float offset, float gain,
             float clip, float eq, float tau,
             float min_v = 0.0f):
        name(name), size(size),
        threshold(threshold), gsparams(offset, gain, clip),
        lr(lr), leaky(LeakyVariable1D(name, eq, tau, min_v))
    { weights = Eigen::VectorXf::Zero(size); }

    ~BaseModulation() {}

    /* float call(const std::vector<float>& u, */
    /*            float x = 0.0f, bool simulate = false) { */
    float call(const Eigen::VectorXf& u,
               float x = 0.0f, bool simulate = false) {

        // forward to the leaky variable
        float v = leaky.call(x, simulate);

        // update the weights
        if (!simulate) {
            for (int i = 0; i < size; i++) {
                float ui = u[i] > threshold ? u[i] : 0.0;
                weights[i] += lr * v * ui;

                // clip the weights in (0, 1)
                if (weights[i] < 0.0) {
                    weights[i] = 0.0;
                } else if (weights[i] > 1.0) {
                    weights[i] = 1.0;
                }
            }
        }

        // compute the output
        output = 0.0;
        for (int i = 0; i < size; i++) {
            output += weights[i] * u[i];
        }

        // apply activation function
        output = utils::generalized_sigmoid(output,
                                            gsparams.offset,
                                            gsparams.gain,
                                            gsparams.clip);

        return output;
    }

    float get_output() { return output; }
    /* std::vector<float> get_weights() { return weights; } */
    Eigen::VectorXf get_weights() {
        Eigen::VectorXf w(size);
        for (int i = 0; i < size; i++) {
            w(i) = weights[i];
        }
        return w;
    }
    float get_leaky_v() { return leaky.get_v(); }
    std::string str() { return name; }
    std::string repr() { return name + "(1D)"; }
    int len() { return size; }
};


// program 1 : maximum activity
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


class Circuits {

    BaseModulation& da;
    BaseModulation& bnd;
    PopulationMaxProgram pmax;

    std::array<float, CIRCUIT_SIZE> output;

public:

    Circuits(BaseModulation& da, BaseModulation& bnd):
        da(da), bnd(bnd), pmax(PopulationMaxProgram()) {}

    ~Circuits() {}

    // CALL
    std::array<float, 3> call(Eigen::VectorXf& u,
                              float collision,
                              float reward,
                              bool simulate = false) {

        output[0] = bnd.call(u, collision, simulate);
        output[1] = da.call(u, reward, simulate);
        output[2] = pmax.call(u);

        return output;
    }

    std::string str() { return "Circuits"; }
    std::string repr() { return "Circuits"; }
    int len() { return 3; }
    std::array<float, 3> get_output() { return output; }
    std::array<float, 2> get_leaky_v() {
        return {da.get_leaky_v(), bnd.get_leaky_v()}; }
};



// target program
class TargetProgram {

    float threshold1;
    float threshold2;
    bool active;
    int max_depth;
    int size;
    Eigen::VectorXf trg_representation;
    BaseModulation& modulator;
    Eigen::MatrixXf wrec;

    std::tuple<Eigen::VectorXf, bool> converge_to_location(
        Eigen::VectorXf representation,
        int depth, Eigen::VectorXf& modulation) {

        // recurrent step in the space
        /* Eigen::VectorXf u = space.fwd_int(representation); */
        Eigen::VectorXf u = wrec * representation + modulation;

        float similarity = utils::cosine_similarity_vec(
            representation, u);

        if (similarity > threshold2) {
            return std::make_tuple(u, true);
        } else if (depth > max_depth) {
            return std::make_tuple(u, false);
        }

        return converge_to_location(u, depth+1, modulation);
    }

public:

    TargetProgram(float threshold1,
                  Eigen::MatrixXf wrec,
                  BaseModulation& modulator,
                  int max_depth = 20,
                  float threshold2=0.8):
        threshold1(threshold1), threshold2(threshold2),
        wrec(wrec), modulator(modulator),
        active(false), size(wrec.rows()), max_depth(max_depth) {

        size = wrec.rows();
        trg_representation = Eigen::VectorXf::Zero(size);

        LOG("[+] TargetProgram created");
    }

    ~TargetProgram() { LOG("[-] TargetProgram destroyed"); }

    // attempt to define a target representation
    void update(float activation) {

        /*
        trg_repr <- converge(modulation)
        enabled
        --
        value <- compare velocities()
        */

        active = activation > threshold1;
        Eigen::VectorXf modulation = modulator.get_weights();

        if (active) {
            Eigen::VectorXf representation = \
                Eigen::VectorXf::Zero(size);
            auto [trg_representation, active] = \
                converge_to_location(representation,
                                     0, modulation);
            /* trg_representation = representation; */
            /* active = false; */
        };
    }

    // quantify how similar a given representation is to the
    // target representation
    float evaluate(Eigen::VectorXf& next_representation,
                   Eigen::VectorXf& curr_representation) {

        if (!active) { return 0.0f; };
        Eigen::VectorXf next_vector = curr_representation - \
            trg_representation;
        Eigen::VectorXf trg_vector = next_representation - \
            trg_representation;
        return utils::cosine_similarity_vec(next_vector,
                                            trg_vector);
    }

    Eigen::VectorXf get_trg_representation() {
        return trg_representation;
    };

    /* float get_trigger_var() { return trigger_var; } */

    std::string str() { return "TargetProgram"; }
    std::string repr() { return "TargetProgram"; }
    int len() { return 1; }
    bool is_active() { return active; }
    void set_wrec(Eigen::MatrixXf wrec) {
        this->wrec = wrec;
    }
};


/* ========================================== */
/* ========= ACTION SAMPLING MODULE ========= */
/* ========================================== */


class ActionSampling2D {

public:

    std::string name;

    ActionSampling2D(std::string name,
                     float speed) : name(name)
                {
        update_actions(speed);
        /* utils::logging.log("[+] ActionSampling2D." + name); */
    }

    ~ActionSampling2D() {
        /* utils::logging.log("[-] ActionSampling2D." + name); */
    }

    std::tuple<std::array<float, 2>, bool, int> call(bool keep = false) {

        // Keep current state
        if (keep) {
            utils::logging.log("-- keep");
            return std::make_tuple(velocity, false, idx);
        }

        // All samples have been used
        if (counter == num_samples) {

            // try to sample a zero index
            int zero_idx = sample_zero_idx();

            if (zero_idx != -1) {
                idx = zero_idx;
            } else {
                // Get the index of the maximum value
                idx = utils::arr_argmax(values);
            }

            velocity = samples[idx];
            return std::make_tuple(velocity, true, idx);
        }

        // Sample a new index
        idx = sample_idx();
        available_indexes[idx] = false;
        velocity = samples[idx];

        return std::make_tuple(velocity, false, idx);
    }

    void update(float score = 0.0f) { values[idx] = score; }
    bool is_done() { return counter == num_samples; }
    std::string str() { return "ActionSampling2D." + name; }
    std::string repr() { return "ActionSampling2D." + name; }
    int len() const { return num_samples; }
    const int get_idx() { return idx; }
    const int get_counter() { return counter; }

    const float get_max_value() {
        if (counter == 0) { return 0.0; }
        return values[idx];
    }

    // @brief get values for the samples
    const std::array<float, 8> get_values() { return values; }

    // @brief random one-time sampling
    std::array<float, 2> sample_once() {
        int idx = utils::random.get_random_int(0, num_samples);
        return samples[idx];
    }

    void reset() {
        idx = -1;
        for (int i = 0; i < num_samples; i++) {
            available_indexes[i] = true;
        }
        values = { 0.0 };
        counter = 0;
    }

private:

    // parameters | trying with only 8 directions (no stop)
    std::array<float, 2> samples[8] = {
        {-1.0, 1.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {-1.0, 0.0},
        {1.0, 0.0},
        {-1.0, -1.0},
        {0.0, -1.0},
        {1.0, -1.0}
    };
    const std::array<unsigned int, 8> indexes = { 0, 1, 2, 3, 4,
                                          5, 6, 7};
    unsigned int counter = 0;
    const unsigned int num_samples = 8;
    std::array<float, 8> values = { 0.0 };
    std::array<float, 2> velocity = { 0.0 };
    int idx = -1;

    // @brief variables
    /* int idx = -1; */
    std::array<bool, 8> available_indexes = { true, true, true,
        true, true, true, true, true};

    // @brief sample a random index
    int sample_idx() {

        int idx = -1;
        bool found = false;
        while (!found) {
            int i = utils::random.get_random_int(0, num_samples);

            // Check if the index is available
            if (available_indexes[i]) {
                idx = i;  // Set idx to the found index
                found = true;  // Mark as found
            };
        };
        counter++;
        return idx;
    }

    // @brief make a set of how values equal to zero
    // and return a random index from it
    int sample_zero_idx() {

        std::vector<int> zero_indexes;
        for (size_t i = 0; i < num_samples; i++) {
            if (values[i] == 0.0) {
                zero_indexes.push_back(i);
            }
        }

        if (zero_indexes.size() > 1) {
            return utils::random.get_random_element_vec(zero_indexes);
        }

        return -1;
    }

    // @brief update the actions given a speed
    void update_actions(float speed) {

        for (size_t i = 0; i < num_samples; i++) {
            float dx = samples[i][0];
            float dy = samples[i][1];
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
            samples[i] = {dx, dy};
        }
    }

};


struct TwoLayerNetwork {

    // @brief forward an input through the network
    // @return a tuple: (float, array<float, 2>)
    std::tuple<float, std::array<float, 2>>
    call(const std::array<float, 5>& x) {
        hidden = {0.0, 0.0};
        output = 0.0;

        // hidden layer
        for (size_t i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                hidden[i] += x[j] * w_hidden[i][j];
            }
        }

        // output layer
        for (size_t i = 0; i < 2; i++) {
            output += hidden[i] * w_output[i];
        }

        return std::make_tuple(output, hidden);
    }

    TwoLayerNetwork(std::array<std::array<float, 2>, 5> w_hidden,
                   std::array<float, 2> w_output)
        : w_hidden(w_hidden), w_output(w_output) {}
    ~TwoLayerNetwork() {}
    std::string str() { return "TwoLayerNetwork"; }

private:

    //  matrix 5x2
    const std::array<std::array<float, 2>, 5> w_hidden;
    const std::array<float, 2> w_output;
    std::array<float, 2> hidden;
    float output;
};


struct OneLayerNetwork {

    // circuit size + target program
    const std::array<float, CIRCUIT_SIZE+1> weights;
    float output;
    int size;


    // @brief forward an input through the network
    // @return tuple(float, array<float, 2>)
    std::pair<std::array<float, CIRCUIT_SIZE+1>, float>
    call(const std::array<float, CIRCUIT_SIZE+1>& x) {

        output = 0.0f;
        std::array<float, CIRCUIT_SIZE+1> z = {0.0f};
        for (size_t i = 0; i < size; i++) {
            output += x[i] * weights[i];
            z[i] = x[i] * weights[i];
        }
        return std::make_pair(z, output);
    }

    OneLayerNetwork(std::array<float, CIRCUIT_SIZE+1> weights)
        : weights(weights) {
        size = weights.size();
    }
    ~OneLayerNetwork() {}
    std::string str() { return "OneLayerNetwork"; }
    int len() { return size; }
    std::array<float, CIRCUIT_SIZE+1> get_weights() { return weights; }
};


/* ========================================== */
/* =========== EXPERIENCE MODULE ============ */
/* ========================================== */

struct Plan {
    std::array<float, 2> action;
    std::vector<std::array<float, 2>> action_seq;
    std::vector<std::array<float, CIRCUIT_SIZE+1>> value_seq;
    std::vector<float> score_seq;
    const int max_depth;
    int depth;
    int t;

    bool is_done() {
        return t == depth;
    }
    bool is_last() {
        return t == (depth - 1);
    }

    // STEP
    std::array<float, 2> step() {
        /* t = is_last() ? t : t + 1; */
        t++;
        /* std::cout << "t: " << t << std::endl; */
        this->action = action_seq[t];
        return action;
    }
    void renew(std::tuple<std::vector<std::array<float, 2>>,
               std::vector<std::array<float, CIRCUIT_SIZE+1>>,
               std::vector<float>,
              int> data) {
        this->action_seq = std::get<0>(data);
        this->value_seq = std::get<1>(data);
        this->score_seq = std::get<2>(data);
        this->depth = std::get<3>(data);
        t = 0;
    }

    Plan(const int max_depth = 10):
        max_depth(max_depth) {
        this->t = 0;
        this->action = {0.0f, 0.0f};
        /* std::vector<std::array<float, 2>> action_seq(depth); */
        this->action_seq = {{0.0f}};
        this->value_seq = {{0.0f}};
        this->score_seq = {};
        this->depth = 0;
    }
    ~Plan() {}
};


struct PlanningPolicy {

    OneLayerNetwork& evaluation_network;
    float score_threshold;

    PlanningPolicy(OneLayerNetwork& evaluation_network,
                   float score_threshold):
        evaluation_network(evaluation_network),
        score_threshold(score_threshold) {}
    ~PlanningPolicy() {}

    std::pair<std::array<float, CIRCUIT_SIZE+1>, float> evaluate(
        std::array<float, CIRCUIT_SIZE+1>& values) {
        std::pair<std::array<float, CIRCUIT_SIZE+1>, float> results = \
            evaluation_network.call(values);

        // check if the score is above the threshold
        if (results.second < score_threshold) {
            return {results.first, 0.0f};
        }
        return results;
    }

};


class ExperienceModule {

    // components
    ActionSampling2D action_space_one;
    ActionSampling2D action_space_two;
    Circuits& circuits;
    TargetProgram& trgp;
    PCNN_REF& space;
    OneLayerNetwork& eval_network;
    const float speed;
    const int action_delay;


    // variables
    Plan plan;

    std::tuple<std::vector<std::array<float, 2>>,
               std::vector<std::array<float, CIRCUIT_SIZE+1>>,
               std::vector<float>,
               int> \
        main_rollout(Eigen::VectorXf curr_representation,
                     std::array<float, 2>& position) {

        action_space_one.reset();

        std::vector<std::array<float, 2>> new_action_seq_f = {{0.0f}};
        std::vector<std::array<float, CIRCUIT_SIZE+1>> values_seq_f = {{0.0f}};
        std::vector<float> scores_seq_f = {0.0f};  // for each action
        int depth_f = 0;
        float best_score = -1000.0f;
        all_values = {0.0f};

        // outer loop
        while (!action_space_one.is_done()) {

            // sample the next action
            std::tuple<std::array<float, 2>, bool, int> action_data = \
                action_space_one.call();

            // step and evaluate
            position[0] += std::get<0>(action_data)[0];
            position[1] += std::get<0>(action_data)[1];
            Eigen::VectorXf next_representation = \
                space.fwd_ext(position);
            /* Eigen::VectorXf next_representation = \ */
            /*     space.fwd_ext(std::get<0>(action_data)); */

            std::pair<std::array<float, CIRCUIT_SIZE+1>, float> \
                evaluation = evaluate_action(curr_representation,
                                             next_representation);

            // update the provisional plan
            std::vector<std::array<float, 2>> new_action_seq;
            std::vector<std::array<float, CIRCUIT_SIZE+1>> values_seq;
            std::vector<float> scores_seq;  // for each action
            int depth = 1;
            new_action_seq.push_back(std::get<0>(action_data));
            values_seq.push_back(evaluation.first);
            scores_seq.push_back(evaluation.second);

            curr_representation = next_representation;

            // inner loop
            inner_rollout(curr_representation,
                          new_action_seq,
                          values_seq,
                          scores_seq,
                          depth,
                          position);

            // check if the average score is the best
            float avg_score = std::accumulate(
                scores_seq.begin(), scores_seq.end(), 0.0) / \
                scores_seq.size();
            all_values[action_space_one.get_idx()] = avg_score;
            if (avg_score > best_score) {
                new_action_seq_f = new_action_seq;
                values_seq_f = values_seq;
                scores_seq_f = scores_seq;
                depth_f = depth;
                best_score = avg_score;
            }
        }

        return {new_action_seq_f, values_seq_f, scores_seq_f, depth_f};
    }

    void inner_rollout(Eigen::VectorXf curr_representation,
                       std::vector<std::array<float, 2>>& new_action_seq,
                       std::vector<std::array<float, CIRCUIT_SIZE+1>>& values_seq,
                       std::vector<float>& scores_seq,
                       int& depth,
                       std::array<float, 2>& position) {

        action_space_two.reset();

        // loop
        while (!action_space_two.is_done()) {

            // sample the next action
            std::tuple<std::array<float, 2>, bool, int> action_data = \
                action_space_two.call();

            // step
            position[0] += std::get<0>(action_data)[0];
            position[1] += std::get<0>(action_data)[1];
            Eigen::VectorXf next_representation = \
                space.fwd_ext(position);
            /* Eigen::VectorXf next_representation = \ */
            /*     space.fwd_ext(std::get<0>(action_data)); */

            // evaluate
            std::pair<std::array<float, CIRCUIT_SIZE+1>, float> \
                evaluation = evaluate_action(curr_representation,
                                             next_representation);

            // check 1: max depth
            if (depth == plan.max_depth) {
                break;
            }

            // check 2: boundary sensor too high
            if (evaluation.first[0] < -0.99) {
                break;
            }

            // update the provisional plan
            new_action_seq.push_back(std::get<0>(action_data));
            values_seq.push_back(evaluation.first);
            scores_seq.push_back(evaluation.second);
            curr_representation = next_representation;
            depth++;
        }
    }

    std::pair<std::array<float, CIRCUIT_SIZE+1>, float> \
        evaluate_action(Eigen::VectorXf& curr_representation,
                        Eigen::VectorXf& next_representation) {

        // bnd, da, pmax
        std::array<float, 3> values_mod = circuits.call(
            next_representation, 0.0, 0.0, true);

        // trg
        float value_trg = trgp.evaluate(
            curr_representation, next_representation);

        // values
        std::array<float, CIRCUIT_SIZE+1> values = \
                {values_mod[0], values_mod[1],
                 values_mod[2], value_trg};

        // final score from the network
        std::pair<std::array<float, CIRCUIT_SIZE+1>, float> score = \
            eval_network.call(values);

        return score;
    }

    bool check_step() { return true; }

public:
    bool new_plan;
    std::array<float, 8> all_values;

    ExperienceModule(float speed,
                     Circuits& circuits,
                     TargetProgram& trgp,
                     PCNN_REF& space,
                     OneLayerNetwork& eval_network,
                     int action_delay = 1):
            action_space_one(ActionSampling2D("action_space_one",
                                              speed)),
            action_space_two(ActionSampling2D("action_space_two",
                                              speed)),
            speed(speed), circuits(circuits),
            trgp(trgp), space(space), eval_network(eval_network),
            plan(Plan()), action_delay(action_delay) {
        new_plan = false;
        all_values = {0.0f};
    }

    ~ExperienceModule() {}

    // CALL
    std::pair<std::array<float, 2>, bool> call(std::string& directive,
                                               std::array<float, 2> position) {

        if (directive == "new") {
            std::tuple<std::vector<std::array<float, 2>>,
                       std::vector<std::array<float, CIRCUIT_SIZE+1>>,
                       std::vector<float>,
                       int> rollout_data = main_rollout(
                            space.get_activation(),
                            position);
            plan.renew(rollout_data);
            new_plan = true;
        };

        new_plan = false;

        // return the next action and whether the plan is done
        return {plan.step(), plan.is_last()};
    }

    std::string str() { return "ExperienceModule"; }
    std::string repr() { return "ExperienceModule"; }
    std::vector<std::array<float, 2>> get_action_seq() {
        return plan.action_seq;
    }
    std::array<float, 8> get_values() {
        return all_values;
    }
    std::tuple<std::vector<std::array<float, 2>>,
               std::vector<std::array<float, CIRCUIT_SIZE+1>>,
               std::vector<float>,
               int> get_plan() {
        return {plan.action_seq, plan.value_seq,
                plan.score_seq, plan.depth};
    }

};



/* ========================================== */
/* ================= BRAIN ================== */
/* ========================================== */


class BrainHex {

    /* BaseModulation& modulator; */
    Circuits& circuits;
    PCNNgridhex& space;
    ActionSampling2D& action_space;
    TargetProgram& trgp;

public:

    BrainHex(Circuits& circuits,
          PCNNgridhex& space,
          ActionSampling2D& action_space,
          TargetProgram& trgp):
        circuits(circuits),
        space(space),
        action_space(action_space),
        trgp(trgp){}

    ~BrainHex() {}

    // with Eigen
    std::array<float, 2> call(const std::array<float, 2>& v,
                              float collision,
                              float reward = 0.0f,
                              std::array<float, 2> position = {-1.0,
                              -1.0}) {

        // update: space
        auto [u, _] = space.call(v);
        space.update(position[0], position[1]);

        // update: circuits 
        std::array<float, 3> state_int = \
            circuits.call(u, collision, reward);

        // update: trgp
        trgp.set_wrec(space.get_wrec());
        trgp.update(state_int[1]);

        // get the action
        std::array<float, 2> action = action_space.sample_once();
        return {action[0], action[1]};
        /* return action_space.sample_once(); */
    }

    std::string str() { return "Brain"; }
    std::string repr() { return "Brain"; }
    Eigen::VectorXf get_trg_representation() {
        return trgp.get_trg_representation();
    }
    Eigen::VectorXf get_representation() {
        return space.get_activation();
    }

};


class Brain {

    /* BaseModulation& modulator; */
    Circuits& circuits;
    PCNN_REF& space;
    TargetProgram& trgp;
    ExperienceModule& expmd;
    Eigen::VectorXf curr_representation;

    std::string directive;

public:

    Brain(Circuits& circuits,
          PCNN_REF& space,
          TargetProgram& trgp,
          ExperienceModule& expmd):
        circuits(circuits),
        space(space),
        trgp(trgp),
        expmd(expmd),
        directive("new"){}

    ~Brain() {}

    // with Eigen
    std::array<float, 2> call(const std::array<float, 2>& v,
                              float collision,
                              float reward = 0.0f,
                              std::array<float, 2> position = {-1.0,
                              -1.0}) {

        // === updates ===
        // :space
        auto [u, _] = space.call(v);
        space.update(position[0], position[1]);

        curr_representation = u;

        // :circuits 
        std::array<float, 3> state_int = \
            circuits.call(u, collision, reward);

        // :trgp
        trgp.set_wrec(space.get_wrec());
        trgp.update(state_int[1]);

        // === plan ===

        // :experience module
        /* LOG("directive: " + directive); */
        std::pair<std::array<float, 2>, bool> action_data = \
            expmd.call(directive, position);

        // check if the plan is done
        if (action_data.second || collision > 0.0f) {
            directive = "new";
        } else {
            directive = "continue";
        }

        // === output ===
        std::array<float, 2> action = action_data.first;

        return {action[0], action[1]};
    }

    std::string str() { return "Brain"; }
    std::string repr() { return "Brain"; }
    Eigen::VectorXf get_trg_representation() {
        return trgp.get_trg_representation();
    }
    Eigen::VectorXf get_representation() {
        return curr_representation;
    }
    std::string get_directive() { return directive; }

    ExperienceModule& get_expmd() { return expmd; }
};


/* ========================================== */
/* ========================================== */


namespace pcl {

/* void test_layer() { */

/*     std::array<float, 4> bounds = {0.0, 1.0, 0.0, 1.0}; */

/*     PCLayer layer = PCLayer(3, 0.1, bounds); */
/*     LOG(layer.str()); */
/*     LOG(std::to_string(layer.len())); */

/* }; */


void testSampling() {

    ActionSampling2D sm = ActionSampling2D("Test", 10);

    sm.str();

    bool keep = false;
    for (int i = 0; i < 28; i++) {
        sm.call(keep);
        if (!sm.is_done()) {
            sm.update(utils::random.get_random_float());
        };

        if (i == (sm.len() + 3)) {
            LOG("resetting...");
            sm.reset();
        };
    };
};


void test_leaky() {

    SPACE("#---#");

    LOG("Testing LeakyVariable...");


    SPACE("#---#");
}


/* void test_pcnn() { */

/*     SPACE("#---#"); */

/*     LOG("Testing PCNN..."); */

/*     int n = 3; */
/*     int Nj = std::pow(n, 2); */

/*     PCLayer xfilter = PCLayer(n, 0.1, {0.0, 1.0, 0.0, 1.0}); */
/*     /1* RandLayer xfilter = RandLayer(Nj); *1/ */
/*     LOG(xfilter.str()); */

/*     PCNN model = PCNN(3, Nj, 10., 0.1, 0.4, 0.1, 0.1, */
/*                       0.5, 8, 0.1, xfilter, "2D"); */

/*     LOG(model.str()); */

/*     LOG("-- input 1"); */
/*     Eigen::Vector2f x = {0.2, 0.2}; */
/*     Eigen::VectorXf y = model.call(x); */
/*     model.update(); */
/*     LOG("model length: " + std::to_string(model.len())); */

/*     LOG("-- input 2"); */

/*     x = {0.1, 0.1}; */
/*     y = model.call(x); */
/*     model.update(); */
/*     LOG("model length: " + std::to_string(model.len())); */

/*     LOG("---"); */
/*     LOG("connectivity:"); */
/*     utils::logging.log_matrix(model.get_connectivity()); */

/*     LOG("wrec:"); */
/*     utils::logging.log_matrix(model.get_wrec()); */

/*     SPACE("#---#"); */
/* } */


/* void test_randlayer() { */

/*     SPACE("#---#"); */

/*     LOG("Testing RandLayer..."); */

/*     int N = 5; */

/*     RandLayer layer = RandLayer(N); */
/*     LOG(layer.str()); */

/*     Eigen::Vector2f x = {0.2, 0.2}; */
/*     Eigen::VectorXf y = layer.call(x); */

/*     // log y */
/*     utils::logging.log_vector(y); */

/*     LOG("layer length: " + std::to_string(layer.len())); */

/*     LOG("matrix:"); */
/*     utils::logging.log_matrix(layer.get_centers()); */

/*     SPACE("#---#"); */
/* } */


/* void test_gridlayer() { */

/*     GridLayer gc(9, 0.1, 0.2); */
/*     /1* printf(gc.str()); *1/ */

/*     utils::logging.log_matrix(gc.get_positions()); */

/*     std::array<float, 2> x = {0.2, 0.2}; */
/*     gc.call(x); */
/* } */


void test_trgp() {

    // make Da
    BaseModulation da = BaseModulation("DA", 5, 0.1, 0.1, 0.0,
                                       0.0, 0.0, 0.0, 0.0);

    // make trg program
    Eigen::MatrixXf wrec = Eigen::MatrixXf::Random(5, 5);
    TargetProgram trgp = TargetProgram(0.0f, wrec, da, 20, 0.8f);

    std::cout << "Created: " << trgp.str() << std::endl;

    // step
    float activation = 0.1f;

    // update
    trgp.update(activation);

    // get trg representation
    Eigen::VectorXf trg = trgp.get_trg_representation();
    utils::logging.log_vector(trg);

}


void test_brain() {

    // make circuit
    BaseModulation da = BaseModulation("DA", 5, 0.1, 0.1, 0.0,
                                       0.0, 0.0, 0.0, 0.0);
    BaseModulation bnd = BaseModulation("BND", 5, 0.1, 0.1, 0.0,
                                        0.0, 0.0, 0.0, 0.0);
    Circuits circuits = Circuits(da, bnd);

    // make space
    std::vector<GridHexLayer> layers;
    layers.push_back(GridHexLayer(0.03, 0.1));
    layers.push_back(GridHexLayer(0.05, 0.09));
    layers.push_back(GridHexLayer(0.04, 0.08));
    layers.push_back(GridHexLayer(0.03, 0.07));
    layers.push_back(GridHexLayer(0.04, 0.06));
    GridHexNetwork xfilter = GridHexNetwork(layers);
    PCNNgridhex space = PCNNgridhex(50, xfilter.len(), 7, 1.5, 0.01,
                                    0.1, 0.8, 0.1,
                                    8, 0.1,
                                    xfilter, "2D");

    // make action space
    ActionSampling2D action_space = ActionSampling2D("Test", 10);

    // make target program
    Eigen::MatrixXf wrec = space.get_wrec();
    TargetProgram trgp = TargetProgram(0.0f, wrec, 
                                       da, 20, 0.8f);

    // make brain
    BrainHex brain = BrainHex(circuits, space, action_space, trgp);

    // call
    std::array<float, 2> v = {0.0f, 0.0f};
    std::array<float, 2> v_arr = brain.call(v, 0.0f, 0.0f, {-1.0f, -1.0f});

    std::cout << "Brain: " << brain.str() << std::endl;
    std::cout << "Action: " << v_arr[0] << ", " << v_arr[1] << std::endl;

    Eigen::VectorXf trg = brain.get_trg_representation();
    utils::logging.log_vector(trg);

    v_arr = brain.call(v, 0.0f, 0.0f, {-1.0f, -1.0f});

    std::cout << "Brain: " << brain.str() << std::endl;
    std::cout << "Action: " << v_arr[0] << ", " << v_arr[1] << std::endl;
    trg = brain.get_trg_representation();
    utils::logging.log_vector(trg);

    int count = space.len();
    std::cout << "Space count: " << count << std::endl;

    // loop
    int duration = 10000;
    std::array<float, 2> pos = {0.0f, 0.0f};
    for (int i = 0; i < duration; i++) {
        v = brain.call(v, 0.0f, 0.0f, {-1.0f, -1.0f});
        pos[0] += v[0];
        pos[1] += v[1];
    }
    std::cout << "count: " << space.len() << std::endl;

}


/* void test_brain_2() { */

/*     // make circuit */
/*     BaseModulation da = BaseModulation("DA", 5, 0.1, 0.1, 0.0, */
/*                                        0.0, 0.0, 0.0, 0.0); */
/*     BaseModulation bnd = BaseModulation("BND", 5, 0.1, 0.1, 0.0, */
/*                                         0.0, 0.0, 0.0, 0.0); */
/*     Circuits circuits = Circuits(da, bnd); */

/*     // make space */
/*     /1* std::vector<GridLayer> layers; *1/ */
/*     /1* layers.push_back(GridLayer(10, 0.03, 0.1)); *1/ */
/*     /1* layers.push_back(GridLayer(10, 0.05, 0.09)); *1/ */
/*     /1* layers.push_back(GridLayer(10, 0.04, 0.08)); *1/ */
/*     /1* layers.push_back(GridLayer(10, 0.03, 0.07)); *1/ */
/*     /1* layers.push_back(GridLayer(10, 0.04, 0.06)); *1/ */
/*     /1* GridNetwork xfilter = GridNetwork(layers); *1/ */
/*     /1* PCNNgrid space = PCNNgrid(50, xfilter.len(), 7, 1.5, 0.01, *1/ */
/*     /1*                           0.1, 0.8, 0.1, *1/ */
/*     /1*                           8, 0.1, *1/ */
/*     /1*                           xfilter, "2D"); *1/ */


/*     std::vector<GridLayerSq> layers; */
/*     layers.push_back(GridLayerSq(0.03, 0.1)); */
/*     layers.push_back(GridLayerSq(0.05, 0.09)); */
/*     layers.push_back(GridLayerSq(0.04, 0.08)); */
/*     layers.push_back(GridLayerSq(0.03, 0.07)); */
/*     layers.push_back(GridLayerSq(0.04, 0.06)); */
/*     GridNetworkSq xfilter = GridNetworkSq(layers); */
/*     PCNNsq space = PCNNsq(50, xfilter.len(), 7, 1.5, 0.01, */
/*                               0.1, 0.8, 0.1, */
/*                               8, 0.1, */
/*                               xfilter, "2D"); */

/*     // make action space */
/*     ActionSampling2D action_space_one = ActionSampling2D("One", 10); */
/*     ActionSampling2D action_space_two = ActionSampling2D("Sec", 10); */
/*     OneLayerNetwork eval_network = OneLayerNetwork({0.1, 0.1, 0.1, 0.1}); */

/*     // make target program */
/*     Eigen::MatrixXf wrec = space.get_wrec(); */
/*     TargetProgram trgp = TargetProgram(0.0f, wrec, */ 
/*                                        da, 20, 0.8f); */

/*     ExperienceModule expmd = ExperienceModule(2.0f, */
/*                                               circuits, trgp, space, */
/*                                               eval_network); */

/*     // make brain */
/*     Brain brain = Brain(circuits, space, trgp, expmd); */

/*     std::cout << "Brain declared..." << "\n"; */

/*     // call */
/*     std::array<float, 2> v = {0.0f, 0.0f}; */
/*     std::array<float, 2> v_arr = brain.call(v, 0.0f, 0.0f, {-1.0f, -1.0f}); */

/*     std::cout << "Brain: " << brain.str() << std::endl; */
/*     std::cout << "Action: " << v_arr[0] << ", " << v_arr[1] << std::endl; */

/*     Eigen::VectorXf trg = brain.get_trg_representation(); */
/*     utils::logging.log_vector(trg); */

/*     v_arr = brain.call(v, 0.0f, 0.0f, {-1.0f, -1.0f}); */

/*     std::cout << "Brain: " << brain.str() << std::endl; */
/*     std::cout << "Action: " << v_arr[0] << ", " << v_arr[1] << std::endl; */
/*     trg = brain.get_trg_representation(); */
/*     utils::logging.log_vector(trg); */

/*     int count = space.len(); */
/*     std::cout << "Space count: " << count << std::endl; */

/*     // loop */
/*     int duration = 10000; */
/*     std::array<float, 2> pos = {0.0f, 0.0f}; */
/*     for (int i = 0; i < duration; i++) { */
/*         v = brain.call(v, 0.0f, 0.0f, pos); */
/*         pos[0] += v[0]; */
/*         pos[1] += v[1]; */
/*         v[0] = utils::random.get_random_float(); */
/*         v[1] = utils::random.get_random_float(); */
/*     } */
/*     std::cout << "count: " << space.len() << std::endl; */

/* } */

};
