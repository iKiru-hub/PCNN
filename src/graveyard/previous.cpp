

/* ========================================== */
/* ============== INPUT LAYER =============== */
/* ========================================== */


// @brief InputLayer abstract class
class InputLayer {

public:
    InputLayer(int N) {

        // assert N is a perfect square
        assert(std::sqrt(N) == std::floor(std::sqrt(N)));
        this->N = N;

        // Initialize the centers
        basis = Eigen::MatrixXf::Zero(N, 2);
        y = Eigen::VectorXf::Zero(N);
    }

    virtual ~InputLayer() {} //LOG("[-] InputLayer destroyed"); }
    virtual Eigen::VectorXf call(const Eigen::Vector2f& x){};

    std::string str() { return "InputLayer"; }
    std::string repr() { return str() + "(N=" + \
        std::to_string(N) + ")"; }
    Eigen::MatrixXf get_centers() { return basis; };
    Eigen::VectorXf get_activation() { return y; }

    int len() const { return N; }

private:
    virtual void make_tuning() {};

protected:
    int N;
    Eigen::MatrixXf basis;
    Eigen::VectorXf y;
};


class PCLayer : public InputLayer {

public:

    PCLayer(int n, float sigma,
            std::array<float, 4> bounds)
        : InputLayer(n*n), n(n), sigma(sigma),
        bounds(bounds) {

        // Compute the centers
        make_tuning();
        LOG("[+] PCLayer created");
    }

    ~PCLayer(){ LOG("[-] PCLayer destroyed"); }

    // @brief Call the PCLayer with a 2D input and compute
    // the Gaussian distance to the centers
    Eigen::VectorXf call(const Eigen::Vector2f& x) {
        for (int i = 0; i < N; i++) {
            float dx = x(0) - basis(i, 0);
            float dy = x(1) - basis(i, 1);
            float dist_squared = std::pow(dx, 2) + \
                std::pow(dy, 2);
            y(i) = std::exp(-dist_squared / denom);
        }
        return y;
    }

    std::string str() { return "PCLayer"; }

private:

    int n;
    float sigma;
    const float denom = 2 * sigma * sigma;
    std::array<float, 4> bounds;

    void make_tuning() {

        // calculate the spacing between the centers
        // given the number of centers and the bounds
        float x_spacing = (bounds[1] - bounds[0]) / (n - 1);
        float y_spacing = (bounds[3] - bounds[2]) / (n - 1);

        // Compute the centers
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                basis(i * n + j, 0) = bounds[0] + \
                    i * x_spacing;
                basis(i * n + j, 1) = bounds[2] + \
                    j * y_spacing;
            }
        }
    }

};


class RandLayer : public InputLayer {

public:

    RandLayer(int N): InputLayer(N) {

        // make matrix
        make_tuning();
        LOG("[+] RandLayer created");
    }

    ~RandLayer() { LOG("[-] PCLayer destroyed"); }

    // @brief Call the RandLayer with a 2D input and
    // apply a the layer's linear projection
    Eigen::VectorXf call(const Eigen::Vector2f& x) {
        for (int i = 0; i < N; i++) {
            y(i) = basis.row(i).dot(x);
        }
        return y;
    }

    std::string str() { return "RandLayer"; }

private:

    // @brief calculate the basis of a linear projection
    // 2D -> nD as set of orthogonal vectors through
    // the Gram-Schmidt process
    void make_tuning() {

        /* size_t n = N; */
        /* std::array<std::array<float, n>, 2> matrix = utils::make_orthonormal_matrix<2, n>(); */

        // define two random vectors of length N
        // in the range [0, 1]
        Eigen::VectorXf v1 = (Eigen::VectorXf::Random(N) + \
            Eigen::VectorXf::Ones(N)) / 2;
        Eigen::VectorXf v2 = (Eigen::VectorXf::Random(N) + \
            Eigen::VectorXf::Ones(N)) / 2;

        // Compute dot products
        float multiplier = v1.dot(v2) / v1.dot(v1);
        Eigen::VectorXf v2_orth = v2 - multiplier * v1;

        // compute the sum of v1
        float sum = v1.sum();
        float sum_orth = v2_orth.sum();

        // normalize the vectors
        v1 = v1 / sum;
        v2_orth = v2_orth / sum_orth;

        // define final matrix
        basis.col(0) = v1;
        basis.col(1) = v2_orth;
    }

};


enum BoundaryType {
    square,
    hexagon,
    circle,
    klein
};

class GridLayer : public InputLayer {

public:

    GridLayer(int N, float sigma, float speed,
              std::array<float, 4> init_bounds = {-1, 1, -1, 1},
              std::string boundary_type = "square",
              std::string basis_type = "square"):
        InputLayer(N), sigma(sigma), speed(speed),
        boundary_type(boundary_type),
        basis_type(basis_type),
        init_bounds(init_bounds){

        this->basis = Eigen::MatrixXf::Zero(N, 2);
        this->positions = Eigen::MatrixXf::Zero(N, 2);

        // record positions
        if (basis_type == "square") {
            square_basis();
        } else if (basis_type == "random_square") {
            random_square_basis();
        } else if (basis_type == "shifted_square") {
            shifted_square_basis();
        } else if (basis_type == "random_circle") {
            random_circle_basis();
        } else {
            throw std::invalid_argument(
                "CUSTOM ERROR: unknown basis type" );
        }

        // record initial positions in the basis
        // pass by value
        positions = basis;

        // record boundary type
        if (boundary_type == "square") {
            boundary_type_num = BoundaryType::square;
        } else if (boundary_type == "hexagon") {
            boundary_type_num = BoundaryType::hexagon;
        } else if (boundary_type == "circle") {
            boundary_type_num = BoundaryType::circle;
        } else if (boundary_type == "klein") {
            boundary_type_num = BoundaryType::klein;
        } else {
            throw std::invalid_argument(
                "CUSTOM ERROR: unknown boundary type" );
        }

        DEBUG("Boundary type: " + boundary_type);
        DEBUG("Basis type: " + basis_type);

        // make matrix
        /* LOG("[+] GridLayer created"); */
    }

    ~GridLayer() {} // LOG("[-] GridLayer destroyed"); }

    // @brief call the GridLayer with a 2D input
    Eigen::VectorXf call(const std::array<float, 2>& v) {

        // update position with velociy
        for (int i = 0; i < N; i++) {
            positions(i, 0) += speed * v[0];
            positions(i, 1) += speed * v[1];
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
        /* std::array<std::array<float, 2>, 25> new_positions; */
        Eigen::MatrixXf new_positions;
        /* new_positions.col(0) = positions.col(0) + speed * \ */
            /* v[0]; */
        /* new_positions.col(1) = positions.col(1) + speed * \ */
            /* v[1]; */
        for (int i = 0; i < N; i++) {
            new_positions(i, 0) = positions(i, 0) + \
                speed * (v[0] - positions(i, 0));
            new_positions(i, 1) = positions(i, 1) + \
                speed * (v[1] - positions(i, 1));
        }

        // check boundary conditions
        /* for (int i = 0; i < N; i++) { */
        /*     std::array<float, 2> new_position = hexagon.call( */
        /*         new_positions[i][0], new_positions[i][1]); */

        /*     new_positions[i][0] = new_position[0]; */
        /*     new_positions[i][1] = new_position[1]; */
            /* std::array<float, 2> new_position = hexagon.call( */
            /*     new_positions(i, 0), new_positions(i, 1)); */
            /* new_positions(i, 0) = new_position[0]; */
            /* new_positions(i, 1) = new_position[1]; */
        /* } */

        boundary_conditions(new_positions);

        // compute the activation
        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        float dist_squared;
        for (int i = 0; i < N; i++) {
            /* dist_squared = std::pow(new_positions(i, 0), 2) + \ */
            /*     std::pow(new_positions(i, 1), 2); */
            dist_squared = std::pow(new_positions(i, 0),
                                    2) + \
                std::pow(new_positions(i, 1), 2);
            yfwd(i) = std::exp(-dist_squared / sigma);
        }

        return yfwd;
    }

    std::array<float, 2> calculate_velocity(Eigen::VectorXf& x) {
        Eigen::Vector2f v = Eigen::Vector2f::Zero();
        for (int i = 0; i < N; i++) {
            v(0) += x(i) * positions(i, 0);
            v(1) += x(i) * positions(i, 1);
        }
        return {v(0), v(1)};
    }

    std::string str() const { return "GridLayer"; }
    std::string repr() const { return "GridLayer(" + \
        boundary_type + ", " + basis_type + ")"; }
    Eigen::MatrixXf get_positions() { return positions; }
    void reset(std::array<float, 2> v) {
        this->positions = basis;
        call(v);
    }

private:
    Eigen::MatrixXf positions;
    float sigma;
    float speed;
    int boundary_type_num;
    std::string boundary_type;
    std::string basis_type;
    Hexagon hexagon;
    std::array<float, 4> init_bounds;

    // define basis type
    void square_basis() {

        int n = static_cast<int>(std::sqrt(N));
        if (n*n != N) {
            LOG("WARNING: downsizing to " + \
                std::to_string(n*n));
            this-> N = n*n;
        }

        float dx = 1.0f / (static_cast<float>(n) + 0.0f);

        // define the centers as a grid excluding
        /* Eigen::VectorXf y = utils::linspace( */
        /*                 0.0f, 1.0f-dx, n); */
        /* Eigen::VectorXf x = utils::linspace( */
        /*                 0.0f, 1.0f-dx, n); */
        Eigen::VectorXf y = utils::linspace(
                        init_bounds[0],
                        init_bounds[1]-dx, n);
        Eigen::VectorXf x = utils::linspace(
                        init_bounds[2],
                        init_bounds[3]-dx, n);

        for (std::size_t i=0; i<N; i++) {
            float xi = x(i / n);
            float yi = y(i % n);
            basis(i, 0) = xi;
            basis(i, 1) = yi;
        }
    }

    void random_square_basis() {

        // sample random points within the unit circle
        for (int i = 0; i < N; i++) {
            float x = utils::random.get_random_float(0, 1);
            float y = utils::random.get_random_float(0, 1);
            basis(i, 0) = x;
            basis(i, 1) = y;
        }
    }

    void shifted_square_basis() {

        int n = static_cast<int>(std::sqrt(N));
        if (n * n != N) {
            LOG("WARNING: downsizing to " + \
                std::to_string(n * n));
            this->N = n * n;
        }
        // Side length of hexagon
        float s = 1.0f / static_cast<float>(n);
        // Horizontal spacing
        float dx = std::sqrt(3.0f) / 2.0f * s;

        // Vertical spacing
        float dy = 1.5f * s;

        // Grid positions
        Eigen::MatrixXf basis(N, 2);

        // Iterate over rows
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int idx = i * n + j;
                basis(idx, 0) = j * dx;
                basis(idx, 1) = i * dy + (j % 2) * (s / 2);
            }
        }

        // Normalize positions to fit within [0, 1] range
        basis.col(0).array() /= (n * dx);
        basis.col(1).array() /= (n * dy);

        this->basis = basis;
    }

    void random_circle_basis() {

        // sample random points within the unit circle
        for (int i = 0; i < N; i++) {
            float theta = utils::random.get_random_float(
                                        0.0f, 6.28f);
            float radius = utils::random.get_random_float(0, 1);
            basis(i, 0) = radius * std::cos(theta);
            basis(i, 1) = radius * std::sin(theta);
        }
    }

    // define boundary type
    void boundary_conditions(Eigen::MatrixXf& pos) {
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = \
                apply_boundary(pos(i, 0), pos(i, 1));
            pos(i, 0) = new_position[0];
            pos(i, 1) = new_position[1];
        }
    }
    // define boundary type
    void boundary_conditions() {
        for (int i = 0; i < N; i++) {
            std::array<float, 2> new_position = apply_boundary(
                            positions(i, 0),
                            positions(i, 1));
            positions(i, 0) = new_position[0];
            positions(i, 1) = new_position[1];
        }
    }

    // gaussian distance of each position to the centers
    void calc_activation() {
        float dist_squared;
        for (int i = 0; i < N; i++) {
            dist_squared = std::pow(positions(i, 0), 2) + \
                std::pow(positions(i, 1), 2);
            y(i) = std::exp(-dist_squared / sigma);
        }
    }

    void calc_inverse_activation(Eigen::VectorXf& x) {

        Eigen::VectorXf z = Eigen::VectorXf::Zero(N);
        float dist_squared;
        for (int i = 0; i < N; i++) {
            dist_squared = std::log(x(i)) * -sigma;
        }

    }

    // apply boundary conditions
    std::array<float, 2> apply_boundary(float x, float y) {

        switch (boundary_type_num) {
            case BoundaryType::square:
                if (x < -1.0) { x += 2.0; }
                else if (x > 1.0) { x -= 2.0; }
                if (y < -1.0) { y += 2.0; }
                else if (y > 1.0) { y -= 2.0; }
                break;
            case BoundaryType::klein:
                if (x < -1.0) { x += 2.0; y = 2.0 - y; }
                else if (x > 1.0) { x -= 2.0; y = 2.0 - y; }
                if (y < -1.0) { y += 2.0; x = 2.0 - x;}
                else if (y > 1.0) { y -= 2.0; x = 2.0 - x;}
                break;
            case BoundaryType::circle:
                utils::logging.log("not implemented yet");
                break;
            case BoundaryType::hexagon:
                std::array<float, 2> P = hexagon.call(x, y);
                x = P[0];
                y = P[1];

        /* positions(i, 0) = x; */
        /* positions(i, 1) = y; */
        }

        return {x, y};
    };

};


class GridNetwork {

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

        LOG("[+] GridNetwork created");
    }

    ~GridNetwork() { LOG("[-] GridNetwork destroyed"); }

    Eigen::VectorXf call(const std::array<float, 2>& x) {

        for (int i = 0; i < num_layers; i++) {
            y.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].call(x);
        }

        return y;
    }

    Eigen::VectorXf fwd_position(
        const std::array<float, 2>& v) {

        Eigen::VectorXf yfwd = Eigen::VectorXf::Zero(N);
        for (int i = 0; i < num_layers; i++) {
            yfwd.segment(i*layers[i].len(), layers[i].len()) = \
                layers[i].fwd_position(v);
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
            basis.block(i*layers[i].len(),
                        0, layers[i].len(), 2) = \
                layers[i].get_positions();
        }
        return basis;
    }
    Eigen::MatrixXf get_positions() {
        for (int i = 0; i < num_layers; i++) {
            basis.block(i*layers[i].len(),
                        0, layers[i].len(), 2) = \
                layers[i].get_positions();
        }
        return basis;
    }

    void reset(std::array<float, 2> v) {
        for (int i = 0; i < num_layers; i++) {
            layers[i].reset(v);
        }
    }

private:
    std::vector<GridLayer> layers;
    int N;
    int num_layers;
    std::string full_repr;
    Eigen::VectorXf y;
    Eigen::MatrixXf basis;
};




/* ========================================== */
/* ================= PCNN =================== */
/* ========================================== */



class PCNNsq {
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

   /* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { */

   /*      // pass the input through the filter layer */
   /*      x_filtered = xfilter.fwd_position(x); */

   /*      // forward it to the network by doing a dot product */
   /*      // with the feedforward weights */
   /*      u = utils::cosine_similarity_vector_matrix( */
   /*              x_filtered, Wff); */

   /*      Eigen::VectorXf sigma = \ */
   /*          Eigen::VectorXf::Constant(u.size(), 0.01); */

   /*      // maybe use cosine similarity? */
   /*      return utils::generalized_sigmoid_vec(u, offset, */
   /*                                         gain, clip_min); */
   /*  } */

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
    std::string str() const { return "PCNNsq." + name; }
    std::string repr() const {
        return "PCNNsq(" + name + std::to_string(N) + \
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
};



class PCNNgrid {
public:
    PCNNgrid(int N, int Nj, float gain, float offset,
         float clip_min, float threshold,
         float rep_threshold,
         float rec_threshold,
         int num_neighbors, float trace_tau,
         GridNetwork xfilter, std::string name = "2D")
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

        ach = 1.0;

        // make vector of free indexes
        for (int i = 0; i < N; i++) {
            free_indexes.push_back(i);
        }
        fixed_indexes = {};
    }

    ~PCNNgrid() { LOG("[-] PCNN destroyed"); }

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
    std::string str() const { return "PCNNgrid." + name; }
    std::string repr() const {
        return "PCNNgrid(" + name + std::to_string(N) + \
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

    GridNetwork xfilter;

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
        u = cosine_similarity_vector_matrix(
                x_filtered, Wff);

        Eigen::VectorXf sigma = \
            Eigen::VectorXf::Constant(u.size(), 0.01);

        // maybe use cosine similarity?
        /* u = gaussian_distance(x_filtered, Wff, sigma); */

        u = generalized_sigmoid_vec(u, offset,
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
        if (check_fixed_indexes() != -1) { return void(); }

        // exit: there are no free neurons
        if (free_indexes.size() == 0) { return void(); }

        // pick new index
        int idx = random_int(0, free_indexes.size() - 1, SEED);

        // determine weight update
        Eigen::VectorXf dw = x_filtered - \
            Wff.row(idx).transpose();

        // trim the weight update
        delta_wff = dw.norm();

        if (delta_wff > 0.0) {

            // update weights
            Wff.row(idx) += dw.transpose();

            // calculate the similarity among the rows
            float similarity = \
                max_cosine_similarity_in_rows(
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
   /* Eigen::VectorXf fwd_ext(const std::array<float, 2>& x) { */

   /*      // pass the input through the filter layer */
   /*      x_filtered = xfilter.fwd_position(x); */

   /*      // forward it to the network by doing a dot product */
   /*      // with the feedforward weights */
   /*      u = cosine_similarity_vector_matrix( */
   /*              x_filtered, Wff); */

   /*      Eigen::VectorXf sigma = \ */
   /*          Eigen::VectorXf::Constant(u.size(), 0.01); */

   /*      // maybe use cosine similarity? */
   /*      return generalized_sigmoid_vec(u, offset, */
   /*                                         gain, clip_min); */
   /*  } */

   /* Eigen::VectorXf fwd_int(const Eigen::VectorXf& a) { */
   /*      return Wrec * a + pre_x; */
   /*  } */

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
        else {return max_idx; }
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
        connectivity = connectivity_matrix(
            Wff, rec_threshold
        );

        // similarity
        Wrec = cosine_similarity_matrix(Wff);

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

        this->fixed_centers = generate_lattice(N, length);

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

        y = generalized_sigmoid_vec(y, offset,
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

        return generalized_sigmoid_vec(z, offset,
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
        else { return max_idx; }
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
        connectivity = connectivity_matrix(
            Wff, rec_threshold
        );

        // similarity
        Wrec = cosine_similarity_matrix(Wff);

        // weights
        Wrec = Wrec.cwiseProduct(connectivity);
    }

};




/* ========================================== */
/* =========== MODULATION MODULES =========== */
/* ========================================== */



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




// target program
class TargetProgramV0 {

    float threshold1;
    float threshold2;
    bool active;
    int max_depth;
    int size;
    Eigen::VectorXf trg_representation;
    Eigen::VectorXf trg_gc_representation;
    BaseModulation& modulator;
    Eigen::MatrixXf wrec;
    Eigen::MatrixXf wff;

    std::tuple<Eigen::VectorXf, bool> converge_to_location(
        Eigen::VectorXf representation,
        int depth, Eigen::VectorXf& modulation) {

        // recurrent step in the space
        /* Eigen::VectorXf u = space.fwd_int(representation); */
        Eigen::VectorXf u = wrec * (representation + modulation);

        float similarity = utils::cosine_similarity_vec(
            representation, u);

        /* std::cout << "similarity: " << similarity << std::endl; */

        if (similarity > threshold2) {
            return std::make_tuple(u, true);
        } else if (depth > max_depth) {
            return std::make_tuple(u, false);
        }

        return converge_to_location(u, depth+1, modulation);
    }

public:

    TargetProgramV0(float threshold1,
                  Eigen::MatrixXf wrec,
                  Eigen::MatrixXf wff,
                  BaseModulation& modulator,
                  int max_depth = 20,
                  float threshold2=0.8):
        threshold1(threshold1), threshold2(threshold2),
        wrec(wrec), wff(wff), modulator(modulator),
        active(false), size(wrec.rows()), max_depth(max_depth) {

        size = wrec.rows();
        trg_representation = Eigen::VectorXf::Zero(size);
        trg_gc_representation = Eigen::VectorXf::Zero(size);

        LOG("[+] TargetProgramV0 created");
    }

    ~TargetProgramV0() { LOG("[-] TargetProgramV0 destroyed"); }

    // attempt to define a target representation
    void update(float activation) {

        /*
        trg_repr <- converge(modulation)
        enabled
        --
        value <- compare velocities()
        */

        /* active = activation > threshold1; */
        active = true;
        Eigen::VectorXf modulation = modulator.get_weights();

        // print modulation
        if (active) {
            trg_representation = \
                Eigen::VectorXf::Zero(size);
            /* auto [trg_representation, active] = \ */
            /*     converge_to_location(representation, */
            /*                          0, modulation); */
            /* trg_representation = representation; */
            /* active = false; */

            // get the index of the maximum value
            Eigen::Index maxIndex;
            float maxValue = modulation.maxCoeff(&maxIndex);

            // set the target representation index
            // to the maximum value
            trg_representation(maxIndex) = maxValue;

            trg_gc_representation = wff.transpose() * trg_representation;

            // forward it through wrec
            /* trg_representation = modulation * trg_representation; */
            /* trg_representation = utils::softmax_eigen( */
            /*     trg_representation, 20.0f); */
            /* trg_representation = wrec * trg_representation; */
            /* trg_representation = utils::softmax_eigen( */
            /*     trg_representation, 20.0f); */
        };
    }

    // quantify how similar a given representation is to the
    // target representation
    float evaluate(Eigen::VectorXf& next_representation) {

        if (!active) { return 0.0f; };

        Eigen::VectorXf next_gc_representation = \
            wff.transpose() * next_representation;

        /* LOG("next gc repr"); */
        /* for (int i = 0; i < next_gc_representation.size(); i++) { */
        /*     std::cout << next_gc_representation(i) << " "; */
        /* } */

        /* LOG("trg gc repr"); */
        /* for (int i = 0; i < trg_gc_representation.size(); i++) { */
        /*     std::cout << trg_gc_representation(i) << " "; */
        /* } */

        float score = trg_gc_representation.dot(next_gc_representation);

        return score;
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
        space.update();

        // update: circuits 
        std::array<float, 3> state_int = \
            circuits.call(u, collision, reward);

        // update: trgp
        trgp.set_wrec(space.get_wrec());
        /* trgp.update(state_int[1]); */

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




struct OneLayerNetwork {

    // circuit size + target program
    const std::array<float, CIRCUIT_SIZE> weights;
    float output;
    int size;


    // @brief forward an input through the network
    // @return tuple(float, array<float, 2>)
    std::pair<std::array<float, CIRCUIT_SIZE>, float>
    call(const std::array<float, CIRCUIT_SIZE>& x) {

        output = 0.0f;
        std::array<float, CIRCUIT_SIZE> z = {0.0f};
        for (size_t i = 0; i < size; i++) {
            output += x[i] * weights[i];
            z[i] = x[i] * weights[i];
        }
        return std::make_pair(z, output);
    }

    OneLayerNetwork(std::array<float, CIRCUIT_SIZE> weights)
        : weights(weights) {
        size = weights.size();
    }
    ~OneLayerNetwork() {}
    std::string str() { return "OneLayerNetwork"; }
};


    /* --- !
    std::tuple<std::vector<std::array<float, 2>>,
               std::vector<std::array<float, 2>>,
               std::vector<std::array<float, CIRCUIT_SIZE>>,
               std::vector<float>,
               int> \
        main_rollout_(Eigen::VectorXf curr_representation,
                     std::array<float, 2>& position) {

        action_space_one.reset();

        std::vector<std::array<float, 2>> new_action_seq_f = {{0.0f}};
        std::vector<std::array<float, 2>> position_seq_f = {{0.0f}};
        std::vector<std::array<float, CIRCUIT_SIZE>> values_seq_f = {{0.0f}};
        std::vector<float> scores_seq_f = {0.0f};  // for each action
        int depth_f = 0;
        float best_score = -100000.0f;
        all_values = {0.0f};

        LOG("best score: " + std::to_string(best_score));

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

            std::pair<std::array<float, CIRCUIT_SIZE>, float> \
                evaluation = evaluate_action(curr_representation,
                                             next_representation);

            // update the provisional plan
            std::vector<std::array<float, 2>> new_action_seq;
            std::vector<std::array<float, 2>> position_seq;
            std::vector<std::array<float, CIRCUIT_SIZE>> values_seq;
            std::vector<float> scores_seq;  // for each action
            int depth = 1;
            new_action_seq.push_back(std::get<0>(action_data));
            new_action_seq.push_back(position);
            values_seq.push_back(evaluation.first);
            scores_seq.push_back(evaluation.second);

            curr_representation = next_representation;

            // inner loop
            inner_rollout_(curr_representation,
                          new_action_seq,
                          position_seq,
                          values_seq,
                          scores_seq,
                          depth,
                          position);

            // print all actions

            // check if the average score is the best
            float avg_score = std::accumulate(
                scores_seq.begin(), scores_seq.end(), 0.0) / \
                scores_seq.size();
            all_values[action_space_one.get_idx()] = avg_score;
            if (avg_score > best_score) {
                new_action_seq_f = new_action_seq;
                position_seq_f = position_seq;
                values_seq_f = values_seq;
                scores_seq_f = scores_seq;
                depth_f = depth;
                best_score = avg_score;
            }
        }

        if (depth_f < 1) {
            std::cout << "Error: depth is -1000.0f" << std::endl;
            LOG("Error: depth is -1000.0f");
            LOG("depht: " + std::to_string(depth_f));
            LOG("Press enter to continue");
            int ground;
        } else {
            LOG("Depth: " + std::to_string(depth_f) + \
                ", Best score: " + std::to_string(best_score));
        }

        return {new_action_seq_f, position_seq_f,
            values_seq_f, scores_seq_f, depth_f};
    }

    void inner_rollout_(Eigen::VectorXf curr_representation,
                       std::vector<std::array<float, 2>>& new_action_seq,
                       std::vector<std::array<float, 2>>& position_seq,
                       std::vector<std::array<float, CIRCUIT_SIZE>>& values_seq,
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

            // evaluate
            std::pair<std::array<float, CIRCUIT_SIZE>, float> \
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
            position_seq.push_back(position);
            values_seq.push_back(evaluation.first);
            scores_seq.push_back(evaluation.second);
            curr_representation = next_representation;
            depth++;
        }
    }
    */



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

    std::array<float, 2> call(bool keep = false) {

        // Keep current state
        if (keep) {
            utils::logging.log("-- keep");
            /* return std::make_tuple(velocity, false, idx); */
            return velocity;
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
            /* return std::make_tuple(velocity, true, idx); */
            return velocity;
        }

        // Sample a new index
        idx = sample_idx();
        available_indexes[idx] = false;
        velocity = samples[idx];

        /* return std::make_tuple(velocity, false, idx); */
        return velocity;
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

    std::array<std::array<float, 2>, 8> get_actions() {
        return samples;
    }

private:

    // parameters | trying with only 8 directions (no stop)
    /* std::array<float, 2> samples[8] = { */
    std::array<std::array<float, 2>, 8> samples = {{
        {-0.707f, 0.707f},
        {0.0f, 1.0f},
        {0.707f, 0.707f},
        {-1.0f, 0.0f},
        {1.0f, 0.0f},
        {-0.707f, -0.707f},
        {0.0f, -1.0f},
        {0.707f, -0.707f}
    }};
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



class ActionSampling2Dv2 {

    std::array<std::array<float, 2>, 8> samples = {{
        {-0.707f, 0.707f},
        {0.0f, 1.0f},
        {0.707f, 0.707f},
        {-1.0f, 0.0f},
        {1.0f, 0.0f},
        {-0.707f, -0.707f},
        {0.0f, -1.0f},
        {0.707f, -0.707f}
    }};
    const std::array<unsigned int, 8> indexes = \
            { 0, 1, 2, 3, 4, 5, 6, 7};
    unsigned int counter = 0;
    int idx = -1;
    int num_samples = 8;
    std::array<bool, 8> available_indexes = { true, true, true,
        true, true, true, true, true};

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

public:
    std::string name;

    std::array<float, 2> call() {

        if (counter == num_samples) {
            counter = 0;
            return samples[0];
        }

        // Sample a new index
        idx = indexes[counter];
        available_indexes[idx] = false;
        counter++;
        return samples[idx];
    }

    void reset() {
        idx = -1;
        for (int i = 0; i < num_samples; i++) {
            available_indexes[i] = true;
        }
        counter = 0;
    }

    int get_idx() { return idx; }
    bool is_done() { return counter == num_samples; }
    std::array<std::array<float, 2>, 8> get_actions() { return samples; }

    ActionSampling2D(std::string name,
                     float speed) : name(name)
    { update_actions(speed); }
    ~ActionSampling2D() {}

};



class ActionSampling2Dv0 {

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

    std::array<float, 2> call(bool keep = false) {

        // Keep current state
        if (keep) {
            utils::logging.log("-- keep");
            /* return std::make_tuple(velocity, false, idx); */
            return velocity;
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
            /* return std::make_tuple(velocity, true, idx); */
            return velocity;
        }

        // Sample a new index
        idx = sample_idx();
        available_indexes[idx] = false;
        velocity = samples[idx];

        /* return std::make_tuple(velocity, false, idx); */
        return velocity;
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

    std::array<std::array<float, 2>, 8> get_actions() {
        return samples;
    }

private:

    // parameters | trying with only 8 directions (no stop)
    /* std::array<float, 2> samples[8] = { */
    std::array<std::array<float, 2>, 8> samples = {{
        {-0.707f, 0.707f},
        {0.0f, 1.0f},
        {0.707f, 0.707f},
        {-1.0f, 0.0f},
        {1.0f, 0.0f},
        {-0.707f, -0.707f},
        {0.0f, -1.0f},
        {0.707f, -0.707f}
    }};
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



struct Planv0 {
    std::array<float, 2> action;
    std::vector<std::array<float, 2>> action_seq;
    std::vector<std::array<float, 2>> position_seq;
    std::vector<std::array<float, CIRCUIT_SIZE>> values_seq;
    std::vector<float> scores_seq;
    const int max_depth;
    int depth;
    int action_delay;
    int t;
    int counter;

    bool is_done() {
        return t == depth;
    }
    bool is_last() {
        /* bool value = t >= (depth-1); */
        /* LOG("[+] is last? t: " + std::to_string(t) + ", depth: " + std::to_string(depth) + ", not last=" + std::to_string(!value)); */
        return t > (action_delay * (depth-1));
    }

    // STEP
    std::array<float, 2> step() {

        this->action = action_seq[t / action_delay];
        this->t++;
        /* LOG("[+] t: " + std::to_string(t) + ", depth: " + std::to_string(depth) + ", action=" + std::to_string(action[0]) + ", " + std::to_string(action[1])); */

        return action;
    }

    void renew(std::tuple<std::vector<std::array<float, 2>>,
               std::vector<std::array<float, CIRCUIT_SIZE>>,
               std::vector<float>,
              int> data) {
        this->action_seq = std::get<0>(data);
        this->values_seq = std::get<1>(data);
        this->scores_seq = std::get<2>(data);
        this->depth = std::get<3>(data);
        this->t = 0;

        /* LOG("[+] Plan created"); */
        /* log_plan(); */
        /* log_action_plan(); */
    }

    void set_action_delay(float action_delay) {
        this->action_delay = static_cast<int>(action_delay);
    }

    void make_plan_positions(std::array<float, 2> position) {
        this->position_seq = {position};
        this->scores_seq = {0.0f};
        /* this->position_seq.push_back(position); */
        for (int i = 0; i < depth; i++) {

            std::array<float, 2> action_ = action_seq[i];
            float score_ = scores_seq[i];

            for (int j = 0; j < action_delay; j++) {

                // step
                position[0] += action_[0];
                position[1] += action_[1];
                this->position_seq.push_back(position);
                this->scores_seq.push_back(score_);
            }
        }
    }

    Plan(const int max_depth = 10,
         float action_delay = 1.0f):
        max_depth(max_depth) {
        this->t = 0;
        this->counter = 0;
        this->action_delay = static_cast<int>(action_delay);;
        this->action = {0.0f, 0.0f};
        this->action_seq = {{0.0f}};
        this->values_seq = {{0.0f}};
        this->position_seq = {{0.0f}};
        this->scores_seq = {};
        this->depth = 0;
    }
    ~Plan() {}

    void log_plan() {
        LOG("Plan: ");
        for (int i = 0; i < depth; i++) {
            LOG("action: " + std::to_string(action_seq[i][0]) + ", " + \
                std::to_string(action_seq[i][1]));
            LOG("values: " + std::to_string(values_seq[i][0]) + ", " + \
                std::to_string(values_seq[i][1]) + ", " + \
                std::to_string(values_seq[i][2]));
            LOG("scores: " + std::to_string(scores_seq[i]));

            if (values_seq[i][0] < 0.0f) {
                LOG("[!] <---------------- boundary sensor");
            }
        }
    }

    void log_action_plan() {
        LOG("[Action plan]");
        for (int i = 0; i < depth; i++) {
            LOG("a[" + std::to_string(i) + "] " + std::to_string(action_seq[i][0]) + ", " + \
                std::to_string(action_seq[i][1]));
        }
        LOG("---");
    }
};



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
    Eigen::MatrixXf& connectivity;
    Eigen::MatrixXf& centers;
    PCNN_REF& space;
    PCNN_REF& space_coarse;
    /* Eigen::VectorXf& da_weights; */

    // internal variables
    ConsecutiveLocationsHandler conlochandler;
    EpisodicMemory episodic_memory;
    float speed;
    bool active;
    int tmp_trg_idx;
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

        // calculate the current node
        Eigen::Index maxIndex_curr;
        curr_representation.maxCoeff(&maxIndex_curr);
        curr_idx = static_cast<int>(maxIndex_curr);

        // plan
        std::vector<int> plan_idxs = \
            utils::spatial_shortest_path(connectivity,
                                         space.get_centers(),
                                         space_weights,
                                         curr_idx, tmp_trg_idx);

        float max_curr_value = curr_representation.maxCoeff();

        // check if the plan is valid, ie size > 1
        if (plan_idxs.size() < 3) {
            /* LOG("[-] short plan"); */
            return false;
        } else if (max_curr_value < 0.00001f) {
            /* LOG("[-] low max curr value"); */
            return false;
        }

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

    TargetProgram(PCNN_REF& space, PCNN_REF& space_coarse,
                  float speed, int max_attempts = 2):
        active(false), space(space), space_coarse(space_coarse),
        wrec(space.get_wrec()),
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
                Eigen::VectorXf& curr_representation_coarse,
                Eigen::VectorXf& space_weights,
                Eigen::VectorXf& space_weights_coarse,
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

        // exit: conlochandler
        if (!conlochandler.is_valid()) {
            space.add_blocked_edge(conlochandler.start_point,
                                   conlochandler.end_point);
            conlochandler.reset();

            // check edges
            return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
        }

        // +current position
        this->curr_position = space.get_position();
        int curr_idx = space.calculate_closest_index(curr_position);

        // +distance
        float dist = utils::euclidean_distance(curr_position, next_position);

        // -- same next position
        if (dist > 0.01f && counter > 0) {
            float dx = next_position[0] - curr_position[0];
            float dy = next_position[1] - curr_position[1];

            // determine velocity magnitude
            if (dist < speed) { local_velocity = {dx, dy}; }
            else {
                float norm = sqrt(dx * dx + dy * dy);
                local_velocity = {speed * dx / norm,
                                  speed * dy / norm};
            }
            return std::make_pair(local_velocity, true);
        }

        // -- move to the next position

        // check: end of the plan
        if (counter > (plan_idxs.size()-1)) {
            this->active = false;
            this->counter = 0;
            this->depth = 0;
            conlochandler.reset();
            return std::make_pair(std::array<float, 2>{0.0f, 0.0f}, false);
        }

        // retrieve next position
        this->next_position = {centers(plan_idxs[counter], 0),
                               centers(plan_idxs[counter], 1)};

        // check if it's the last point
        if (plan_idxs[counter] == tmp_trg_idx) {
        } else if (plan_idxs[counter] < -10000 || plan_idxs[counter] > 10000) {
            LOG("[!] !!! possible memory leak?? " + \
                std::to_string(plan_idxs[counter]) + ", counter=" + \
                std::to_string(counter) + " | plan size=" + \
                std::to_string(plan_idxs.size()));
        }

        // distance netween the current and next position
        float dx = next_position[0] - curr_position[0];
        float dy = next_position[1] - curr_position[1];
        dist = utils::euclidean_distance(curr_position, next_position);

        // determine velocity magnitude
        if (dist < speed) { local_velocity = {dx, dy}; }
        else {
            float norm = sqrt(dx * dx + dy * dy);
            local_velocity = {speed * dx / norm,
                              speed * dy / norm};
        }

        // set the new points
        conlochandler.set_points(plan_idxs[counter-1], plan_idxs[counter]);

        counter++;

        // update the current position
        /* this->curr_position = {curr_position[0] + local_velocity[0], */
        /*                        curr_position[1] + local_velocity[1]}; */

        return std::make_pair(local_velocity, true);
    }

    void step_episodic_memory(float reward) { episodic_memory.call(reward); }

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




// ExperienceModule


/*     // !main rollout */
/*     std::tuple<std::vector<std::array<float, 2>>, */
/*                std::vector<std::array<float, CIRCUIT_SIZE>>, */
/*                std::vector<float>, int> \ */
/*         main_rollout(Eigen::VectorXf curr_representation) { */

/*         action_space_one.reset(); */

/*         // initialize the plan */
/*         std::vector<std::array<float, 2>> new_action_seq_f = {{0.0f}}; */
/*         std::vector<std::array<float, CIRCUIT_SIZE>> values_seq_f = \ */
/*                                             {{0.0f}}; */
/*         std::vector<float> scores_seq_f = {0.0f};  // for each action */
/*         int depth_f = 0; */
/*         float best_score = -100000.0f; */
/*         all_values = {0.0f}; */
/*         this->all_values_seq_f = {{{0.0f}}}; */

/*         // initial gc positions */
/*         std::vector<std::array<std::array<float, 2>, GCL_SIZE>> \ */
/*             gc_positions_zero = space.get_gc_positions_vec(); */
/*         std::vector<std::array<std::array<float, 2>, GCL_SIZE>> \ */
/*             gc_positions_next = space.get_gc_positions_vec(); */

/*         // outer loop */
/*         while (!action_space_one.is_done()) { */

/*             // sample the starting action */
/*             std::array<float, 2> new_action = \ */
/*                 action_space_one.call(); */

/*             // step | <pc_representation, gc_positions> */
/*             std::pair<Eigen::VectorXf, std::vector<std::array<std::array<float, 2>, GCL_SIZE>>> \ */
/*                 sim_results = space.simulate(new_action, gc_positions_zero); */

/*             // evaluate: (values, score) */
/*             std::pair<std::array<float, CIRCUIT_SIZE>, float> \ */
/*                 evaluation = evaluate_action(curr_representation, */
/*                                              sim_results.first); */

/*             // make the provisional plan */
/*             std::vector<std::array<float, 2>> new_action_seq; */
/*             std::vector<std::array<float, 2>> position_seq; */
/*             std::vector<std::array<float, CIRCUIT_SIZE>> values_seq; */
/*             std::vector<float> scores_seq;  // for each action */
/*             int depth = 1; */

/*             // memory | sum of previous representations */
/*             Eigen::VectorXf memory_representation = sim_results.first; */

/*             // update the provisional plan */
/*             new_action_seq.push_back({new_action[0] / action_delay_float, */
/*                                       new_action[1] / action_delay_float}); */
/*             /1* position_seq.push_back(position); *1/ */
/*             values_seq.push_back(evaluation.first); */
/*             scores_seq.push_back(evaluation.second); */

/*             // update the current representation with the new one */
/*             curr_representation = sim_results.first; */
/*             gc_positions_next = sim_results.second; */

/*             // inner loop | explore the next steps */
/*             for (int i = 0; i < plan.max_depth; i++) { */
/*                  std::tuple<std::array<float, CIRCUIT_SIZE>, */
/*                  float, std::array<float, 2>, \ */
/*                  Eigen::VectorXf, \ */
/*                  std::vector<std::array<std::array<float, 2>, GCL_SIZE>>> \ */
/*                     inner_data = inner_rollout(curr_representation, */
/*                                                gc_positions_next); */

/*                 curr_representation = std::get<3>(inner_data); */
/*                 gc_positions_next = std::get<4>(inner_data); */

/*                 // record : action, position, values, score */
/*                 values_seq.push_back(std::get<0>(inner_data)); */
/*                 scores_seq.push_back(std::get<1>(inner_data)); */
/*                 /1* new_action_seq.push_back(std::get<2>(inner_data)); *1/ */
/*                 new_action_seq.push_back({ */
/*                     std::get<2>(inner_data)[0] / action_delay_float, */
/*                     std::get<2>(inner_data)[1] / action_delay_float */
/*                 }); */
/*                 depth++; */
/*             } */

/*             // check if the average score is the best */
/*             float avg_score = std::accumulate( */
/*                 scores_seq.begin(), scores_seq.end(), 0.0) / \ */
/*                 scores_seq.size(); */
/*             /1* float avg_score = utils::random.get_random_float(0.0, 1.0); *1/ */
/*             all_values[action_space_one.get_idx()] = avg_score; */

/*             this->all_values_seq_f.push_back(values_seq); */

/*             if (avg_score > best_score) { */
/*                 new_action_seq_f = new_action_seq; */
/*                 /1* position_seq_f = position_seq; *1/ */
/*                 values_seq_f = values_seq; */
/*                 scores_seq_f = scores_seq; */
/*                 depth_f = depth; */
/*                 best_score = avg_score; */
/*             } */
/*         } */

/*         /1* LOG("best_score: " + std::to_string(best_score)); *1/ */

/*         if (depth_f < 1) { */
/*             LOG("Error: depth is -1000.0f"); */
/*             LOG("depth: " + std::to_string(depth_f)); */
/*             LOG("scores: " + std::to_string(best_score)); */

/*             std::cout << "Scores: "; */
/*             for (auto& v : all_values) { */
/*                 std::cout << (std::to_string(v)) << "; "; */
/*             } */
/*         } */

/*         return {new_action_seq_f, values_seq_f, scores_seq_f, depth_f}; */
/*     } */

/*     // !inner rollout | evaluate all action from a given state/representation/position */
/*     std::tuple<std::array<float, CIRCUIT_SIZE>, */
/*     float, std::array<float, 2>, */
/*     Eigen::VectorXf, */
/*     std::vector<std::array<std::array<float, 2>, GCL_SIZE>>> \ */
/*         inner_rollout(Eigen::VectorXf& curr_representation, */
/*                       std::vector<std::array<std::array<float, 2>, GCL_SIZE>> curr_gc_positions) { */

/*         action_space_two.reset(); */

/*         std::array<float, CIRCUIT_SIZE> values = {0.0f}; */
/*         float best_score = -1000.0f; */
/*         std::array<float, 2> proposed_action = {0.0f, 0.0f}; */
/*         std::vector<std::array<std::array<float, 2>, GCL_SIZE>> \ */
/*             gc_positions_next = curr_gc_positions; */
/*         Eigen::VectorXf next_representation = curr_representation; */

/*         // loop */
/*         while (!action_space_two.is_done()) { */

/*             // sample the next action */
/*             std::array<float, 2> new_action = \ */
/*                 action_space_two.call(); */

/*             // simulate a step */
/*             std::pair<Eigen::VectorXf, std::vector<std::array<std::array<float, */
/*                                     2>, GCL_SIZE>>> \ */
/*                 sim_results = space.simulate(new_action, curr_gc_positions); */

/*             // evaluate: (values, score) */
/*             std::pair<std::array<float, CIRCUIT_SIZE>, float> \ */
/*                 evaluation = evaluate_action(curr_representation, */
/*                                              sim_results.first); */

/*             // check 2: boundary sensor too high */
/*             /1* if (evaluation.first[0] < -0.99) { *1/ */
/*             /1*     continue; *1/ */
/*             /1* } *1/ */
/*             /1* curr_representation = next_representation; *1/ */

/*             // compare */
/*             if (evaluation.second > best_score) { */
/*                 values = evaluation.first; */
/*                 best_score = evaluation.second; */
/*                 proposed_action = new_action; */
/*                 next_representation = sim_results.first; */
/*                 gc_positions_next = sim_results.second; */
/*             } */
/*         } */

/*         return std::make_tuple(values, best_score, proposed_action, */
/*                                next_representation, */
/*                                gc_positions_next); */
/*     } */

// parth of trg module
    /* // !unused for now */
    /* std::array<float, 2> converge_to_location( */
    /*     Eigen::VectorXf& representation) { */

    /*     // weights for the centers */
    /*     float cx, cy; */
    /*     float sum = representation.sum(); */
    /*     if (sum == 0.0f) { */
    /*         /1* LOG("[-] sum is zero"); *1/ */
    /*         return {-1000.0f, 0.0f}; */
    /*     } */

    /*     for (int i = 0; i < representation.size(); i++) { */
    /*         cx += representation(i) * centers(i, 0); */
    /*         cy += representation(i) * centers(i, 1); */
    /*     } */

    /*     cx /= sum; */
    /*     cy /= sum; */
    /*     /1* LOG("[+] cx: " + std::to_string(cx) + ", cy: " + std::to_string(cy)); *1/ */

    /*     return {cx, cy}; */
    /* } */



/* IN EXPLORATION MODULE */

    /* int make_plan(int rejected_idx) { */

    /*     // check: the current position is at an open boundary */
    /*     int curr_idx = space.calculate_closest_index(space.get_position()); */
    /*     float value = open_boundary_value(curr_idx, circuits.get_bnd_weights(), */
    /*                                       space.get_node_degrees()); */

    /*     // [+] new random walk plan at an open boundary */
    /*     if (value < open_threshold || rejected_idx == 404 || \ */
    /*         edge_route_time < edge_route_interval) { return -1; } */

    /*     // check: there are points at the open boundary */
    /*     int open_boundary_idx = get_open_boundary_idx(rejected_idx); */

    /*     // [+] new random walk plan at an open boundary */
    /*     if (open_boundary_idx < 1) { */
    /*         return -1; } */

    /*     // [+] new trg plan to reach the open boundary */
    /*     LOG("[Exp] new trg plan to reach the open boundary"); */
    /*     return open_boundary_idx; */
    /* } */

    /* int get_open_boundary_idx(int rejected_idx) { */

    /*     Eigen::VectorXf& bnd_weights = circuits.get_bnd_weights(); */
    /*     Eigen::VectorXf& node_degrees = space.get_node_degrees(); */

    /*     if (rejected_idx > -1) { rejected_indexes(rejected_idx) = 1.0f; } */

    /*     // check each neuron */
    /*     int idx = -1; */
    /*     float min_value = 1000.0f; */

    /*     for (int i = 1; i < space.get_size(); i++) { */
    /*         float value = open_boundary_value( */
    /*                 i, bnd_weights, space.get_node_degrees()); */
    /*         if (value < min_value && value > 0) { */
    /*             idx = i; */
    /*             min_value = value; */
    /*         } */
    /*     } */
    /*     this->edge_idx = idx; */
    /*     return idx; */
    /* } */

    /* float open_boundary_value(int idx, Eigen::VectorXf& bnd_weights, */
    /*                           Eigen::VectorXf& node_degrees) { */

    /*     if (bnd_weights(idx) > 0.0f) { return 10000.0f; } */
    /*     return node_degrees(idx); */
    /* } */
