#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <array>
#include <ctime>
#include <cmath>
#include <queue>
#include <numeric>
#include <algorithm>
#include <iterator>


/* declare classes */

class Logger;
class RandomGenerator;


/* ========================================== */
/* ================ PRIVATE =================
/* ========================================== */

/* LOGGING */


std::string get_datetime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%X", &tstruct);
    return buf;
}


class Logger {

public:
    Logger() {
        /* std::cout << get_datetime() << " [+] Logger" << std::endl; */
    }

    ~Logger() {
        /* std::cout << get_datetime() << " [-] Logger" << std::endl; */
    }

    void log(const std::string &msg,
             const std::string &src = "MAIN") {
        std::cout << get_datetime() << " | " << src \
            << " | " << msg << std::endl;
    }

    void space(const ::std::string &symbol = ".") {
        std::cout << symbol << std::endl;
    }

    template <std::size_t N>
    void log_arr(const std::array<float, N> &arr,
                  const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | [";
        for (size_t i = 0; i < arr.size(); i++) {
            std::cout << arr[i];
            if (i != arr.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    void log_vector(const Eigen::VectorXf &vec,
                    const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | [";
        for (size_t i = 0; i < vec.size(); i++) {
            std::cout << vec[i];
            if (i != vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    void log_matrix(const Eigen::MatrixXf &mat,
                    const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | " << std::endl;

        for (size_t i = 0; i < mat.rows(); i++) {
            std::cout << "[";
            for (size_t j = 0; j < mat.cols(); j++) {
                std::cout << mat(i, j);
                if (j != mat.cols() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }

    template <std::size_t N>
    void log_arr_bool(const std::array<bool, N> &arr,
                      const std::string &src = "MAIN") {

        std::cout << get_datetime() << " | " << src << " | [";
        for (size_t i = 0; i < arr.size(); i++) {
            std::cout << arr[i];
            if (i != arr.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
};


/* RANDOM GENERATORS */


class RandomGenerator {
public:
    // Constructor with an optional seed for deterministic randomness
    RandomGenerator(int seed = 0) : gen(std::mt19937(seed)) {
        /* std::cout << get_datetime() << " [+] RandomGenerator" << std::endl; */
    }

    ~RandomGenerator() {
        /* std::cout << get_datetime() << " [-] RandomGenerator" << std::endl; */
    }

    // Template function to get a random element from an array
    template <std::size_t N>
    int get_random_element(const std::array<int,
                           N>& container) {
        // Check that the container is not empty
        if (container.empty()) {
            throw std::invalid_argument(
                "Container must not be empty");
        }

        // Create a uniform distribution in the range
        // [0, container.size() - 1]
        std::uniform_int_distribution<> dist(0,
                                container.size() - 1);
        // Use the member random engine
        return container[dist(gen)];
    }

    // @brief function to get a random element from
    // a vector of integers
    int get_random_element_vec(
        const std::vector<int>& container) {
        // Check that the container is not empty
        if (container.empty()) {
            throw std::invalid_argument(
                "Container must not be empty");
        }

        // Create a uniform distribution in the
        // range [0, container.size() - 1]
        std::uniform_int_distribution<> dist(0,
                                    container.size() - 1);
        // Use the random engine
        return container[dist(gen)];
    }

    // Function to get a random float within a specified range
    float get_random_float(float min = 0.0f,
                           float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);  // Use the member random engine
    }

    // random integer in the range [min, max]
    int get_random_int(int min = 0, int max = 10) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    void set_seed(int seed = -1) {
        if (seed == -1) {
            // Seed with current time
            gen.seed(static_cast<unsigned>(std::time(0)));
        } else {
            // Seed with provided seed
            gen.seed(seed);
        }
    }

private:
    std::mt19937 gen;  // Mersenne Twister random engine
};


void log_hello() {
    Logger logging = Logger();
    logging.log("Hello, World!");
}


/* ========================================== */
/* ========================================== */
/* NAMESPACE */

namespace utils {


/* ========================================== */
/* ========== ACTIVATION FUNCTIONS ========== */
/* ========================================== */


// @brief: Generalized sigmoid function
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
    const std::vector<float>& x,
    float offset = 1.0f,
    float gain = 1.0f,
    float clip = 0.0f) {

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

// @brief like above with float
float generalized_sigmoid(const float x,
    float offset = 1.0f,
    float gain = 1.0f,
    float clip = 0.0f) {
    // Offset each element by `offset`, apply the gain,
    // and then compute the sigmoid
    float result = 1.0f / (1.0f + \
        exp(-gain * (x - offset)));

    return (result >= clip) ? result : 0.0f;
}

// @brief Calculate a generalized tanh function with offset, gain, and clipping.
float generalized_tanh(float x, float offset = 0.0f,
                       float gain = 1.0f) {
    // Offset the input, apply gain, and compute tanh
    return std::tanh(gain * (x - offset));
}

// @brief softmax function
Eigen::VectorXf softmax_eigen(const Eigen::VectorXf& x, float beta) {
    // Calculate the exponential of the input vector
    Eigen::VectorXf exp_x = (beta * x.array()).exp();

    // Calculate the sum of the exponentials
    float sum_exp_x = exp_x.sum();

    // Calculate the softmax
    return exp_x / sum_exp_x;
}


/* ========================================== */
/* =========== VECTOR FUNCTIONS ============= */
/* ========================================== */


// @brief: get the index of the maximum element in an array
template <std::size_t N>
int arr_argmax(std::array<float, N> arr) {

    // Get iterator to the maximum element
    auto max_it = std::max_element(arr.begin(), arr.end());

    // Calculate the index
    return std::distance(arr.begin(), max_it);
}

// @brief cosine similarity
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

// @brief cosine similarity between a vector and each
// row in a matrix
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

// @brief cosine similarity for a matrix
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

// @brief: calculate the Gaussian distance between a vector
// and a matrix of vectors
Eigen::VectorXf gaussian_distance(
    const Eigen::VectorXf& x,
    const Eigen::MatrixXf& y,
    const Eigen::VectorXf& sigma,
    const bool assert_dims = true) {

    if (assert_dims) {
        // Validate input dimensions
        assert(x.size() == y.cols() && "Input vector" + \
               f" dimension must match matrix columns");
        assert(sigma.size() == y.cols() && "Sigma vector" + \
               f" dimension must match matrix columns");
    }

    // Calculate the squared Euclidean distance
    Eigen::MatrixXf diff = y.rowwise() - x.transpose();

    // Calculate the squared Euclidean distance
    // and divide by the squared sigma
    Eigen::VectorXf distance =
        diff.array().square().rowwise().sum().array() /
        sigma.array().square();

    return distance;
}

// @brief: calculate the exponential of the
// Gaussian distance
Eigen::VectorXf gaussian_distance_exp(
    const Eigen::VectorXf& x,
    const Eigen::MatrixXf& y,
    const Eigen::VectorXf& sigma,
    const bool assert_dims = true) {

    // Calculate the Gaussian distance
    Eigen::VectorXf distance = gaussian_distance(
                        x, y, sigma, assert_dims);

    // Calculate the exponential of the distance
    return (-distance.array()).exp();
}

// @brief: calculate the maximum cosine similarity in a rows
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

// @brief: linspace [from stackoverflow]
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

// @brief threshold a vector
std::vector<float> threshold_vector(const std::vector<float>& vec,
                                    float threshold,
                                    float min_value = 0.0f) {
    std::vector<float> result;
    for (size_t i = 0; i < vec.size(); i++) {
        result.push_back(vec[i] > threshold ? vec[i] : min_value);
    }
    return result;
}

// @brief: eigen to vector
std::vector<float> eigen_to_vector(const Eigen::VectorXf& vec) {
    std::vector<float> result;
    for (int i = 0; i < vec.size(); i++) {
        result.push_back(vec(i));
    }
    return result;
}

// @brief: vector to eigen
Eigen::VectorXf vector_to_eigen(const std::vector<float>& vec) {
    Eigen::VectorXf result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result(i) = vec[i];
    }
    return result;
}

// @brief: generate a lattice of points
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

/* ========================================== */
/* ============ MATRIX FUNCTIONS ============ */
/* ========================================== */

// @brief calculate the connectivity given a matrix
Eigen::MatrixXf connectivity_matrix(
    const Eigen::MatrixXf& matrix,
    float threshold = 0.5f) {

    // Compute the row to row similarity matrix
    Eigen::MatrixXf similarity_matrix = \
        cosine_similarity_matrix(matrix);

    // Threshold the similarity matrix
    Eigen::MatrixXf connectivity = (similarity_matrix.array() \
        > threshold).cast<float>();

    return connectivity;
}

// @brief: given a connectivity matrix, ensure that each
// row has at most k connections (chosen as the k most)
// similar elements in the row
Eigen::MatrixXf k_highest_neighbors(
    Eigen::MatrixXf& connectivity, int k) {

    // Check that k is within bounds
    if (k < 0 || k >= connectivity.cols()) {
        throw std::out_of_range("k is out of bounds.");
    }

    // Initialize the result matrix with zeros
    Eigen::MatrixXf result = Eigen::MatrixXf::Zero(
        connectivity.rows(), connectivity.cols());

    // Iterate over each row
    for (size_t i = 0; i < connectivity.rows(); i++) {
        // Get the row of the connectivity matrix
        Eigen::VectorXf row = connectivity.row(i);

        // Store indices and values
        std::vector<std::pair<float, int>> value_index_pairs;
        for (int j = 0; j < row.size(); j++) {
            value_index_pairs.emplace_back(row(j), j);
        }

        // Sort pairs based on values in descending order
        std::partial_sort(
            value_index_pairs.begin(),
            value_index_pairs.begin() + k,
            value_index_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        // Set the top-k indices in the result matrix to 1
        for (size_t j = 0; j < k; j++) {
            int idx = value_index_pairs[j].second;
            result(i, idx) = 1;
        }
    }

    return result;
}

// @brief: fill a matrix with random values
void fill_random(Eigen::MatrixXf &M, float sparsity = 0.0) {

    // make random generator
    RandomGenerator random = RandomGenerator();

    for (size_t i = 0; i < M.rows(); i++) {
        for (size_t j = 0; j < M.cols(); j++) {
            if (random.get_random_float() > sparsity) {
                M(i, j) = random.get_random_float();
            }
        }
    }
}

// @brief: make an row-orthonormal matrix
template<size_t num_rows, size_t num_cols>
std::array<std::array<float, num_cols>, num_rows> make_orthonormal_matrix() {

    std::array<std::array<float, num_cols>, num_rows> matrix = {};

    // Dot product function
    auto dot_ = [](const std::array<float, num_cols>& v1,
                   const std::array<float, num_cols>& v2) -> float {
        float result = 0.0f;
        for (size_t j = 0; j < num_cols; ++j) {
            result += v1[j] * v2[j];
        }
        return result;
    };

    // Normalize function
    auto normalize = [&dot_](std::array<float,
                             num_cols>& v) -> void {

        // Calculate L2 norm (Euclidean length)
        float norm = std::sqrt(dot_(v, v));

        // Prevent division by zero
        if (std::abs(norm) > 1e-6) {
            for (size_t i = 0; i < num_cols; i++) {
                v[i] /= norm;
            }
        }
    };

    RandomGenerator random = RandomGenerator();

    // Loop over matrix's rows
    for (size_t i = 0; i < num_rows; i++) {

        // New vector (currently using constant value)
        std::array<float, num_cols> v;
        for (size_t j = 0; j < num_cols; j++) {
            v[j] = random.get_random_float();
        }

        // Assign first vector
        if (i == 0) {
            normalize(v);
            matrix[i] = v;
            continue;
        }

        // Gram-Schmidt orthogonalization
        std::array<float, num_cols> proj_v = {};
        for (size_t k = 0; k < i; k++) {
            std::array<float, num_cols> u = matrix[k];
            float proj = dot_(v, u) / dot_(u, u);

            for (size_t j = 0; j < num_cols; j++) {
                proj_v[j] += proj * u[j];
            }
        }

        // Update v
        for (size_t j = 0; j < num_cols; j++) {
            v[j] -= proj_v[j];
        }

        normalize(v);
        matrix[i] = v;
    }

    return matrix;
}


template <std::size_t N, std::size_t M>
Eigen::MatrixXf array_to_eigen_matrix(const std::array<std::array<float, M>, N>& arr) {
    Eigen::MatrixXf mat(N, M);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            mat(i, j) = arr[i][j];
        }
    }
    return mat;
}

float euclidean_distance(const std::array<float, 2>& v1,
                         const std::array<float, 2>& v2) {
    return std::sqrt(std::pow(v1[0] - v2[0], 2) +
                     std::pow(v1[1] - v2[1], 2));
}


/* ========================================== */
/* ============= MISCELLANEOUS ============== */
/* ========================================== */

// @brief: given a weight matrix and a set of centers,
// calculate the new centers based on the weighted
// average of the inputs
Eigen::MatrixXf calc_centers_from_layer(
    const Eigen::MatrixXf& wff,
    const Eigen::MatrixXf& centers) {

    // Initialize vectors for the centers' x and y coordinates
    Eigen::VectorXf X = centers.col(0);
    Eigen::VectorXf Y = centers.col(1);

    // Calculate the weighted averages along each axis
    Eigen::VectorXf wff_sum = wff.rowwise().sum();
    Eigen::VectorXf x = (wff * X).cwiseQuotient(wff_sum);
    Eigen::VectorXf y = (wff * Y).cwiseQuotient(wff_sum);

    // Replace NaN values with -inf
    for (int i = 0; i < x.size(); ++i) {
        if (std::isnan(x(i))) x(i) = \
            -std::numeric_limits<float>::infinity();
        if (std::isnan(y(i))) y(i) = \
            -std::numeric_limits<float>::infinity();
    }

    // Combine x and y into a matrix with two columns
    Eigen::MatrixXf result(x.size(), 2);
    result.col(0) = x;
    result.col(1) = y;

    return result;
}

// @brief: calculate a position given an activity vector,
// a list of associated centers, and their indexes
Eigen::Vector2f calculate_position(
    const Eigen::VectorXf& a, const Eigen::MatrixXf& centers,
    const Eigen::VectorXf& indexes)
{
    // Handle the case where the sum of activities is zero
    if (a.sum() == 0.0f) {
        return Eigen::Vector2f(-1.0f, -1.0f);
    }

    // Collect selected centers and activities based on indexes
    Eigen::MatrixXf selected_centers(indexes.size(),
                                     centers.cols());
    Eigen::VectorXf selected_activities(indexes.size());

    for (int i = 0; i < indexes.size(); ++i) {
        int idx = static_cast<int>(indexes(i));
        selected_centers.row(i) = centers.row(idx);
        selected_activities(i) = a(idx);
    }

    // Calculate the position as the weighted sum
    // of selected centers
    float activity_sum = selected_activities.sum();
    if (activity_sum == 0.0f) {
        // Avoid divide-by-zero
        return Eigen::Vector2f(-1.0f, -1.0f);
    }

    Eigen::Vector2f position = \
        (selected_centers.transpose() * \
        selected_activities) / activity_sum;
    return position;
}

// @brief: calculate the intersection of two lines
// credits: stackoverflow
int get_segments_intersection(float p0_x, float p0_y,
                              float p1_x, float p1_y,
                              float p2_x, float p2_y,
                              float p3_x, float p3_y,
                              float *i_x, float *i_y) {

    float s02_x, s02_y, s10_x, s10_y, s32_x;
    float s32_y, s_numer, t_numer, denom, t;
    s10_x = p1_x - p0_x;
    s10_y = p1_y - p0_y;
    s32_x = p3_x - p2_x;
    s32_y = p3_y - p2_y;

    denom = s10_x * s32_y - s32_x * s10_y;
    if (denom == 0)
        return 0; // Collinear
    bool denomPositive = denom > 0;

    s02_x = p0_x - p2_x;
    s02_y = p0_y - p2_y;
    s_numer = s10_x * s02_y - s10_y * s02_x;
    if ((s_numer < 0) == denomPositive)
        return 0; // No collision

    t_numer = s32_x * s02_y - s32_y * s02_x;
    if ((t_numer < 0) == denomPositive)
        return 0; // No collision

    if (((s_numer > denom) == denomPositive) || ((t_numer > denom) == denomPositive))
        return 0; // No collision
    //
    // Collision detected
    t = t_numer / denom;
    if (i_x != NULL)
        *i_x = p0_x + (t * s10_x);
    if (i_y != NULL)
        *i_y = p0_y + (t * s10_y);

    return 1;
}

// @brief: reflect a point over a line
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

// @brief: calculate the shortest path between two nodes in a graph
std::vector<int> shortest_path_bfs(const Eigen::MatrixXf connectivity_matrix,
                                   int start_node, int end_node) {

    int num_nodes = connectivity_matrix.rows();
    std::vector<bool> visited(num_nodes, false);
    std::vector<int> parent(num_nodes, -1);
    std::queue<int> queue;

    queue.push(start_node);
    visited[start_node] = true;

    while (!queue.empty()) {
        int current_node = queue.front();
        queue.pop();

        if (current_node == end_node) break;

        for (int neighbor = 0; neighbor < num_nodes; ++neighbor) {
            if (connectivity_matrix(current_node, neighbor) == 1 && !visited[neighbor]) {
                visited[neighbor] = true;
                parent[neighbor] = current_node;
                queue.push(neighbor);
            }
        }
    }

    // Reconstruct the path
    std::vector<int> path;
    int current = end_node;
    while (current != -1) {
        path.push_back(current);
        current = parent[current];
    }
    std::reverse(path.begin(), path.end());

    // Check if path exists
    if (path.empty() || path[0] != start_node) {
        return {};
    }


    return path;
}

/* ========================================== */
/* ================ OBJECTS ================= */
/* ========================================== */


Logger logging = Logger();
RandomGenerator random = RandomGenerator();

/* ========================================== */
/* ================= TESTS ================== */
/* ========================================== */

void test_random_1() {

    RandomGenerator random = RandomGenerator();
    random.set_seed();

    std::vector<int> vec = {10, 1, 2, 3};

    logging.log("size: " + std::to_string(vec.size()));

    int val = random.get_random_element_vec(vec);

    logging.log("Hello, this is: " + std::to_string(val));
}


void test_max_cosine() {
    Eigen::MatrixXf matrix(3, 3);
    matrix << 1, 0, 0,
              1, 0, 1,
              0, 1, 0;

    // print the matrix
    logging.log("Matrix:");
    for (int i = 0; i < matrix.rows(); i++) {
        logging.log_vector(matrix.row(i));
    }

    Eigen::MatrixXf similarity_matrix = cosine_similarity_matrix(matrix);
    float max_sim = max_cosine_similarity_in_rows(matrix, 0);
    logging.log("Max cosine similarity: " + std::to_string(max_sim));

    // Print the similarity matrix
    logging.log("Similarity matrix:");
    for (int i = 0; i < similarity_matrix.rows(); i++) {
        logging.log_vector(similarity_matrix.row(i));
    }
}


void test_connectivity() {

    Eigen::MatrixXf matrix(4, 4);
    matrix << 1, 0, 0, 0,
              1, 0, 1, 0,
              0, 1, 0, 0,
              1, 1, 0, 1;

    Eigen::MatrixXf connectivity = connectivity_matrix(matrix,
                                                       0.01f);

    Eigen::MatrixXf similarity_matrix = \
        cosine_similarity_matrix(matrix);

    // element-wise product with the connectivity matrix
     Eigen::MatrixXf weights = \
        similarity_matrix.cwiseProduct(connectivity);

    logging.log("weights matrix:");
    for (int i = 0; i < weights.rows(); i++) {
        logging.log_vector(weights.row(i));
    };

    Eigen::MatrixXf k_neighbors = k_highest_neighbors(weights,
                                                      2);

    logging.log("k neighbors:");
    for (int i = 0; i < k_neighbors.rows(); i++) {
        logging.log_vector(k_neighbors.row(i));
    }
}


void test_make_position() {

    Eigen::VectorXf a = Eigen::VectorXf::Zero(4);
    a << 0.0, 0.5, 0.0, 1.0;

    Eigen::MatrixXf centers(4, 2);
    centers << 0.0, 0.0,
               1.0, 1.0,
               2.0, 2.0,
               3.0, 3.0;

    Eigen::VectorXf indexes = Eigen::VectorXf::Zero(3);
    indexes << 0, 1, 2;

    Eigen::Vector2f position = calculate_position(a, centers, indexes);

    logging.log("Position: ");
    logging.log_vector(position);
}


template<size_t num_rows, size_t num_cols>
void test_orth_matrix() {

    std::array<std::array<float, num_cols>, num_rows> matrix = make_orthonormal_matrix<num_rows, num_cols>();

    std::cout << "Testing orthonormal matrix" << "\n";

    std::cout << std::to_string(num_rows) << ", ";
    std::cout << std::to_string(num_cols) << "\n";

    for (size_t i = 0; i < num_rows; i++) {
        std::cout << "\n";
        for (size_t j = 0; j < num_cols; j++) {
            std::cout << std::to_string(matrix[i][j]) << " ";
        }
    }
}


}

