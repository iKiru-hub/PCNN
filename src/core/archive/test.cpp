/* #include "../include/utils.hpp" */
#include "../include/pcnn.hpp"
#include <iostream>

#define LOG(msg) utils::logging.log(msg, "TEST")
#define SPACE utils::logging.space



//


// MAIN

int main() {

    /* testSampling(); */
    /* testLeaky(); */

    /* from utils.hpp
    /* utils::test_random_1(); */
    /* /1* utils::test_max_cosine(); *1/ */
    /* utils::test_connectivity(); */
    /* utils::test_make_position(); */
    /* utils::test_orth_matrix<2, 5>(); */
    /* float out = utils::generalized_tanh(-1); */
    /* LOG(std::to_string(out)); */

    /* /1* from pcnn.hpp */
    /* pcl::testSampling(); */

    pcl::simple_env(100000, 1000, -1.0f);
    /* pcl::test_env(); */
    /* pcl::test_bnd(2000); */


    return 0;
}


// ====================================================================== //

