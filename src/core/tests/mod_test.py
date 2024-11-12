import pytest
import numpy as np

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mod_core as modc
import pcnn_core as pcnnc

@pytest.fixture
def target_module():

    """Fixture to initialize TargetModule instance."""

    params = {
        "N": N,
        "Nj": Nj,
        "alpha": 0.17,
        "beta": 35.0,
        "clip_min": 0.005,
        "threshold": 0.3,
        "rep_threshold": 0.8,
        "rec_threshold": 0.7,
        "calc_recurrent_enable": True,
        "k_neighbors": 7,
        "name": "PCNN"
    }

    # pc filter
    pclayer = pcnn.PClayer(n=13, sigma=0.01)
    logger.debug(f"{pclayer=}")
    params["xfilter"] = pclayer

    # pcnn
    model = pcnn.PCNN(**params)
    trg_module = mod.TargetModule(pcnn=model,
                                  pcnn_plotter=model_plotter,
                                  circuits=circuits,
                                  speed=SPEED,
                                  score_weight=10.,
                                  visualize=True,
                                  visualize_action=False,
                                  number=1)

    return modc.TargetModule()

def test_target_module_output_shape(target_module):
    """Test that the output shape is as expected."""
    # Define input array of shape (N, 1)
    N = 5
    input_array = np.random.rand(N, 1)

    # Call the target module and check the output shape
    output_array = target_module(input_array)
    M, output_dim = output_array.shape

    # Assert the output shape is (M, 1)
    assert output_dim == 1, f"Expected output dimension 1, got {output_dim}"
    assert M > 0, f"Expected M > 0, got {M}"

def test_target_module_value_properties(target_module):
    """Test that output values satisfy expected properties."""
    # Define input array of shape (N, 1)
    N = 5
    input_array = np.random.rand(N, 1)

    # Call the target module
    output_array = target_module(input_array)

    # Example check: assert that the output is not empty
    assert output_array.size > 0, "Output array is empty"

    # Add other checks based on expected properties of your function
    # For example, if output values should be positive, uncomment the line below:
    # assert np.all(output_array > 0), "Not all output values are positive"

# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main()
