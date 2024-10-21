import numpy as np




def _drift_policy(self, beta: float,
                  W: np.ndarray, representation: np_ndarray) -> tuple:

    """ calculate the drift representation """

    # --- proximal positions
    drift_representation = W @ representation.reshape(-1, 1) - \
         1. * representation.reshape(-1, 1)
    drift_representation = np.maximum(drift_representation, 0)

    # --- exit 1
    if drift_representation.max() == 0.:
        return None, None, None

    # --- only neurons within the fov
    drift_representation = self._trim_active_neurons(
                                representation=drift_representation)

    # % modulate 1
    # drift_representation = self._max_var_drift(
    #                                 a=drift_representation)

    # % modulate 2
    drift_representation = self._variable_drift(a=drift_representation,
                                                threshold=0.001,
                                                beta=beta_drift)
    drift_position = self._pcnn._calc_average_pos(
                                a=drift_representation.copy())
    drift_velocity = drift_position - self.position
    drift_velocity /= drift_velocity.sum()




