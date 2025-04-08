import numpy as np




def calc_density_values(graph: np.ndarray, rw_position: np.ndarray,
                        rw_radius: float, area_radius: float,
                        env_length: float) -> tuple:

    """
    calculation of the average density around the reward location in
    an area of radius #area_radius (but not within the reward radius)
    and outside.

    Parameters
    ----------
    graph: np.ndarray
        node centers, as a (N, 2) array
    rw_position: np.ndarray
        reward position
    rw_radius: float
    area_radius: float
    env_length: float

    Returns
    -------
    tuple: (float, float)
        density near, density far
    """

    rx, ry = rw_position
    num_near = 0
    num_far = 0

    # calculate counts
    for c in graph:
        dist = np.sqrt((c[0]-rx)**2 + (c[1]-ry)**2)
        if dist < area_radius:
            num_near += 1
        else:
            num_far += 1

    # calculate areas
    area_near = 2 * np.pi * (area_radius - rw_radius)
    area_far = env_length ** 2

    # densities
    return num_near / area_near, num_far / area_far

