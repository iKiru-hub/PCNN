import numpy as np
import matplotlib.pyplot as plt
import core.build.pclib as pclib



if __name__ == "__main__":

    gl = pclib.GridLayer(0.2, 0.1, [0., 1., 0., 1.])
    gl = pclib.GridLayer(0.2, 0.1)
    print(gl)

    tot = 200
    X = [[-0.2, 0.4]]*tot
    A = []
    for x in X: A += [gl(x)]

    for a in np.array(A).T: plt.plot(range(tot), a)
    plt.show()

    print(gl.get_positions())
