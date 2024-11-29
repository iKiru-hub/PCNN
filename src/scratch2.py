import libs.pclib as pclib
import numpy as np
import matplotlib.pyplot as plt



N = 80
Nj = 12**2
sigma = 2
BOUNDS = (0, 100, 0, 100)

xfilter = pclib.PCLayer(int(np.sqrt(Nj)), sigma, BOUNDS)

# definition
pcnn2D = pclib.PCNN(N=N, Nj=Nj, gain=3., offset=1.,
                    clip_min=0.09,
                    threshold=0.04,
                    rep_threshold=0.3,
                    rec_threshold=0.01,
                    num_neighbors=8, trace_tau=0.1,
                    xfilter=xfilter,
                    name="2D")


print(dir(pcnn2D))

print(xfilter.get_centers())

