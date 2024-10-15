import inputools.Trajectory as it
import matplotlib.pyplot as plt
# import src.simplerl.pcnn_wrapper as pw
import src.spatial_cells as sc
import numpy as np

import numpy as np
import math



gc = sc.GridCellLayer(nx=4, ny=4, sigma=0.05)
gc2 = sc.GridCellLayer(nx=3, ny=3, sigma=0.1)
pc = sc.PlaceCellLayer(nx=4, ny=4, sigma=0.1)

print(gc)
print(gc2)
print(pc)

group = sc.LayerGroup(layers=[gc, gc2, pc])

print(group)

a = group(np.array([0.5, 0.5]))
print(a.shape)

group.plot()
