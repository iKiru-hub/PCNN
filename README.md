# PCNN
Acronym for Place Cells Neural Network.

### Gist
*an agent should have a machinery able to form a spatial representation of the input with the following features:*
- place cells encoding for (unique) *positions* in the input space 
- place cells should form quickly and *online*
*How?*
- rate neurons
- hebbian plasticity 
- background stimulation from internal theta oscillations 
- neuromodulation-dependant learning (dopamine)

The results below are obtained from a simulation where the agent walked a trajectory spanning most of the environment.
![Formed place fields on the path](media/pcnn_plot_b.png)

![Trajectory](media/pcnn_plot_d.png)

![roaming](media/goods/roaming_222509.gif)

![navigation](media/nav_recording.mov)

### Branches 
- **main** 
- **cold** study of how to implement learning of non-hot encoded representations. _NB: only visible in the original repository for now_

### Forks 
[BioAI_Oslo]
synched every time there is a push to the original repository.

### TODO
- [ ] more tailored unittests
- [ ] make explicit the PC fields online skewing
- [ ] design and implement a RL grid-world + task, see https://minigrid.farama.org/index.html & https://github.com/Farama-Foundation/Miniworld
- [ ] code up the RL agent


