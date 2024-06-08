## Spiking network model of CA3 replay

This repository contains a spiking network model of a
simplified microcircuit of CA3, a hippocampal brain region associated
with memory formation. Results obtained with this model are reported in the paper:

> Inhibitory plasticity supports replay generalization in the
> hippocampus.  Zhenrui Liao, Satoshi Terada, Ivan Georgiev Raikov,
> Darian Hadjiabadi, Ivan Soltesz, Attila Losonczy. Nat Neurosci 2024.

During online training, the pyramidal cells in the model receive
spatially-structured place and cue input, which entrains spatial receptive
fields on a virtual linear track. A fraction of pyramidal cells receive sensory 
cue input at a randomly selected location on each lap. During the simulated offline
period, the pyramidal cells in the model receive random spiking input and undergo 
spontaneous sequential reactivations, consistent with spontaneous memory replay. 
During these replay-like events, cue cells are suppressed while place cells are activated. 
This model shows that inhibitory plasticity is sufficient for cue cell
suppression during replay events, and suggests a possible mechanism
for cognitive map formation that is robust to distractor sensory inputs.



## Prerequisites

1) **Numpy** 

The standard python module for matrix and vector computations: https://pypi.python.org/pypi/numpy.

2) **Scipy** 

The standard python module for statistical analysis: http://www.scipy.org/install.html.

3) **Matplotlib**

The standard python module for data visualization: http://matplotlib.org/users/installing.html.

4) **BRIAN2**

A simulator for biophysical models of neurons and networks of neurons: https://github.com/brian-team/brian2

5) **Datajoint**

## Running Simulations

## Clearing existing results
Clear results from previous runs as follows
  python run_simulation.py -c

## Simplified combined run
Run the full simulation as follows:
  python run_simulation.py 
Both the online phase and offline phase will be run. However, if the simulation is interrupted, rerunning the command will resume where the previous run left off

### Training (online) phase

Code related to the online (running) phase lives in stdp.py


### Offline phase
Code related to the offline (stillness) phase lives in spw_network.py

## Model configurations/parameters associated with paper

The following model configurations were used to produce the results in
the paper. 
- params.json

### Analysis
See ipython notebooks in `notebooks`

## References
- Ecker et al. "Hippocampal sharp wave-ripples and the associated sequence replay emerge from structured synaptic interactions in a network model of area CA3." eLife (2022). https://elifesciences.org/articles/71850v1
- Milstein et al. "Offline memory replay in recurrent neuronal networks emerges from constraints on online dynamics". The Journal of Physiology (2023). https://physoc.onlinelibrary.wiley.com/doi/10.1113/JP283216
- Terada et al. "Adaptive stimulus selection for consolidation in the hippocampus." Nature (2022). https://www.nature.com/articles/s41586-021-04118-6
