# -*- coding: utf8 -*-
from .ecker2022 import *

import os
import sys
import shutil
import numpy as np
import random as pyrandom
from brian2 import *
set_device("cpp_standalone", build_on_run=False)
import matplotlib.pyplot as plt
from .helper import preprocess_monitors

import pickle as pkl

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

def run_simulation_interneurons(wmx_PC_E, wmx_IN_PC, wmx_PC_IN, STDP_mode, 
        place_cells, cue_cells, n_place_ins, selection, seed, time=10, 
        ei_plasticity=True, ee_plasticity=True, ie_plasticity=True, verbose=True):
    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param STDP_mode: asym/sym STDP mode used for the learning (see `stdp.py`) - here used only to set the weights
    :param cue: if True it adds an other Brian2 `SpikeGeneratorGroup` to stimulate a subpop in the beginning (cued replay)
    :param save: bool flag to save PC spikes after the simulation (used by `bayesian_decoding.py` later)
    :param seed: random seed used for running the simulation
    :param verbose: bool flag to report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: Brian2 monitors (+ array of selected cells used by multi state monitor)
    """
    
    device = get_device()
    device.reinit()
    device.activate(build_on_run=False)

    start_scope()
    np.random.seed(seed)
    pyrandom.seed(seed)

    print("Starting...")
    # synaptic weights (see `/optimization/optimize_network.py`)
    w_PC_I = 0.65  # nS
    w_BC_E = 0.85
    w_BC_I = 5.
    if STDP_mode == "asym":
        w_PC_MF = 21.5
    elif STDP_mode == "sym":
        w_PC_MF = 19.15
    else:
        raise ValueError("STDP_mode has to be either 'sym' or 'asym'!")


    nPCs = len(wmx_PC_E)
    nBCs = len(wmx_IN_PC)

    PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0

    I_ext = np.zeros(nPCs)
    if cue_cell_ratio > 0:
        I_ext[np.array(cue_cells)] = 0. #10
    PCs.external_inhibition = I_ext*nA 

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    # PLACE INTERNEURONS
    place_INs = NeuronGroup(n_place_ins, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    place_INs.vm  = Vrest_BC; place_INs.g_ampa = 0.0; place_INs.g_gaba = 0.0

    MF = PoissonGroup(nPCs, rate_MF)
    C_PC_MF = Synapses(MF, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
    C_PC_MF.connect(j="i")

    # weight matrix used here
    C_PC_E = Synapses(PCs, PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    nonzero_weights = np.nonzero(wmx_PC_E)
    C_PC_E.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    if ee_plasticity:
        C_PC_E.w_exc = wmx_PC_E[nonzero_weights].flatten()
        del wmx_PC_E

    C_IN_PC = Synapses(place_INs, PCs, "w_inh:1", on_pre="x_gaba+=norm_PC_I*w_inh", delay=delay_PC_I)
    nonzero_weights = np.nonzero(wmx_IN_PC)
    C_IN_PC.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    if ie_plasticity:
        C_IN_PC.w_inh = wmx_IN_PC[nonzero_weights].flatten()
        del wmx_IN_PC

    C_PC_IN = Synapses(PCs, place_INs, "w_exc2:1",  on_pre="x_ampa+=norm_BC_E*w_exc2", delay=delay_BC_E)
    nonzero_weights = np.nonzero(wmx_PC_IN)
    #C_PC_IN.connect(p=connection_prob_PC)
    C_PC_IN.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    if ei_plasticity:
        C_PC_IN.w_exc2 = wmx_PC_IN[nonzero_weights].flatten()
        del wmx_PC_IN


    # Wrong convention, oops
    C_PC_I = Synapses(BCs, PCs, on_pre="x_gaba+=norm_PC_I*w_PC_I", delay=delay_PC_I)
    C_PC_I.connect(p=connection_prob_BC)

    C_BC_E = Synapses(PCs, BCs, on_pre="x_ampa+=norm_BC_E*w_BC_E", delay=delay_BC_E) # 1.1 # 1.8 x ??? 
    C_BC_E.connect(p=connection_prob_PC)

    C_BC_I = Synapses(BCs, BCs, on_pre="x_gaba+=norm_BC_I*w_BC_I", delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)

    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PCs)
    RM_BC = PopulationRateMonitor(BCs)

    StateM_PC = StateMonitor(PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"], record=selection, dt=0.1*ms)
    StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    print("Running")
    if verbose:
        run(time*1000*ms, report="text")
    else:
        run(time*1000*ms)
    device.build(directory='output', clean=False, compile=True, run=True, debug=False)

    return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC
