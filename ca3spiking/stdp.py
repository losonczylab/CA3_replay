# -*- coding: utf8 -*-

from .ecker2022 import *

import os, sys, warnings
import numpy as np
import random as pyrandom
from brian2 import *
set_device("cpp_standalone", build_on_run=False) # True  # speed up the simulation with generated C++ code
import matplotlib.pyplot as plt
from .helper import load_spike_trains, save_wmx

from brian2.devices.device import active_device

warnings.filterwarnings("ignore")
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

def learning_with_interneurons(spiking_neurons, spike_times, spiking_ins, 
        in_spike_times, taup, taum, Ap, Am, wmax, w_init, time=400, nPCs=9000, 
        n_place_ins=100, seed=12345, itaup=None, itaum=None, ee_plasticity=True, 
        ie_plasticity=True, ei_plasticity=True):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other, and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param spiking_neurons, spike_times: np.arrays for Brian2's SpikeGeneratorGroup (list of lists created by `generate_spike_train.py`) - spike train used for learning
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :return weightmx: learned synaptic weights
    """

    device = get_device()

    device.reinit()
    device.activate(build_on_run=False)

    start_scope()

    if itaup is None:
        itaup = taup
    if itaum is None:
        itaum = taum

    taup *= ms
    taum *= ms
    itaup *= ms
    itaum *= ms

    np.random.seed(seed)
    pyrandom.seed(seed)

    fixed_synapse = """
    w : 1
    """

    #plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")

    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    if ee_plasticity:
        STDP = Synapses(PC, PC,
                """
                w : 1
                dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
                dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
                """,
                on_pre="""
                A_presyn += Ap
                w = clip(w + A_postsyn, 0, wmax)
                """,
                on_post="""
                A_postsyn += Am
                w = clip(w + A_presyn, 0, wmax)
                """)
        STDP.connect(condition="i!=j", p=connection_prob_PC)
        STDP.w = w_init
    else:
        STDP = Synapses(PC, PC, fixed_synapse)
        STDP.connect(condition="i!=j", p=connection_prob_PC)
        STDP.w = w_init#*np.random.rand(STDP.w.shape)


    IN = SpikeGeneratorGroup(n_place_ins, spiking_ins, in_spike_times*second)

    if ie_plasticity:
        print("using i->e plasticity")
        IN_STDP = Synapses(IN, PC,
                """
                w : 1
                dA_presyn/dt = -A_presyn/(itaup) : 1 (event-driven)
                dA_postsyn/dt = -A_postsyn/(itaum) : 1 (event-driven)
                """,
                on_pre="""
                A_presyn += Ap
                w = clip(w + A_postsyn, 0, wmax) # +!
                """,
                on_post="""
                A_postsyn += Am
                w = clip(w + A_presyn, 0, wmax) # +!
                """)
        IN_STDP.connect(p=connection_prob_BC)#PC_pIN)
        IN_STDP.w = w_init
    else:
        print("not using i->e plasticity")
        IN_STDP = Synapses(IN, PC, fixed_synapse)
        IN_STDP.connect(p=connection_prob_BC)#PC_pIN)
        IN_STDP.w = w_init# * np.random.rand(IN_STDP.w.shape)


    if ei_plasticity:
        print("using E->I plasticity")
        PC_IN_STDP = Synapses(PC, IN,
                """
                w : 1
                dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
                dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
                """,
                on_pre="""
                A_presyn += Ap
                w = clip(w + A_postsyn, 0, wmax) # +!
                """,
                on_post="""
                A_postsyn += Am
                w = clip(w + A_presyn, 0, wmax) # +!
                """)
        PC_IN_STDP.connect(p=connection_prob_PC_pIN)
        PC_IN_STDP.w = w_init
    else:
        print("Not using E->I plasticity")
        PC_IN_STDP = Synapses(PC, IN, fixed_synapse)
        PC_IN_STDP.connect(p=connection_prob_PC_pIN)
        PC_IN_STDP.w = w_init #* np.random.rand(PC_IN_STDP.w.shape)


    run(time*second, report="text")
    device.build(directory=None, clean=False, compile=True, run=True, debug=False)

    in_weight_mx = np.zeros((n_place_ins, nPCs))

    if ie_plasticity:
        in_weight_mx[IN_STDP.i[:], IN_STDP.j[:]] = IN_STDP.w[:]
    else:
        in_weight_mx[IN_STDP.i[:], IN_STDP.j[:]] = np.random.rand(*IN_STDP.w[:].shape) * w_init #IN_STDP.w[:]

    pc_in_weight_mx = np.zeros((nPCs, n_place_ins))

    if ei_plasticity:
        pc_in_weight_mx[PC_IN_STDP.i[:], PC_IN_STDP.j[:]] = PC_IN_STDP.w[:]
    else:
        pc_in_weight_mx[PC_IN_STDP.i[:], PC_IN_STDP.j[:]] = np.random.rand(*PC_IN_STDP.w[:].shape) * w_init

    weightmx = np.zeros((nPCs, nPCs))

    if ee_plasticity:
        weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]
    else:
        weightmx[STDP.i[:], STDP.j[:]] = np.random.rand(*STDP.w[:].shape) * w_init


    return weightmx, in_weight_mx, pc_in_weight_mx