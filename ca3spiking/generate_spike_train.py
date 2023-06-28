# -*- coding: utf8 -*-

import os, pickle
import numpy as np
from tqdm import tqdm  # progress bar
from .poisson_proc import hom_poisson, inhom_poisson, inhom_poisson_cue, inhom_poisson_interneuron, cue_poisson
from .helper import save_place_fields, refractoriness 

from .ecker2022 import *

base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

def generate_place_fields(n_neurons, place_cell_ratio, linear, ordered=True, seed=1234, cue_cell_ratio=0., interneuron=False):
    """
    Generates hippocampal like spike trains (used later for learning the weights via STDP)
    :param n_neurons: #{neurons}
    :param place_cell_ratio: ratio of place cells in the whole population
    :param linear: flag for linear vs. circular track
    :param ordered: bool to order neuronIDs based on their place fields (used for teaching 2 environments - see `stdp_2nd_env.py`)
    :param seed: starting seed for random number generation
    :return: spike_trains - list of lists with indiviual neuron's spikes
    """

    ##assert n_neurons >= 1000, "The assumptions made during the setup hold only for a reasonably big group of neurons"

    neuronIDs = np.arange(0, n_neurons)
    # generate random neuronIDs being place cells and starting points for place fields
    if ordered:
        np.random.seed(seed)

        if linear:
            p_uniform = 1./n_neurons
            tmp = (1 - 2*2*100*p_uniform)/(n_neurons-200)
            p = p_uniform * np.ones(n_neurons)
            #p = np.concatenate([2*p_uniform*np.ones(100), tmp*np.ones(n_neurons-2*100), 2*p_uniform*np.ones(100)])  # slightly oversample (double prop.) the 2 ends (first and last 100 neurons) of the track
            place_cells = np.sort(np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), p=p, replace=False), kind="mergsort") # np.arange(int(n_neurons*place_cell_ratio))#
        else:
            place_cells = np.sort(np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), replace=False), kind="mergsort") # np.arange(int(n_neurons*place_cell_ratio))#
        phi_starts = np.sort(np.random.rand(n_neurons), kind="mergesort")[place_cells] * 2*np.pi

        # if linear:
        #     phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
        #     pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl"%place_cell_ratio)
        # else:
        #     pklf_name = os.path.join(base_path, "files", "PFstarts_%s.pkl"%place_cell_ratio)
        if linear:
            phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
            pklf_name = os.path.join(base_path, "files", f"{'INTERNEURON_' if interneuron else ''}PFstarts_{place_cell_ratio}_linear_no.pkl")
        else:
            pklf_name = os.path.join(base_path, "files", f"{'INTERNEURON_' if interneuron else ''}PFstarts_{place_cell_ratio}_no.pkl")        

    else:
        np.random.seed(seed+1)

        place_cells = np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), replace=False) # np.arange(int(n_neurons*place_cell_ratio)) #
        phi_starts = np.sort(np.random.rand(n_neurons)[place_cells], kind="mergesort") * 2*np.pi

        if linear:
            phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
            pklf_name = os.path.join(base_path, "files", f"{'INTERNEURON_' if interneuron else ''}PFstarts_{place_cell_ratio}_linear_no.pkl")
        else:
            pklf_name = os.path.join(base_path, "files", f"{'INTERNEURON_' if interneuron else ''}PFstarts_{place_cell_ratio}_no.pkl")


    place_fields = {neuron_id:phi_starts[i] for i, neuron_id in enumerate(place_cells)}
    if cue_cell_ratio > 0:
        import pickle as pkl
        cue_cells = place_cells[1::int(1./cue_cell_ratio)]
        #with open(os.path.join(base_path, "files", "cue_cells.pkl"), "wb") as f:
        #    pkl.dump(cue_cells, f)
    else:
        cue_cells = set([])

    #save_place_fields(place_fields, pklf_name)

    return cue_cells, place_fields

def place_fields_to_spike_trains(n_neurons, cue_cells, place_fields, linear, infield_rate, 
        outfield_rate, t_max, seed=1234, interneuron=False):
    # generate spike trains
    spike_trains = []
    for neuron_id in tqdm(range(0, n_neurons)):
        if neuron_id in cue_cells:
            spike_train = inhom_poisson_cue(infield_rate, t_max, place_fields[neuron_id], linear, seed)
        elif neuron_id in place_fields:
            #if interneuron:
            #    spike_train = inhom_poisson_interneuron(infield_rate, t_max, place_fields[neuron_id], linear, seed)
            #else:
            # NO INVERSE PFS!
            spike_train = inhom_poisson(infield_rate, t_max, place_fields[neuron_id], linear, seed)
        else:
            spike_train = hom_poisson(outfield_rate, 100, t_max, seed)
        spike_trains.append(spike_train)
        seed += 1

    return spike_trains

def generate_spike_train(n_neurons, place_cell_ratio, linear, infield_rate, ordered=True, seed=1234, cue_cell_ratio=0.):
    cue_cells, place_fields = generate_place_fields(n_neurons, place_cell_ratio, linear, ordered=ordered, 
                                                    infield_rate=infield_rate, seed=seed, cue_cell_ratio=cue_cell_ratio)
    spike_trains = place_fields_to_spike_trains(n_neurons, cue_cells, place_fields, linear, seed=seed)

    return spike_trains

def generate_place_interneuron_spike_train(n_neurons, linear, infield_rate, ordered=True, seed=1234):
    cue_cells, place_fields = generate_place_fields(n_neurons, place_cell_ratio=0.01, linear=linear, ordered=ordered, 
                                                    seed=seed, cue_cell_ratio=1., interneuron=True)
    in_spike_trains = place_fields_to_spike_trains(n_neurons, cue_cells, place_fields, linear=linear, 
            infield_rate=infield_rate, seed=seed, interneuron=True)

    return in_spike_trains

def jitter(spike_train, sigma=0.001):
    return np.clip(spike_train + sigma*np.random.randn(len(spike_train)), 0, None)


def joint_generate_place_interneurons(n_neurons, n_interneurons, place_cell_ratio, linear, infield_rate, 
        ordered=True, seed=1234, cue_cell_ratio=0., in_cue_cell_ratio=0.):
    cue_cells, place_fields = generate_place_fields(n_neurons, place_cell_ratio, linear, ordered=ordered, 
                                                    seed=seed, cue_cell_ratio=cue_cell_ratio)
    spike_trains = place_fields_to_spike_trains(n_neurons, cue_cells, place_fields, linear, 
            infield_rate=infield_rate, seed=seed)

    n_cue_ins = int(n_interneurons * in_cue_cell_ratio)

    if n_cue_ins > 0:
        in_spike_trains = [cue_poisson(infield_rate, t_max, seed + i) for i in range(n_cue_ins)]
        #cue_ins = np.random.choice(cue_cells, size=n_cue_ins, replace=True)
    else:
        cue_ins = []
    n_non_cue_ins = n_interneurons - n_cue_ins

    #in_spike_trains = [jitter(spike_trains[i]) for i in cue_ins]

    if n_non_cue_ins > 0:
        _, in_place_fields = generate_place_fields(n_non_cue_ins, 1., linear, ordered=ordered, seed=seed, cue_cell_ratio=0.)
        in_spike_trains.extend(place_fields_to_spike_trains(n_non_cue_ins, set([]), in_place_fields, linear,
            infield_rate=infield_rate, seed=seed))

    return spike_trains, in_spike_trains
