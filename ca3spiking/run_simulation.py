import datajoint as dj


dj.config['database.host'] = 'localhost'
dj.config['database.user'] = 'djuser'
dj.config['database.password'] = 'simple'

dj.config['stores'] = {
  'ca3-external': dict( # 'raw' storage for this pipeline
                protocol='file',
                location='/data/Zhenrui/datajoint/ca3'),
    'plots': {
            'protocol': 'file',
            'location': '/home/zhenrui/code/packages/ca3spiking/ca3spiking/plots'
            }
}
# external object cache - see fetch operation below for details.
dj.config['cache'] = '/fastscratch/scratch/zhenrui-testing/datajoint'

import ca3spiking as ca3
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

import argparse

from ca3spiking.stdp import learning_with_interneurons
from ca3spiking.spw_network import run_simulation_interneurons, preprocess_monitors

schema = dj.schema('ca3_model', locals())


def repeat_zip(nid, spt):
    for i, times in zip(nid, spt):
        yield np.repeat(i, len(times)), times


@schema
class NetworkParameters(dj.Lookup):
    definition = """
    network_paramset: int
    ---
    n_e : int
    n_i : int
    place_cell_ratio : float
    cue_cell_ratio : float
    in_cue_cell_ratio: float
    
    infield_rate: float
    outfield_rate: float
    """    

@schema
class SynapseParameters(dj.Lookup):
    definition = """
    synapse_paramset: int
    ---
    stdp_type: enum("sym", "asym")
    taup : float
    taum : float
    ap : float
    am : float
    wmax: float
    w_init: float
    itaup: float
    itaum: float    
    scale_factor: float
    ee_plasticity: tinyint
    ie_plasticity: tinyint # I->E
    ei_plasticity: tinyint # E->I
    """        
    
@schema
class OnlineParameters(dj.Lookup):
    definition = """
    online_paramset: int
    ---
    linear: enum("true", "false")
    seed: int
    duration: float    
    """
    
@schema
class OnlineSimulation(dj.Computed):
    definition = """
    -> OnlineParameters
    -> NetworkParameters
    """

    class Neuron(dj.Part):
        definition = """
        -> master
        neuron_id: int
        ---
        brian_id: int                 # Brian tracks interneurons and neurons separately                
        place_cell: tinyint
        cue_cell: tinyint
        interneuron: tinyint
        place_field_start: float
        """
        
    class OnlineSpikeTimes(dj.Part):
        definition = """
        -> master.Neuron
        -> master
        ---
        online_spike_times: longblob
        """
        
    def make(self, key):
        network_params = NetworkParameters() & key
        simulation_params = OnlineParameters() & key
        
        self.insert1(key)
        
        count = itertools.count()        
        
        n_pyramidal = network_params.fetch1("n_e")
        n_interneurons = network_params.fetch1("n_i")
        place_cell_ratio = network_params.fetch1("place_cell_ratio")
        cue_cell_ratio=network_params.fetch1("cue_cell_ratio")
        in_cue_cell_ratio=network_params.fetch1("in_cue_cell_ratio")
        ordered = True
        
        seed = simulation_params.fetch1("seed")
        linear = simulation_params.fetch1("linear")
        
        cue_cells, place_fields = ca3.generate_place_fields(n_pyramidal, place_cell_ratio, linear, ordered=True,
                                                        seed=seed, cue_cell_ratio=cue_cell_ratio)
        spike_trains = ca3.place_fields_to_spike_trains(n_pyramidal, cue_cells, place_fields, linear, seed=seed,
                                                        t_max=simulation_params.fetch1("duration"),
                                                        outfield_rate=network_params.fetch1("outfield_rate"),
                                                        infield_rate=network_params.fetch1("infield_rate"))
        spike_trains = ca3.refractoriness(spike_trains)
        
        for i, spt in enumerate(spike_trains):
            nid = next(count)
            self.Neuron.insert1(dict(key, neuron_id=nid, 
                                                 place_cell=(i in place_fields), 
                                                 cue_cell=(i in cue_cells), 
                                                 interneuron=False, 
                                                 place_field_start=place_fields[i] if i in place_fields else 0.,
                                                 brian_id=nid))
            self.OnlineSpikeTimes.insert1(dict(key, neuron_id=nid, online_spike_times=spt))
        
        
        n_cue_ins = int(n_interneurons * in_cue_cell_ratio)
        
        for i in range(n_cue_ins):
            nid = next(count)
            self.Neuron.insert1(dict(key, neuron_id=nid, 
                                                 place_cell=False, 
                                                 cue_cell=True, 
                                                 interneuron=True, 
                                                 place_field_start=0., 
                                                 brian_id=i)) 
            
            
            spt = ca3.refractoriness([ca3.cue_poisson(network_params.fetch1("infield_rate"), 
                                                     simulation_params.fetch1("duration"), seed + i)])

            self.OnlineSpikeTimes.insert1(dict(key, neuron_id=nid,
                                                           online_spike_times=spt[0]))
            
        
        n_non_cue_ins = n_interneurons - n_cue_ins
        _, in_place_fields = ca3.generate_place_fields(n_non_cue_ins, 1., linear, ordered=ordered, seed=seed, cue_cell_ratio=0.)
        in_spike_trains = ca3.place_fields_to_spike_trains(n_non_cue_ins, set([]), in_place_fields, linear, seed=seed,
                                                        t_max=simulation_params.fetch1("duration"),
                                                        outfield_rate=network_params.fetch1("outfield_rate"),
                                                        infield_rate=network_params.fetch1("infield_rate"))
        in_spike_trains = ca3.refractoriness(in_spike_trains)
        
        for i, spt in enumerate(in_spike_trains):
            nid = next(count)
            self.Neuron.insert1(dict(key, neuron_id=nid, 
                                                 place_cell=i in in_place_fields, 
                                                 cue_cell=False, 
                                                 interneuron=True, 
                                                 place_field_start=in_place_fields[i] if i in in_place_fields else 0., 
                                                 brian_id=i+n_cue_ins))
            self.OnlineSpikeTimes.insert1(dict(key, neuron_id=nid, online_spike_times=spt))
            

@schema
class Weights(dj.Computed):
    definition = """
    -> OnlineSimulation
    -> SynapseParameters
    ---
    ee_weights: blob@ca3-external
    ie_weights: blob@ca3-external # i->e
    ei_weights: blob@ca3-external # e->i
    """
        
    def make(self, key):        
        print("Starting online simulation...")
        syn_params = SynapseParameters() & key
        online_params = OnlineParameters() & key 
        nw_params = NetworkParameters() & key
        
        pcs = (OnlineSimulation.OnlineSpikeTimes() & (OnlineSimulation.Neuron() & 'interneuron = 0'))
        ins = (OnlineSimulation.OnlineSpikeTimes() & (OnlineSimulation.Neuron() & 'interneuron = 1'))
        
        spiking_neurons, spike_times = spike_times_helper(pcs, OnlineSimulation.Neuron())
        spiking_ins, in_spike_times = spike_times_helper(ins, OnlineSimulation.Neuron())
        
        weightmx, in_weight_mx, pc_in_weight_mx = learning_with_interneurons(spiking_neurons, spike_times, 
                                                                             spiking_ins, in_spike_times, 
                                                                             syn_params.fetch1("taup"), 
                                                                             syn_params.fetch1("taum"), 
                                                                             syn_params.fetch1("ap") * syn_params.fetch1("wmax"), 
                                                                             syn_params.fetch1("am") * syn_params.fetch1("wmax"), 
                                                                             syn_params.fetch1("wmax"), 
                                                                             syn_params.fetch1("w_init"),
                                                                             nPCs=nw_params.fetch1("n_e"),
                                                                             seed=online_params.fetch1("seed"),
                                                                             n_place_ins=nw_params.fetch1("n_i"),                                                                             
                                                                             itaum=syn_params.fetch1("itaum"), 
                                                                             itaup=syn_params.fetch1("itaup"),
                                                                             time=online_params.fetch1("duration"),
                                                                             ee_plasticity=syn_params.fetch1("ee_plasticity"),
                                                                             ie_plasticity=syn_params.fetch1("ie_plasticity"),
                                                                             ei_plasticity=syn_params.fetch1("ei_plasticity"))
        key['ee_weights'] = weightmx
        key['ie_weights'] = in_weight_mx
        key['ei_weights'] = pc_in_weight_mx
        
        self.insert1(key)

def spike_times_helper(spike_times_db, neuron_db):
    map_result = [([nid]*len(spt), spt) for nid, spt in zip(*(spike_times_db * neuron_db).fetch('brian_id', 'online_spike_times'))]
    spiking_neurons = []
    spike_times = []

    for neurons, times in map_result:
        spiking_neurons.append(neurons)
        spike_times.append(times)
        
    return np.concatenate(spiking_neurons), np.concatenate(spike_times)


@schema
class OfflineParameters(dj.Lookup):
    definition = """
    offline_paramset: int
    ---
    offline_duration: float
    selection: longblob
    seed: int
    """
    

@schema
class Replay(dj.Computed):
    definition = """
    -> Weights
    -> OfflineParameters
    ---
    spike_times_pc: blob@ca3-external
    spiking_neurons_pc: blob@ca3-external
    rate_pc: blob@ca3-external
    isi_hist_pc: blob@ca3-external
    bin_edges_pc: blob@ca3-external
    spike_times_in: blob@ca3-external
    spiking_neurons_in: blob@ca3-external
    rate_in: blob@ca3-external
    """


    # Use source->target convention throughout
    
    def _make_tuples(self, key):
        weights = (Weights() & key).fetch1()
        
        wmx_PC_E = weights["ee_weights"] * 1e9
        # swapped 6/30
        wmx_IN_PC = weights["ie_weights"] * 1e9
        wmx_PC_IN = weights["ei_weights"] * 1e9
        
        stdp_mode = (SynapseParameters() & key).fetch1("stdp_type")
        syn_params = SynapseParameters() & key
        
        offline_parameters = (OfflineParameters() & key).fetch1()
        
        place_cells = ((OnlineSimulation.Neuron() & key) & 'place_cell = 1' & 'interneuron = 0').fetch("brian_id")
        cue_cells = ((OnlineSimulation.Neuron() & key) & 'cue_cell = 1'  & 'interneuron = 0').fetch("brian_id")        
        place_ins = ((OnlineSimulation.Neuron() & key) & 'interneuron = 1').fetch("brian_id")
        
        SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation_interneurons(wmx_PC_E, 
                                                                                                  wmx_IN_PC, 
                                                                                                  wmx_PC_IN, 
                                                                                                  STDP_mode=stdp_mode, 
                                                                                                  place_cells=place_cells,
                                                                                                  cue_cells=cue_cells,
                                                                                                  n_place_ins=len(place_ins), 
                                                                                                  selection=offline_parameters['selection'],
                                                                                                  seed=offline_parameters['seed'],
                                                                                                  time=offline_parameters['offline_duration'],
                                                                                                 ee_plasticity=syn_params.fetch1("ee_plasticity"),
                                                                                                 ie_plasticity=syn_params.fetch1("ie_plasticity"),
                                                                                                 ei_plasticity=syn_params.fetch1("ei_plasticity"),
                                                                                                  verbose=True)
        
        spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        
        key['spike_times_pc'] = spike_times_PC
        key['spiking_neurons_pc'] = spiking_neurons_PC
        key['rate_pc'] = rate_PC
        key['isi_hist_pc'] = ISI_hist_PC
        key['bin_edges_pc'] = bin_edges_PC
        key['spike_times_in'] = spike_times_BC
        key['spiking_neurons_in'] = spiking_neurons_BC
        key['rate_in'] = rate_BC
        
        self.insert1(key)

@schema
class OfflineSimulation(dj.Computed):
    definition = """
    -> Weights
    -> OfflineParameters
    ---
    """
    class OfflineSpikeTimes(dj.Part):
        definition = """
        -> OnlineSimulation.Neuron
        -> master
        ---
        offline_spike_times: blob@ca3-external
        """

    class OfflinePotentials(dj.Part):
        definition = """
        -> OnlineSimulation.Neuron
        -> master
        ---
        offline_vm: blob@ca3-external
        """
        
    class OfflineRate(dj.Part):
        definition = """
        -> OnlineSimulation.Neuron
        -> master
        ---
        rate: float
        """
        
    def make(self, key):
        
        self.insert1(key)
        
        SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = self.run_simulation(key)
        
        offline_spikes = self.OfflineSpikeTimes() & key
        neuron = OnlineSimulation.Neuron() & key
         
        excitatory_lookup = pd.DataFrame(neuron & 'interneuron = 0').set_index("brian_id")['neuron_id']
        inhibitory_lookup = pd.DataFrame(neuron & 'interneuron = 1').set_index("brian_id")['neuron_id']
        
        offline_spikes.insert({'neuron_id':excitatory_lookup[brian_id], **key, 'offline_spike_times': spt} 
                              for brian_id, spt in SM_PC.spike_trains().items())
        offline_spikes.insert({'neuron_id':inhibitory_lookup[brian_id], **key, 'offline_spike_times': spt} 
                              for brian_id, spt in SM_BC.spike_trains().items())
        
        #import pudb
        #pudb.set_trace()
        #offline_rate = self.OfflineRate() & key
        #offline_rate.insert({'neuron_id':excitatory_lookup[brian_id], **key, 'rate': rate} 
        #                      for brian_id, rate in enumerate(np.array(RM_PC.rate_).reshape(-1, 10).mean(axis=1)))
        #offline_rate.insert({'neuron_id':inhibitory_lookup[brian_id], **key, 'rate': rate} 
        #                      for brian_id, rate in enumerate(np.array(RM_BC.rate_).reshape(-1, 10).mean(axis=1)))        
        
        
    
    def run_simulation(self, key):
        print("Starting offline simulation...")
        weights = (Weights() & key).fetch1()

        wmx_PC_E = weights["ee_weights"] * 1e9
        wmx_IN_PC = weights["ie_weights"] * 1e9
        wmx_PC_IN = weights["ei_weights"] * 1e9

        stdp_mode = (SynapseParameters() & key).fetch1("stdp_type")

        offline_parameters = (OfflineParameters() & key).fetch1()

        place_cells = ((OnlineSimulation.Neuron() & key) & 'place_cell = 1' & 'interneuron = 0').fetch("brian_id")
        cue_cells = ((OnlineSimulation.Neuron() & key) & 'cue_cell = 1'  & 'interneuron = 0').fetch("brian_id")
        place_ins = ((OnlineSimulation.Neuron() & key) & 'interneuron = 1').fetch("brian_id")

        SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation_interneurons(wmx_PC_E,
                                                                                                  wmx_IN_PC,
                                                                                                  wmx_PC_IN,
                                                                                                  STDP_mode=stdp_mode,
                                                                                                  place_cells=place_cells,
                                                                                                  cue_cells=cue_cells,
                                                                                                  n_place_ins=len(place_ins),
                                                                                                  selection=offline_parameters['selection'],
                                                                                                  seed=offline_parameters['seed'],
                                                                                                  time=offline_parameters['offline_duration'],
                                                                                                  verbose=True)
        return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC


@schema
class ReplayRaster(dj.Computed):
    definition = """
    -> OfflineSimulation
    ---
    raster: attach@plots
    """
    
    def _make_tuples(self, key):
        sns.set(style='ticks', context='talk')
        offline_simulation = OfflineSimulation() & key
        neuron_id, offline_spike_times, is_cue, is_interneuron = (offline_simulation.OfflineSpikeTimes() * OnlineSimulation.Neuron()).fetch(
                                                                    "neuron_id", 'offline_spike_times', 'cue_cell', 'interneuron')
        fig, ax = plt.subplots()
        
        nid, t = list(zip(*repeat_zip(neuron_id, offline_spike_times)))
        nid = np.array(nid)
        t = np.array(t)
        idx = np.arange(len(nid))

        try:
            in_idx, = np.where(is_interneuron& (idx%100 == 0))
            ax.scatter(np.concatenate(t[in_idx]), np.concatenate(nid[in_idx]), marker='|', color='r')
        except:
            pass
        
        try:
            cue_idx, = np.where(is_cue& (idx%100 == 0))
            ax.scatter(np.concatenate(t[cue_idx]), np.concatenate(nid[cue_idx]), marker='|', color='goldenrod')
        except:
            pass
        
        try:
            cue_in_idx = np.where(is_cue & is_interneuron & (idx%100 == 0))
            ax.scatter(np.concatenate(t[cue_in_idx]), np.concatenate(nid[cue_in_idx]), marker='|', color='g')        
        except:
            pass
        
        try:
            other_idx, = np.where((~is_interneuron) &(~is_cue) & (idx%100 == 0))
            ax.scatter(np.concatenate(t[other_idx]), np.concatenate(nid[other_idx]), marker='|', color='k')
        except:
            pass
        
        ax.set_xlabel("time (s)")
        ax.set_ylabel("neuron #")
        fig.savefig("raster.pdf", bbox_inches='tight')
        self.insert1({**key, 'raster': 'raster.pdf'})


if __name__ == "__main__":
    import json as json
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", "-c", action="store_true")
    args = parser.parse_args()

    if args.clear:
        schema.drop(force=True)
        quit()

    with open('params.json') as f:
        params = json.load(f)

    network_parameters = NetworkParameters()
    network_parameters.insert1(params['network'] , skip_duplicates=True)

    simulation_parameters = OnlineParameters()
    simulation_parameters.insert1(params['online'], skip_duplicates=True)

    synapse_parameters = SynapseParameters()
    #synapse_parameters.insert1(params['synapse'], skip_duplicates=True)
    synapse_parameters.insert1(params['synapse2'], skip_duplicates=True)
    #synapse_parameters.insert1(params['synapse3'], skip_duplicates=True)

    online_simulation = OnlineSimulation()
    online_simulation.populate()

    weights = Weights()
    weights.populate()

    offline_parameters = OfflineParameters()
    offline_parameters.insert1(params['offline'], skip_duplicates=True)
    offline = OfflineSimulation()
    offline.populate()

    raster = ReplayRaster()
    raster.populate()
