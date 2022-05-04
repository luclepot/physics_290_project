from argparse import ArgumentParser
import sys
import os

_LOGMSG = "Converter :: "

def _get_host_id(hostname, known_hosts):
    """Parse hostname string using known_hosts dictionary of form {<glob>:<id>} to 
    find and return matching id for a given hostname. Throws error on multiple matching.
    """
    host_list = []
    for host in known_hosts.keys():
        if glob.fnmatch.fnmatch(hostname, host):
            host_list.append(host)
    if len(host_list) > 1:
        _logger.error("Host specs {} all match host with name '{}'".format(host_list, hostname))
        return None
    elif len(host_list) < 1:
        _logger.error("Current hostname '{}' not recognized (checked {})".format(hostname, list(known_hosts.keys())))
        return None
    else:
        return known_hosts[host_list[0]]

def _smartpath(s):
    if s.startswith('~'):
        return s
    return os.path.abspath(s)

def log(
    msg
):
    if isinstance(msg, str):
        for line in msg.split('\n'):
            print(_LOGMSG + line)
    else:
        print(_LOGMSG + str(msg))
# do argparse stuff
parser = ArgumentParser()
parser.add_argument('-o', '--output', dest="outputdir", action="store", type=_smartpath, help="output dir path", required=True)
parser.add_argument('-i', '--input', dest="inputdir", action="store", type=_smartpath, help="input dir globstring", required=True)    
parser.add_argument('-n', '--n-samples', dest='max_events', action='store', type=int, default=-1, help='take only first N samples of selection')
parser.add_argument('-p', '--pt-threshold', dest='pt_threshold', action='store', type=int, default=200, help='Transverse momentum requirement for jet pt')

argv = sys.argv[1:]
args = None
if len(argv) > 0:
    args = parser.parse_args(argv)
else:
    parser.print_help()
    sys.exit(0)
log('checking input arguments')
    
# interpret argparse stuff and start program
input_globstr = args.inputdir
output_dir = args.outputdir
max_events = args.max_events
pt_threshold = args.pt_threshold

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# hlf_path = os.path.join(output_dir, 'hlf.h5')
# const_path = os.path.join(output_dir, 'constituents.h5')
data_path = os.path.join(output_dir, 'data.h5')
if os.path.exists(data_path):                       #os.path.exists(hlf_path) or os.path.exists(const_path):
    raise FileExistsError("output directory {} is already full!".format(output_dir))

log('passed input logic check')

# MAIN PROGRAM START

import numpy as np
import math
import os
import argparse
import sys
import time
from traceback import format_exc
import h5py
import ROOT as rt
from collections import OrderedDict as odict
import glob 
import tqdm
import pandas as pd

# do host stuff
_HOSTNAME = os.uname()[1]
_KNOWN_HOSTS = {
    'DESKTOP-H3LO5BL': 'lenovo',
    'cori*': 'cori',
}
host = _get_host_id(_HOSTNAME, _KNOWN_HOSTS)

# load Delphes
log('loading Delphes...')
if host == 'cori':
    DELPHES_DIR = os.environ["DELPHES_DIR"]
    rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))
elif host == 'lenovo':
    rt.gSystem.Load("libDelphes.so")

log('loading rootfiles...')
tree_name = "Delphes"
files = glob.glob(input_globstr)

log('Found {} files'.format(len(files)))

tree = rt.TChain("Delphes")
for f in files:
    tree.AddFile(f)

tree_size = tree.GetEntries()
if max_events < 0:
    max_events = tree_size
max_constituents = 100

constituent_feature_names = ['Eta', 'Phi', 'Mass', 'Pt']
jet_feature_names = [
    'Eta',
    'Phi',
    'Charge',
    'ChargedEnergyFraction',
    'EhadOverEem',
    'PT',
    'Mass',
    'NCharged',
    'NNeutrals',
    'NSubJetsPruned',
    'NSubJetsSoftDropped',
    'NeutralEnergyFraction',
    'PTD',
]
jet_OTHERfeature_names = [
    'event_number',
    'jet_number',
]

log('creating datasets...')

N_pt_pass = tree.Draw("FatJet.PT>>hist","FatJet.PT>{}".format(pt_threshold),"goff")
N_to_use = min([max_events, N_pt_pass])

const_arr = -np.ones(shape=(N_to_use, max_constituents, len(constituent_feature_names)))
hlf_arr = np.empty(shape=(N_to_use, len(jet_feature_names) + len(jet_OTHERfeature_names)))
N_jets = 0

class LoopFinished(Exception): pass

log('Running main loop, expecting {} total passed events from {} files'.format(N_to_use, len(files)))
pbar = tqdm.tqdm(position=0, leave=True)

i = 0
try:
    for event_count, event in enumerate(tree):
        for jet_count, jet in enumerate(event.FatJet):
            pbar.set_description(_LOGMSG + "{}/{}".format(i,N_to_use))
            if i >= N_to_use:
                pbar.close()
                raise LoopFinished
            pbar.update(1)

            N_jets += 1
            if jet.PT < pt_threshold:
                continue
            # add constituent p4s
            for const_count,c in zip(range(max_constituents), jet.Constituents):
                p4 = c.P4()
                const_arr[i, const_count, :] = [p4.Eta(), p4.Phi(), p4.M(), p4.Pt()]

            # add hlfs
            for jf_idx,name in enumerate(jet_feature_names):
                hlf_arr[i, jf_idx] = getattr(jet, name)
            hlf_arr[i,-1] = jet_count
            hlf_arr[i,-2] = event_count
            i += 1

except LoopFinished:
    pass

pbar.close()

with h5py.File(data_path, 'w') as f:
    dset = f.create_dataset('const', const_arr.shape, dtype=float)
    dset[:] = const_arr

    dset_hlf = f.create_dataset('hlf', hlf_arr.shape, dtype=float)
    dset_hlf[:] = hlf_arr

log("Saved {} events to path {}".format(i, data_path))

            # const_arr[i,j,k] = [c.Eta, c.Phi, c.M, c.Pt]
# class Converter:

#     LOGMSG = "Converter :: "

#     def __init__(
#         self,
#         input_globstr,
#         outputdir,
#         name,
#         n_constituent_particles=100,
#         save_constituents=False,
#     ):
#         self.outputdir = outputdir
#         self.name = name
#         self.input_files = glob.glob(input_globstr)

#         # core tree, add files, add all trees
#         self.tree = rt.TChain("Delphes")
#         for f in self.input_files:
#             self.tree.AddFile(f)
#         # self.files = [rf for rf in [rt.TFile(f) for f in self.inputfiles] if rf.GetListOfKeys().Contains("Delphes")]
    
#         self.nEvents = self.tree.GetEntries()

#         self.log("Found {0} files".format(len(self.input_files)))
#         self.log("Found {0} total events".format(self.nEvents))
#         self.n_jets
#         self.jet_feature_names = [
#             'Eta',
#             'Phi',
#             'Charge',
#             'ChargedEnergyFraction',
#             'EhadOverEem',
#             'Pt',
#             'Mass',
#             'NCharged',
#             'NNeutrals',
#             'NSubJetsPruned',
#             'NSubJetsSoftDropped',
#             'NeutralEnergyFraction',
#             'PT',
#             'PTD',
#             'Energy',
#         ]

#         self.jet_constituent_names = [
#             'Eta',
#             'Phi',
#             'PT',
#             'Rapidity',
#             'Energy',
#         ]

#         self.n_constituent_particles=n_constituent_particles
#         self.jet_features = None
#         self.jet_constituents = None

#         hlf_dict = {}
#         particle_dict = {}

#         self.save_constituents = save_constituents

#         # self.selections_abs = np.asarray([sum(self.sizes[:s[0]]) + s[1] for s in self.selections])
#         self.log("found {0} selected events, out of a total of {1}".format(sum(map(len, self.selections.values())), self.nEvents))

#     def log(
#         self,
#         msg
#     ):
#         if isinstance(msg, str):
#             for line in msg.split('\n'):
#                 print(self.LOGMSG + line)
#         else:
#             print(self.LOGMSG + str(msg))

#     def convert(
#         self,
#         rng=(-1,-1),
#         return_debug_tree=False
#     ):
#         rng = list(rng)

#         if rng[0] < 0:
#             rng[0] = 0

#         if rng[1] > self.nEvents or rng[1] < 0:
#             rng[1] = self.nEvents

#         nmin, nmax = rng
        
#         total_size = nmax - 1 - nmin

#         self.log("selecting on range {0}".format(rng))
        
#         self.jet_features = np.empty((total_size, self.n_jets, len(self.jet_feature_names)))
#         self.log("jet feature shapes: {}".format(self.event_features.shape))

#         self.jet_constituents = -np.ones((total_size, self.n_jets, self.n_constituent_particles, len(self.jet_constituent_names)))
#         self.log("jet constituent shapes: {}".format(self.jet_constituents.shape))
    
#         if not self.save_constituents:
#             self.log("ignoring jet constituents")
 
#         for event_n, tree_index in enumerate(range(nmin, nmax)):

#             self.log('event {0}, index {1}'.format(event_n, tree_index))

#             # tree (and, get the entry)
#             self.tree.GetEntry(tree_index)

#             # jets
#             jets_raw = [self.tree.FatJet[i] for i in range(min([self.n_jets, self.tree.Jet_size]))]
#             jets = [j.P4() for j in jets_raw]
            
#             # constituent 4-vectors per jet
#             constituents_by_jet = [j.constituents for j in jets_raw]

#             for jet_n, (jet_raw, jet_p4, constituents) in enumerate(zip(jets_raw, jets, constituents_by_jet)):
            
#                 self.jet_features[total_count, jet_n, :] =  self.get_jet_features(jet_raw, constituents)
                
#                 if self.save_constituents:
#                     self.jet_constituents[total_count, jet_n, :] = self.get_jet_constituents(constituents)
                
#             total_count += 1 
        
#         return None


# # if __name__ == "__main__":
# #     if len(sys.argv) == 10:
# #         (_, outputdir, pathspec, name, dr, nc, rmin, rmax, constituents, basis_n) = sys.argv
# #         # print outputdir
# #         # print pathspec
# #         # print filespecls 
# #         core = Converter(outputdir, pathspec, name, float(dr), int(nc), bool(int(constituents)), int(basis_n))
# #         ret = core.convert((int(rmin), int(rmax)))
# #         core.save()
# #     # elif len(sys.argv) == 0:
# #     else:
# #         print "TEST MODE"

# #         core = Converter(".", '/afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/data/background/process/data_0_selection.txt', "data", save_constituents=True, energyflow_basis_degree=-1, n_constituent_particles=100)
# #         ret = core.convert((0,100), return_debug_tree=False)
# #         core.save("TEST2.h5")