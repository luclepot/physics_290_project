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

_HOSTNAME = os.uname()[1]
_KNOWN_HOSTS = {
    'DESKTOP-H3LO5BL': 'lenovo',
    'cori*': 'cori',
}
host = _get_host_id(_HOSTNAME, _KNOWN_HOSTS)

# load Delphes
if host == 'cori':
    DELPHES_DIR = os.environ["DELPHES_DIR"]
    rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))
elif host == 'lenovo':
    rt.gSystem.Load("libDelphes.so")

class Converter:

    LOGMSG = "Converter :: "

    def __init__(
        self,
        outputdir,
        spath,
        name,
        jetDR=0.5,
        n_constituent_particles=100,
        save_constituents=False,
        energyflow_basis_degree=-1,
    ):
        self.outputdir = outputdir

        self.name = name

        self.selections = get_data_dict([spath])
        self.inputfiles = list(self.selections.keys())

        # core tree, add files, add all trees
        self.files = [rf for rf in [rt.TFile(f) for f in self.inputfiles] if rf.GetListOfKeys().Contains("Delphes")]
        self.trees = [tf.Get("Delphes") for tf in self.files]
        self.sizes = [int(t.GetEntries()) for t in self.trees]
        self.nEvents = sum(self.sizes)

        self.log("Found {0} files".format(len(self.files)))
        self.log("Found {0} delphes trees".format(len(self.trees)))
        self.log("Found {0} total events".format(self.nEvents))

        self.jetDR = jetDR

        self.n_jets = 2

        self.jet_feature_names = [
            'Eta',
            'Phi',
            'Pt',
            'M',
            'ChargedFraction',
            'PTD',
            'Axis2',
            'Flavor',
            'Energy',
        ]

        ## ADD MET
        self.event_feature_names =  [
            'MET',
            'METEta',
            'METPhi',
            'MT',
            'Mjj',
        ]

        self.jet_constituent_names = [
            'Eta',
            'Phi',
            'PT',
            'Rapidity',
            'Energy',
        ]

        self.n_constituent_particles=n_constituent_particles
        self.event_features = None
        self.jet_features = None
        self.jet_constituents = None
        self.energy_flow_bases = None

        hlf_dict = {}
        particle_dict = {}

        self.save_constituents = save_constituents
        self.efbn = energyflow_basis_degree

        if self.efbn < 0:
            self.save_eflow = False
            self.efbn = 0
            self.efp_size = 0
        else:
            self.save_eflow = True
            self.log("creating energyflow particle set with degree d <= {0}...".format(self.efbn))
            self.efpset = ef.EFPSet("d<={0}".format(self.efbn), measure='hadr', beta=1.0, normed=True, verbose=True)
            self.efp_size = self.efpset.count()
            self.log("eflow set is size {}".format(self.efp_size))
        # self.selections_abs = np.asarray([sum(self.sizes[:s[0]]) + s[1] for s in self.selections])
        self.log("found {0} selected events, out of a total of {1}".format(sum(map(len, self.selections.values())), self.nEvents))

    def log(
        self,
        msg
    ):
        if isinstance(msg, str):
            for line in msg.split('\n'):
                print(self.LOGMSG + line)
        else:
            print(self.LOGMSG + str(msg))

    def jet_axis2_pt2(
        self,
        jet,
        constituents,
    ):
        sum_weight = 0
        sum_pt = 0
        sum_deta = 0
        sum_dphi = 0
        sum_deta2 = 0
        sum_detadphi = 0 
        sum_dphi2 = 0

        for i,c in enumerate(constituents):
            deta = c.Eta() - jet.Eta()
            dphi = c.DeltaPhi(jet)
            cpt = c.Pt()
            weight = cpt*cpt

            sum_weight += weight
            sum_pt += cpt
            sum_deta += deta*weight
            sum_dphi += dphi*weight
            sum_deta2 += deta*deta*weight
            sum_detadphi += deta*dphi*weight
            sum_dphi2 += dphi*dphi*weight

        a,b,c,ave_deta,ave_dphi,ave_deta2,ave_dphi2=0,0,0,0,0,0,0

        if sum_weight > 0.:
            ave_deta = sum_deta/sum_weight
            ave_dphi = sum_dphi/sum_weight
            ave_deta2 = sum_deta2/sum_weight
            ave_dphi2 = sum_dphi2/sum_weight
            a = ave_deta2 - ave_deta*ave_deta                                                    
            b = ave_dphi2 - ave_dphi*ave_dphi                                                    
            c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi)
        
        delta = np.sqrt(np.abs((a - b)*(a - b) + 4*c*c))
        axis2 = np.sqrt(0.5*(a+b-delta)) if a + b - delta > 0 else 0
        ptD = np.sqrt(sum_weight)/sum_pt if sum_weight > 0 else 0

        return ptD, axis2

    def get_jet_features(
        self,
        jet_raw,
        constituentp4s
    ):
        j = jet_raw.P4()
        nc, nn = map(float, [jet_raw.NCharged, jet_raw.NNeutrals])
        n_total = nc + nn
        jet_cfrac = nc / n_total if n_total > 0 else -1

        ptd, axis2 = self.jet_axis2_pt2(j, constituentp4s)

        return [
            j.Eta(),
            j.Phi(),
            j.Pt(), 
            j.M(),
            jet_cfrac,
            ptd,
            axis2,
            jet_raw.Flavor,
            j.E(),
        ]
    
    def get_event_features(
        self,
        tree
    ):
        assert tree.Jet_size > 1

        met = tree.MissingET[0].MET
        metphi = tree.MissingET[0].Phi
        meteta = tree.MissingET[0].Eta

        Vjj = tree.Jet[0].P4() + tree.Jet[1].P4()
        met_py = met*np.sin(metphi)
        met_px = met*np.cos(metphi)

        Mjj = Vjj.M()
        Mjj2 = Mjj*Mjj
        ptjj = Vjj.Pt()
        ptjj2 = ptjj*ptjj
        ptMet = Vjj.Px()*met_px + Vjj.Py()*met_py

        MT = np.sqrt(Mjj2 + 2.*(np.sqrt(Mjj2 + ptjj2)*met - ptMet))

        # Mjj = Vjj.M()
        # Mjj2 = Mjj*Mjj
        # ptjj = Vjj.Pt()
        # ptjj2 = ptjj*ptjj
        # ptMet = Vjj.Px()*metpx + Vjj.Py()*metpy
        # MT2 = np.sqrt(Mjj2 + 2*(np.sqrt(Mjj2 + ptjj2)*met - ptMet))

        return [
            met,
            meteta,
            metphi,
            MT, 
            Mjj,
        ]

    def get_jet_constituents(
        self,
        constituentp4s
    ):
        ret = -np.ones((len(constituentp4s), len(self.jet_constituent_names)))
        for i,c in enumerate(constituentp4s):
            ret[i,:] = [c.Eta(), c.Phi(), c.Pt(), c.Rapidity(), c.E()]
        
        return self.pad_to_n(ret, self.n_constituent_particles, 2)

    def get_constituent_p4s(
        self,
        tree,
        jets,
        dr=0.8
    ):
        constituents = [[] for i in range(len(jets))]
        for i,c in enumerate(tree.EFlowTrack): # .1
            if c.PT > 0.1:
                vec = c.P4()
                for j,jet in enumerate(jets):
                    deta = vec.Eta() - jet.Eta()
                    dphi = jet.DeltaPhi(vec)
                    if deta**2. + dphi**2. < dr**2.:
                        constituents[j].append(vec)
            
        for i,c in enumerate(tree.EFlowNeutralHadron): #.5
            if c.ET > 0.5:
                vec = c.P4()
                for j,jet in enumerate(jets):
                    deta = vec.Eta() - jet.Eta()
                    dphi = jet.DeltaPhi(vec)
                    if deta**2. + dphi**2. < dr**2.:
                        constituents[j].append(vec)
            
        for i,c in enumerate(tree.EFlowPhoton): #.2
            if c.ET > 0.2:
                vec = c.P4()
                for j,jet in enumerate(jets):
                    deta = vec.Eta() - jet.Eta()
                    dphi = jet.DeltaPhi(vec)
                    if deta**2. + dphi**2. < dr**2.:
                        constituents[j].append(vec)
                        
        return map(np.asarray, constituents)

    def pad_to_n(
        self,
        data,
        n,
        sort_index
    ):    
        data = data[np.argsort(data[:,sort_index]),:][::-1][:n,:]
        data = np.pad(data, ((0,n-data.shape[0]), (0,0)), 'constant')
        return data

    def get_eflow_variables(
        self, 
        constituents,
    ):
        return self.efpset.compute(
            ef.utils.ptyphims_from_p4s(
                map(
                    lambda c: (c.E(), c.Px(), c.Py(), c.Pz()),
                    constituents
                )
            )
        )

    def convert(
        self,
        rng=(-1,-1),
        return_debug_tree=False
    ):
        rng = list(rng)

        gmin, gmax = min(self.sizes), max(self.sizes)

        if rng[0] < 0 or rng[0] > gmax:
            rng[0] = 0

        if rng[1] > gmax or rng[1] < 0:
            rng[1] = gmax

        nmin, nmax = rng
        selections_iter = self.selections.copy()

        for k,v in selections_iter.items():
            v = np.asarray(v).astype(int)
            selections_iter[k] = v[(v > nmin) & (v < nmax)]

        total_size = sum(map(len, selections_iter.values()))
        total_count = 0

        self.log("selecting on range {0}".format(rng))
        self.event_features = np.empty((total_size, len(self.event_feature_names)))
        self.log("event feature shapes: {}".format(self.event_features.shape))

        self.jet_features = np.empty((total_size, self.n_jets, len(self.jet_feature_names)))
        self.log("jet feature shapes: {}".format(self.event_features.shape))

        self.jet_constituents = np.empty((total_size, self.n_jets, self.n_constituent_particles, len(self.jet_constituent_names)))
        self.log("jet constituent shapes: {}".format(self.jet_constituents.shape))

        self.energy_flow_bases = np.empty((total_size, self.n_jets, self.efp_size))
        self.log("eflow bases shapes: {}".format(self.energy_flow_bases.shape))
            
        if not self.save_constituents:
            self.log("ignoring jet constituents")
        
        # self.log("selections: {}".format(selections_iter))

        ftn = 0

        # selection is implicit: looping only through total selectinos
        for tree_n,tree_name in enumerate(self.inputfiles):

            for event_n,event_index in enumerate(selections_iter[tree_name]):

                self.log('tree {0}, event {1}, index {2}, total count {3}'.format(tree_n, event_n, event_index, total_count))

                # tree (and, get the entry)
                tree = self.trees[tree_n]
                tree.GetEntry(event_index)

                # jets
                jets_raw = [tree.Jet[i] for i in range(min([self.n_jets, tree.Jet_size]))]
                jets = [j.P4() for j in jets_raw]
               
                # constituent 4-vectors per jet
                constituents_by_jet = self.get_constituent_p4s(tree, jets, self.jetDR)

                self.event_features[total_count, :] = np.asarray(self.get_event_features(tree))

                for jet_n, (jet_raw, jet_p4, constituents) in enumerate(zip(jets_raw, jets, constituents_by_jet)):
                
                    self.jet_features[total_count, jet_n, :] =  self.get_jet_features(jet_raw, constituents)
                    
                    if self.save_constituents:
                        self.jet_constituents[total_count, jet_n, :] = self.get_jet_constituents(constituents)
                    
                    if self.save_eflow:
                        self.energy_flow_bases[total_count, jet_n, :] = self.get_eflow_variables(constituents)

                total_count += 1 
        
        return None

    def save(
        self,
        outputfile=None,
    ):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        outputfile = outputfile or os.path.join(self.outputdir, "{}_data.h5".format(self.name))

        if not outputfile.endswith(".h5"):
            outputfile += ".h5"

        self.log("saving h5 data to file {0}".format(outputfile))

        f = h5py.File(outputfile, "w")
        features = f.create_group("event_features")
        assert self.event_features.shape[-1] == len(self.event_feature_names)
        self.log("creating feature 'event_features'")
        features.create_dataset('data', data=self.event_features)
        features.create_dataset('labels', data=self.event_feature_names)

        jet_features = f.create_group("jet_features")
        assert self.jet_features.shape[-1] == len(self.jet_feature_names)
        self.log("creating feature 'jet_features'")
        jet_features.create_dataset('data', data=self.jet_features)
        jet_features.create_dataset('labels', data=self.jet_feature_names)

        if self.save_constituents:
            jet_constituents = f.create_group("jet_constituents")
            assert self.jet_constituents.shape[-2] == self.n_constituent_particles
            assert self.jet_constituents.shape[-1] == len(self.jet_constituent_names)
            self.log("creating feature 'jet_constituents'")
            jet_constituents.create_dataset('data', data=self.jet_constituents)
            jet_constituents.create_dataset('labels', data=self.jet_constituent_names)

        if self.save_eflow:
            eflow = f.create_group("jet_eflow_variables")           
            assert self.energy_flow_bases.shape[-1] == self.efp_size
            self.log("creating feature 'jet_eflow_variables'")
            eflow.create_dataset('data', data=self.energy_flow_bases)
            eflow.create_dataset('labels', data=[str(i) for i in range(self.efp_size)])


        self.log("Successfully saved!")
        f.close()
 
# if __name__ == "__main__":
#     if len(sys.argv) == 10:
#         (_, outputdir, pathspec, name, dr, nc, rmin, rmax, constituents, basis_n) = sys.argv
#         # print outputdir
#         # print pathspec
#         # print filespecls 
#         core = Converter(outputdir, pathspec, name, float(dr), int(nc), bool(int(constituents)), int(basis_n))
#         ret = core.convert((int(rmin), int(rmax)))
#         core.save()
#     # elif len(sys.argv) == 0:
#     else:
#         print "TEST MODE"

#         core = Converter(".", '/afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/data/background/process/data_0_selection.txt', "data", save_constituents=True, energyflow_basis_degree=-1, n_constituent_particles=100)
#         ret = core.convert((0,100), return_debug_tree=False)
#         core.save("TEST2.h5")