from __future__ import print_function
import os
import ROOT as rt
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DELPHES_DIR = os.environ["DELPHES_DIR"]
rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))

tree = rt.TChain("Delphes")
paths = list(map(os.path.abspath, glob.glob('../../E290/samples/v0.0.1/ttbar/*/*.root')))

for path in paths:
    tree.AddFile(path)

# h=rt.TH2F('himage','',50,-1,1,50,-1,1)
data = []
jet_count = 0
for i,e in enumerate(tree):
    if i%1000==0:
        print (str(i) + '/' + str(1000000) ),
    leading = True
    for fj in e.FatJet:
        
        # require jet PT > 500
        if fj.PT < 500:
            continue
        
        # Isolate jets in +/- 20 GeV signal region about 173 
        # if abs(fj.Mass-173)>20:
        #     continue
        
        data_elt = []
        # Add constituents to the image
        for c in fj.Constituents:
            if type(c) is not rt.Tower:
                continue # not a calorimeter thing

            deta=c.Eta-fj.Eta
            dphi=c.Phi-fj.Phi

            data_elt.append([deta, dphi, c.ET,])
            jet_count += 1
        #     h.Fill(deta,dphi, c.ET)
        data.append(data_elt)
        break; # Leading jet only
#    break
    if jet_count > 10:
        break
    
# Draw the jet image
# c=rt.TCanvas()
# h.Draw("COL")

# h.GetXaxis().SetTitle('#Delta#eta')
# h.GetYaxis().SetTitle('#Delta#Phi')

# c.SetLogz(True)

# c.SaveAs('jetimage.pdf')