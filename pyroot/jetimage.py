import ROOT
import sys

from math import *

# Check for input arguments
if len(sys.argv)==1:
    print('usage: {} input0.root [input1.root ...]')
    sys.exit(1)
inputs=sys.argv[1:]

# Get rid of stats box
ROOT.gStyle.SetOptStat(0)

# Load Delphes
ROOT.gSystem.Load('libDelphes.so')

# Load input files
t=ROOT.TChain('Delphes')
for inpath in inputs:
    t.AddFile(inpath)

# Loop the loop
h=ROOT.TH2F('himage','',50,-1,1,50,-1,1)
for e in t:
    for fj in e.FatJet:
        # Simple selection
        if fj.PT<500: continue
        if abs(fj.Mass-173)>20: continue

        # Add constituents to the image
        for c in fj.Constituents:
            if type(c) is not ROOT.Tower:
                continue # not a calorimeter thing

            deta=c.Eta-fj.Eta
            dphi=c.Phi-fj.Phi

            h.Fill(deta,dphi, c.ET)
        break; # Leading jet only
#    break
    
# Draw the jet image
c=ROOT.TCanvas()
h.Draw("COL")

h.GetXaxis().SetTitle('#Delta#eta')
h.GetYaxis().SetTitle('#Delta#Phi')

c.SetLogz(True)

c.SaveAs('jetimage.pdf')
