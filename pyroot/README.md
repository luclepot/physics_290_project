# Setup

Analysis of the ntuples requires the presence of PyROOT and Delphes. The latter is needed for some custom objects stored by the simulation program.

To setup a conda envrionment with the necessary packages:
```
conda create -n e290 -c conda-forge root
conda install ${CFS}/atlas/kkrizka/E290/delphes-3.5.0-0.tar.bz2
```

To initialize the environment at the start of every session
```
conda activate e290
```

# Example
To create a "jet image":
```
python jetimages.py ${CFS}/atlas/kkrizka/E290/samples/v0.0.1/ttbar/run_02_*/tag_1_delphes_events.root
```