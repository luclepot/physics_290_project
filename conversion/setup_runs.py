import glob
import os

pt_cutoff = 500
dir_str = "pt{}/{}_out/files_{}0-{}9/"
with open("dijet_runfiles.sh", "w+") as f:
    for i in range(10):
        this_dir = dir_str.format(pt_cutoff, 'dijet', i, i)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        f.write(
            "python convert.py -i \"../../E290/samples/v0.0.1/dijet/run_01_{}[0-9]/tag_1_delphes_events.root\" -o {} -p {} > {} 2>&1 &\n".format(
                i if i > 0 else '', this_dir, pt_cutoff, os.path.join(this_dir, 'log')
            )
        )

with open("ttbar_runfiles.sh", "w+") as f:
    for i in range(10):
        this_dir = dir_str.format(pt_cutoff, 'ttbar', i, i)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        f.write(
            "python convert.py -i \"../../E290/samples/v0.0.1/ttbar/run_02_{}[0-9]/tag_1_delphes_events.root\" -o {} -p {} > {} 2>&1 &\n".format(
                i if i > 0 else '', this_dir, pt_cutoff, os.path.join(this_dir, 'log')
            )
        )
