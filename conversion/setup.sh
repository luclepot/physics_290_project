# FIRST: run..
# shifter --image=zlmarshall/atlas-grid-centos7:20211123 --module=cvmfs /bin/bash
echo "first run 'shifter --image=zlmarshall/atlas-grid-centos7:20211123 --module=cvmfs /bin/bash'"
# source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3_ATLAS_3/x86_64-centos7-gcc8-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh
source $DELPHES_DIR/delphes-env.sh
pip install tqdm pandas --user
