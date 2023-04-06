import os
import pandas as pd
import pubchempy as pcp

path = r'./'
cids = pd.read_csv(os.path.join(path, 'deepcarc_compounds.csv'), dtype=str, usecols=['CID']).CID.to_list()
for cid in cids:
    sdfpath = os.path.join(path, 'SDFs', "{}.sdf".format(cid))
    try:
        pcp.download('SDF', sdfpath, overwrite=True, identifier=cid, record_type='3d')
    except pcp.NotFoundError as e:
        print("No 3d Conformer for {}.".format(cid))
