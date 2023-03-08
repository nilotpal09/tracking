import pandas as pd
import pickle
from tabulate import tabulate
import sys
import glob
from tqdm import tqdm
import pdb

def convert_mm():
    mm_path  = '/storage/agrp/nilotpal/tracking/raw_data/df_MMTriplet_3hits_ptCut1GeV_woutSec_woutOC_90kevents_woutElectron.txt'
    map_name = mm_path.split('/')[-1].split('.')[0]

    target_path = '/afs/cern.ch/user/n/nkakati/private/tracking/transformed_data/module_map/'

    column_names = [
		"module1", "module2", "module3", "occurence", "z0max_12", "z0min_12", "dphimax_12", "dphimin_12",
		"phiSlopemax_12", "phiSlopemin_12", "detamax_12", "detamin_12", "z0max_23", "z0min_23", "dphimax_23",
		"dphimin_23", "phiSlopemax_23", "phiSlopemin_23", "detamax_23", "detamin_23", "diff_dzdr_max",
		"diff_dzdr_min","diff_dydx_max", "diff_dydx_min"]


    df = pd.read_csv(mm_path, sep=" ", header=None)
    df.columns = column_names

    # set of modules
    s1, s2, s3 = set(df['module1']), set(df['module2']), set(df['module3'])
    module_ids = s1.union(s2).union(s3)

    # dict (module hash -> integer)
    hash2int_dict = {hash_i: i for i, hash_i in enumerate(module_ids)}

    p1, p2, p3 = (set(tuple(sorted(p)) for p in zip(df['module1'], df['module2'])),
                  set(tuple(sorted(p)) for p in zip(df['module1'], df['module3'])),
                  set(tuple(sorted(p)) for p in zip(df['module2'], df['module3'])))

    # Unique module pairs
    module_pairs = p1.union(p2).union(p3)
    pa, pb = zip(*list(module_pairs))
    pa_mapped, pb_mapped = list(map(hash2int_dict.get, list(pa))), list(map(hash2int_dict.get, list(pb)))
    # Sort
    pa_sorted, pb_sorted = zip(*list(sorted(list(tuple(sorted(p)) for p in zip(pa_mapped, pb_mapped)))))
    module_pairs_mapped = {
        'module1': pa_sorted,
        'module2': pb_sorted
    }

    # save it for later usage
    #with open('transformed_data/module_map/hash2int_dict.pkl', 'wb') as f:
    #    pickle.dump(hash2int_dict, f)

    # update the module indices
    df['module1'] = df['module1'].map(hash2int_dict) 
    df['module2'] = df['module2'].map(hash2int_dict) 
    df['module3'] = df['module3'].map(hash2int_dict)

    pairs = list(zip(pa_sorted, pb_sorted))

    df['pair_a'] = list(pairs.index(tuple(sorted(p))) for p in zip(df['module1'], df['module2']))
    df['pair_b'] = list(pairs.index(tuple(sorted(p))) for p in zip(df['module2'], df['module3']))

    df_pairs = pd.DataFrame(module_pairs_mapped)
 
    # save the new module map
    new_mm_path = f'/srv01/agrp/shieldse/storage/ML/trackingData/transformed_data/module_map/{map_name}.csv'
    df.to_csv(new_mm_path, sep=" ", index=False, header=False)

    mm_pairs_path = f'/srv01/agrp/shieldse/storage/ML/trackingData/transformed_data/module_map/{map_name}_pairs.csv'
    df_pairs.to_csv(mm_pairs_path, sep=" ", index=False, header=False)
    print('transoformation done!')

    # validation
    print('validating...')
    df_new = pd.read_csv(new_mm_path, sep=" ", header=None)
    df_new.columns = column_names

    nan_count = df_new.isnull().sum().sum()
    if nan_count == 0:
        print('\tNo NaN found in the trasformed data')
        print(df_new.head())
        print(tabulate(df_new.head(), headers='keys', tablefmt='psql'))
    else: 
        print(f'Found {nan_count} NaNs in the trasformed data')

convert_mm()