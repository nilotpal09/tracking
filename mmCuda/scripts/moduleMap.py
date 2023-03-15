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

    '''
    Create mapping from module hash to integer
    '''
    # set of modules
    s1, s2, s3 = set(df['module1']), set(df['module2']), set(df['module3'])
    module_ids = s1.union(s2).union(s3)

    # dict (module hash -> integer)
    hash2int_dict = {hash_i: i for i, hash_i in enumerate(module_ids)}
    #int2hash_dict = {i: hash_i for hash_i, i in hash2int_dict.items()}

    # save it for later usage
    #with open('transformed_data/module_map/hash2int_dict.pkl', 'wb') as f:
    #    pickle.dump(hash2int_dict, f)

    # update the module indices
    df['module1'] = df['module1'].map(hash2int_dict) 
    df['module2'] = df['module2'].map(hash2int_dict) 
    df['module3'] = df['module3'].map(hash2int_dict)

    '''
    Create unique doublets
    '''
    p1 = set(zip(df['module1'], df['module2']))
    p2 = set(zip(df['module2'], df['module3']))

    dict_names = ['module1', 'module2', 'z0min', 'z0max', 'dphimin', 'dphimax', 'phiSlopemin', 'phiSlopemax', 'detamin', 'detamax']
    # Unique module pairs
    module_pairs = p1.union(p2)
    # Sort
    tup = zip(*list(sorted(list(module_pairs))))
    module_pairs_mapped = {name: l for name, l in zip(dict_names, tup)}
    module_pairs_mapped.update({name: [] for name in dict_names[2:]})

    #for i1, i2 in zip(module_pairs_mapped['module1'], module_pairs_mapped['module2']):
    #    df_unique = df[((df['module1'] == i1) & (df['module2'] == i2)) | ((df['module2'] == i1) & (df['module3'] == i2))]
    #    module_pairs_mapped['z0min'].append(df_unique['z0min_12'].min())
    #    module_pairs_mapped['z0max'].append(df_unique['z0max_12'].max())
    #    module_pairs_mapped['dphimin'].append(df_unique['dphimin_12'].min())
    #    module_pairs_mapped['dphimax'].append(df_unique['dphimax_12'].max())
    #    module_pairs_mapped['phiSlopemin'].append(df_unique['phiSlopemin_12'].min())
    #    module_pairs_mapped['phiSlopemax'].append(df_unique['phiSlopemax_12'].max())
    #    module_pairs_mapped['detamin'].append(df_unique['detamin_12'].min())
    #    module_pairs_mapped['detamax'].append(df_unique['detamax_12'].max())


    # save it for later usage
    #with open('transformed_data/module_map/hash2int_dict.pkl', 'wb') as f:
    #    pickle.dump(hash2int_dict, f)

    
    # update the module indices
    df['module1'] = df['module1'].map(hash2int_dict) 
    df['module2'] = df['module2'].map(hash2int_dict) 
    df['module3'] = df['module3'].map(hash2int_dict)

    pairs = list(zip(module_pairs_mapped['module1'], module_pairs_mapped['module2']))

    df['pair_a'] = list(pairs.index(p) for p in zip(df['module1'], df['module2']))
    df['pair_b'] = list(pairs.index(p) for p in zip(df['module2'], df['module3']))
 
    # save the new module map
    new_mm_path = f'/srv01/agrp/shieldse/storage/ML/trackingData/transformed_data/module_map/{map_name}.csv'
    df.to_csv(new_mm_path, sep=" ", index=False, header=False)
    
    # df_pairs = pd.DataFrame(module_pairs_mapped)
    # mm_pairs_path = f'/srv01/agrp/shieldse/storage/ML/trackingData/transformed_data/module_map/{map_name}_pairs.csv'
    # df_pairs.to_csv(mm_pairs_path, sep=" ", index=False, header=False)
    # print('transoformation done!')

    # validation
    '''
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
    '''

convert_mm()