import pandas as pd
import pickle
import argparse
import pdb

def process_event(event_path, output_path):
    
    f = open('/srv01/agrp/shieldse/storage/ml/trackingData/transformed_data/module_map/hash2int_dict.pkl', 'rb')
    hash2int_dict = pickle.load(f)

    df = pd.read_csv(event_path) #, header=None)
    df['ID'] = df['ID'].map(hash2int_dict)
    
    df.to_csv(output_path, header=None)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='processEvent', description='Map the hit indicies from hashes to integers')
    parser.add_argument('-i', '--input', type=str, default='/storage/agrp/nilotpal/tracking/raw_data/new_data/event000000001-truth.csv')
    parser.add_argument('-o', '--output', type=str, default='/srv01/agrp/shieldse/storage/ml/trackingData/transformed_data/event000000001-truth.csv')
    args = parser.parse_args()

    process_event(args.input, args.output)