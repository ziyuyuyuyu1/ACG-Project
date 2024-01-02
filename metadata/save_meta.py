import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--extension', type=str, default='pt')
    args = parser.parse_args()

    type_name = ['car', 'rifle', 'table', 'chair', 'airplane']

    fpath_list = sorted([os.path.join(os.path.join(args.data_path, t), fname) 
                     for t in type_name 
                     for fname in os.listdir(os.path.join(args.data_path, t)) 
                     if fname.endswith('.' + args.extension)])
    json.dump(fpath_list, open(args.json_path, 'w'))

