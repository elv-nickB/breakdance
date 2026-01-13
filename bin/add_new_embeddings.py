import argparse
import pickle
import os

from src.postprocess import *

"""
This script merges several operations into one - given a directory with content-level embedding files,
it merges them with a given dataset pickle file for compatibility with Sauptik's training pipeline.

The dataset used before Sauptik left is saved under 'original.pkl'. This script merges new embeddings into that dataset.
"""

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--original', type=str, default='original.pkl')
    parser.add_argument('--input', type=str, help='Path to content level embedding files')
    parser.add_argument('--output', type=str, help='Output file', default='embeddings.pkl')
    args = parser.parse_args()

    qids = os.listdir(args.input)

    # added "/" cause original function just merges the strings and I don't want to change it
    data = mergedataALL(args.input + "/", qids)
    data = filterdata(*data)

    # rename 'Hard None' to 'None'
    mod = [x if x != 'Hard None' else 'None' for x in data[1]]

    data = (data[0], mod, data[2], data[3], data[4], data[5])
    data = partition(*data)

    with open(args.original,'rb') as f:
        original = pickle.load(f)

    print(f'Original data train shape: {original[0].shape}')

    new = merge_datasets(original, data)

    print(f'New data train shape: {new[0].shape}')

    with open(args.output,'wb') as f:
        pickle.dump(new, f)

if __name__ == '__main__':
    main()