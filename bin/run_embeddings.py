import argparse
import os
import json
import pickle
from src.embedding import getembedpartpaths_mod
from imagebind import imagebind_model

VID_DIR = 'vids'
TAG_DIR = 'tags'

def main():
    parser = argparse.ArgumentParser(description = 'Label and Embed')
    parser.add_argument('-iq', type = str, help = "This is the IQ for which we need embedding and labeling done!")
    parser.add_argument('-iqlist',default = '', type = str, help = "The list of iqs as json")
    parser.add_argument('-verbose', default = 1, type = int)
    parser.add_argument('--max_parts', default = None, type = int)
    
    args = parser.parse_args()
    
    iq = args.iq

    if len(args.iqlist) == 0:
        iqlist = [iq]
    else:
        iqlist = json.load(open(args.iqlist,'r'))
        print('Total {} files'.format(len(iqlist)))

    print('Loading Model')
    device = 'cuda'
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    for iq in iqlist:
        vid_dir = '{}/{}'.format(VID_DIR, iq)
        print('Loading Videos from {}'.format(vid_dir))
        files = sorted(os.listdir(vid_dir))
        eventfilepaths = []

        if args.max_parts:
            files = files[:args.max_parts]

        for file in files: eventfilepaths.append(os.path.join(vid_dir, file))

        all_tags = json.load(open(f'{TAG_DIR}/{iq}.json','r')) 

        all_embed, all_labels, all_labels_dict = getembedpartpaths_mod(eventfilepaths, all_tags, model = model, verbose = args.verbose, fast = True)
        allresults = (all_embed, all_labels, all_labels_dict)
        pickle.dump(allresults, open('embeds/{}.pkl'.format(iq),'wb'))

if __name__ == '__main__':
    main()