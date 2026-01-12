from collections import Counter


import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch

from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset


class BreakdanceDataLoader(Dataset):

    def __init__(self, X, y, labels, altlabels = [], device = torch.device('cpu')):
        
        self.device = device
        # print('Transforming the Data')
        self.X, self.y, self.labels = X, y, labels
        self.altlabels = altlabels
        # self.__transformlabels__(self.y)

    def getCostRatios(self):
        ratios  = dict()
        ratios = Counter(self.y)
        return ratios

    # def __transformlabels__(self,y):
    #     # Relabel them
    #     labelmap = {}
    #     label_vals = sorted(set(self.y))
    #     for i, l in enumerate(label_vals):
    #         labelmap[l] = i
        
    #     for i in range(len(self.y)):
    #         self.y[i] = labelmap[self.y[i]]
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        altlabels_idx = []
        for altlabel in self.altlabels:
            altlabels_idx.append(altlabel[idx])

        return self.X[idx], self.y[idx], self.labels[idx], altlabels_idx





class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    # Modified from : https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
    """
    def __init__(self, dataset, n_classes, n_samples, class_id = None, allSamples = False):
        loader = torch.utils.data.DataLoader(dataset)
        self.labels_list = []
        
        for _,label,_,_ in loader:
            self.labels_list.append(label.to(dtype=torch.long))
        
        
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        
        
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        self.class_id = class_id
        self.allSamples = allSamples

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            
            if self.class_id is not None:
                classes = self.class_id
            else:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            
            indices = []
            for class_ in classes:
                if not self.allSamples: 
                    indices.extend(self.label_to_indices[class_][
                                   self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                        class_] + self.n_samples])
                else: 
                    indices.extend(self.label_to_indices[class_])
                
                self.used_label_indices_count[class_] += self.n_samples
                
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                    
            yield indices
            
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size



def getContrastdata(X_pos, alllabels_pos, X_neg, alllabels_neg, n_samp = 50, pos_labels = ['Power:Windmill']): # This is pairwise
    # Get all the positive ones!!

    X_all = torch.cat((X_pos, X_neg),0)
    all_labels = alllabels_pos+alllabels_neg
    
    pos_ind = []
    neg_ind = []
    for i,label in enumerate(all_labels):
        if label in pos_labels: pos_ind.append(i)
        else: neg_ind.append(i)

    X_pivot = []
    X_ref = []
    rating = []
    label_cont = []
    # Construct Positive Pairs
    for i in range(n_samp):
        pivot_samp = pos_ind[np.random.randint(len(pos_ind), size=1)[0]]
        pos_samp = pos_ind[np.random.randint(len(pos_ind), size=1)[0]]
        neg_samp =  neg_ind[np.random.randint(len(neg_ind), size=1)[0]]

        label_cont += [[all_labels[pivot_samp], all_labels[pos_samp]],
                        [all_labels[pivot_samp], all_labels[neg_samp]]]

        x_p = X_all[pivot_samp].reshape(1,-1)
        x_plus = X_all[pos_samp].reshape(1,-1)
        x_min = X_all[neg_samp].reshape(1,-1)


        if len(X_pivot):X_pivot = torch.cat((X_pivot,x_p, x_p),0)
        else: X_pivot = torch.cat((x_p, x_p),0)

        if len(X_ref): X_ref = torch.cat((X_ref,x_plus, x_min),0)
        else: X_ref = torch.cat((x_plus, x_min),0)

        # print('{}, Pivot = {}, Ref = {}'.format(i, X_pivot.shape,X_ref.shape ))
        rating+=[+1., -1.]

    return X_pivot, X_ref, rating, label_cont


def getMCContrastdata(X_pos, alllabels_pos, X_neg, alllabels_neg, n_samp = 25, pos_labels = ['Power:Windmill:Munchmill/Babymill','Power:Windmill:Barrel Windmill', 'Power:Windmill:Bellymill', 'Power:Windmill:Windmill']): # This is pairwise
    # nsamp per class
    if len(alllabels_neg):
        X_all = torch.cat((X_pos, X_neg),0)
        all_labels = alllabels_pos+alllabels_neg
    
    else: 
        X_all = torch.tensor(X_pos)
        all_labels = alllabels_pos

    pos_ind_list = [[] for _ in pos_labels]
    neg_ind_list = [[] for _ in pos_labels]

    for i,label in enumerate(all_labels):
        for k, pos_label in enumerate(pos_labels): 

            if label == pos_label: pos_ind_list[k].append(i)
            else: neg_ind_list[k].append(i)


    X_pivot = []
    X_ref = []
    rating = []
    label_cont = []
    # Construct Positive Pairs
    for k, label in  enumerate(pos_labels):
        pos_ind = pos_ind_list[k]
        neg_ind = neg_ind_list[k]
        for i in range(n_samp):
            pivot_samp = pos_ind[np.random.randint(len(pos_ind), size=1)[0]]
            pos_samp = pos_ind[np.random.randint(len(pos_ind), size=1)[0]]
            neg_samp =  neg_ind[np.random.randint(len(neg_ind), size=1)[0]]

            label_cont += [[all_labels[pivot_samp], all_labels[pos_samp]],
                            [all_labels[pivot_samp], all_labels[neg_samp]]]

            x_p = X_all[pivot_samp].reshape(1,-1)
            x_plus = X_all[pos_samp].reshape(1,-1)
            x_min = X_all[neg_samp].reshape(1,-1)


            if len(X_pivot):X_pivot = torch.cat((X_pivot,x_p, x_p),0)
            else: X_pivot = torch.cat((x_p, x_p),0)

            if len(X_ref): X_ref = torch.cat((X_ref,x_plus, x_min),0)
            else: X_ref = torch.cat((x_plus, x_min),0)

            # print('{}, Pivot = {}, Ref = {}'.format(i, X_pivot.shape,X_ref.shape ))
            rating+=[+1., -1.]

    return X_pivot, X_ref, rating, label_cont


def getlevelLabels(all_labels,level = -1): # level = [-1, 0, 1, 2]
    levelKlabel = []
    for label in all_labels:
        if level>=0:
            labellist = label.split(':')
            if len(labellist)>level:
                levstr = ':'.join(labellist[0:level+1])
                levelKlabel.append(levstr)
            else:
                levelKlabel.append('Other')
        else:
            if label in ['None','Easy None','Hard None']: levelKlabel.append('None')
            else:levelKlabel.append('Other')
    return levelKlabel



def getMCContrastdata_ver2(X_all, all_labels, n_samp = 25): # This is pairwise
    # nsamp per class
    pos_labels = list(set(all_labels))
    # level = -1 # Special case where we contrast with move vs not

    pos_ind_list = [[] for _ in pos_labels]
    neg_ind_list = [[] for _ in pos_labels]

    for i,label in enumerate(all_labels):
        for k, pos_label in enumerate(pos_labels): 
            if label == pos_label: pos_ind_list[k].append(i)
            else: neg_ind_list[k].append(i)


    X_pivot = []
    X_ref = []
    rating = []
    label_cont = []
    # Construct Positive Pairs
    for i in range(n_samp):
        k = np.random.randint(len(pos_labels), size=1)[0]
        pos_ind = pos_ind_list[k]
        neg_ind = neg_ind_list[k]
    
        pivot_samp = pos_ind[np.random.randint(len(pos_ind), size=1)[0]]
        pos_samp = pos_ind[np.random.randint(len(pos_ind), size=1)[0]]
        neg_samp =  neg_ind[np.random.randint(len(neg_ind), size=1)[0]]

        label_cont += [[all_labels[pivot_samp], all_labels[pos_samp]],
                        [all_labels[pivot_samp], all_labels[neg_samp]]]

        x_p = X_all[pivot_samp].reshape(1,-1)
        x_plus = X_all[pos_samp].reshape(1,-1)
        x_min = X_all[neg_samp].reshape(1,-1)


        if len(X_pivot):X_pivot = torch.cat((X_pivot,x_p, x_p),0)
        else: X_pivot = torch.cat((x_p, x_p),0)

        if len(X_ref): X_ref = torch.cat((X_ref,x_plus, x_min),0)
        else: X_ref = torch.cat((x_plus, x_min),0)

        # print('{}, Pivot = {}, Ref = {}'.format(i, X_pivot.shape,X_ref.shape ))
        rating+=[+1., -1.]

    return X_pivot, X_ref, rating, label_cont