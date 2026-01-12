
import torch
import os
import json
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report

def imagebindembed(vid_dir, labelfile, device = torch.device('cpu')):
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)


    def getlabels(filepaths, label, batch_sze=40): 
        labelist = []
        for fp in filepaths:
            fname = fp.split('/')[-1].split('.')[0]
            if fname not in label:
                print('WARNING: No Label for {}'.format(fname))
                labelist.append(None)
            else: labelist.append(label[fname])
        return labelist

    label = json.load(open(labelfile,'r'))

    # vid_dir = '/home/elv-sauptik/PROJECTS/VideoLLaMA3/assets/redbullpos/'
    files = os.listdir(vid_dir)
    eventfilepaths = []

    for file in files:
        if file.endswith('.mp4'):
            eventfilepaths.append(vid_dir+file)

    # label = json.load(open('/home/elv-sauptik/PROJECTS/VideoLLaMA3/assets/label.json','r'))
    
    all_embed = []
    N = len(eventfilepaths)//batch_sze
    alllabels = []
    for i in range(N+1):
        print('Processing Batch = {}'.format(i))
        eventfilepaths_chunk = eventfilepaths[i*batch_sze:(i+1)*batch_sze]
        alllabels+=getlabels(eventfilepaths_chunk, label)
        event_inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(eventfilepaths_chunk, device),
        }
        with torch.no_grad(): event_embeddings = model(event_inputs)
        embed_np = event_embeddings['vision'].cpu().numpy()
        if len(all_embed):
            all_embed = np.vstack((all_embed, embed_np))
        else: all_embed = embed_np
    assert len(alllabels) == len(all_embed)
    return all_embed, alllabels





def plothist(labels):
    cnts = Counter(labels)
    categories=[]
    values = []
    for k in cnts:
        categories.append(k)
        values.append(cnts[k])

    # Creating a bar plot
    plt.bar(categories, values, color='skyblue', width=0.6)

    # Adding labels and title
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Plot Example')
    plt.xticks(rotation=45, ha='right')

    # Displaying the plot
    plt.show()


def filterLabelData(Xpos, labpos, Xnone, minsamples=1):

    cnts = Counter(labpos)

    labelset = set()
    for k in cnts:
        if cnts[k]>=minsamples:
            labelset.add(k)


    label2class = dict()
    class2label = dict()

    for i, label in enumerate(set(labpos)):
        label2class[label] = i+1  # All class of Interest labeled from 1 .. L
        class2label[i+1] = label
    label2class['None'] = 0
    class2label[0] = 'None'

    y = []
    Xsel = []
    label_sel = []
    for i,label in enumerate(labpos):
        if label in labelset:
            y.append(label2class[label])
            label_sel.append(label)
            if len(Xsel):
                Xsel = np.concatenate((Xsel,Xpos[i].reshape(1,-1)),axis=0)
            else: 
                Xsel = Xpos[i].reshape(1,-1)

    target_names = []
    for i in range(len(set(labels))+1):
        target_names.append(class2label[i])

    y+=[0 for _ in range(len(Xnone))]
    label_sel+=['None' for _ in range(len(Xnone))]
    X_all = np.concatenate((Xsel, Xnone))
    y_all = np.array(y)
    return X_all, y_all, label_sel



def buildData(labels):
    # level 1
    label1 = []
    label2 = []
    label3 = []

    for label in labels:
        if label == 'None':
            label1.append('None')
            label2.append('None')
            label3.append('None')
            continue
        alllevels = label.split(':')
        label1.append(alllevels[0])
        if len(alllevels)>1:
            label2.append(":".join(alllevels[0:2]))
        else:
            label2.append('Other')
        if len(alllevels)>2:
            label3.append(":".join(alllevels[0:3]))
        else:
            label3.append('Other')
    
    return label1, label2, label3


def partitiondata(X,y,labels, altlabels = [], ratio = 0.8, seed = 0, verbose = False, filterlabels = []):
    label_cat = set(labels)
    trn_ind = []
    tst_ind= []
    np.random.seed(seed)
    for label in label_cat:
        if label in filterlabels or len(filterlabels) == 0:
            lab_ind =  [i for i, j in enumerate(labels) if j == label]
            lab_ind = list(np.random.permutation(lab_ind))
            trnN = int(ratio*len(lab_ind))

            trn_ind+= lab_ind[:trnN]
            tst_ind+=lab_ind[trnN:]
            if verbose:print("{} --> Total = {}, Trn = {}, Test = {}".format(label,len(lab_ind), len(lab_ind[:trnN]), len(lab_ind[trnN:])))
            
            trainX, testX = np.take(X, trn_ind, axis=0), np.take(X, tst_ind, axis=0)
            trainy, testy = np.take(y, trn_ind, axis=0), np.take(y, tst_ind, axis=0)
            trainlabel, testlabel = np.take(labels, trn_ind, axis=0), np.take(labels, tst_ind, axis=0)

            trainaltlabels, testaltlabels = [], []
            for altlabel in altlabels:
                trainaltlabel, testaltlabel = np.take(altlabel, trn_ind, axis=0), np.take(altlabel, tst_ind, axis=0)
                trainaltlabels.append(trainaltlabel)
                testaltlabels.append(testaltlabel)

    return trainX, trainy, trainlabel,testX, testy, testlabel, trainaltlabels, testaltlabels

def relabel(filterlabels, label):
    # Will relabel the label based on the filterlabels!
    labelmap = dict()
    for i, lab in enumerate(filterlabels):
        labelmap[lab] = i
    y = []
    for lab in label:
        y.append(labelmap[lab])
    return y,labelmap



def classMetrics(y_true, y_pred, label):
    acc = balanced_accuracy_score(y_true, y_pred)#accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=label)
    return acc, report