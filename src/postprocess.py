import pickle
import numpy as np
from collections import Counter

from src.utils import partitiondata

def mergedataMult(videodir, files):
    all_embed = []    
    alllabels = []    
    labelsdictall = []
    for i, f in enumerate(files):
        if f.endswith('.pkl'):
            embed_np, finallabels, labelsdict = pickle.load(open(videodir+f,'rb'))
        else: continue

        print('Loading and Merging {}/{}'.format(i,len(files)))
        # embed_np, finallabels, labelsdict = pickle.load(open(videodir+f,'rb'))
        for seglabels, ldict in zip(finallabels, labelsdict):
            tmp = []
            for label in seglabels:
                if label == 'None':
                    tmp.append('None')
                elif label == 'Hard None':
                    tmp.append('Hard None')
                else:
                    lablist =  label.split(',')[0].split(':')
                    lablist[0] = lablist[0].strip()
                    if lablist[0] == 'Category':
                        tmp.append(lablist[1].strip())
                    else: print("Dropping {}".format(lablist[0]))
            
            labset = set(tmp)
            if 'Powermoves' in labset or 'Toprocks' in labset or 'Footwork' in labset:
                if 'Powermoves' in labset: 
                    lab = 'powermove'
                    if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
                    else: all_embed = embed_np
                    alllabels.append(lab)
                    ldict['iq'] = f.split('.')[0]
                    labelsdictall.append(ldict)

                if 'Toprocks' in labset: 
                    lab = 'toprock'
                    if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
                    else: all_embed = embed_np
                    alllabels.append(lab)
                    ldict['iq'] = f.split('.')[0]
                    labelsdictall.append(ldict)

                if 'Footwork' in labset: 
                    lab = 'footwork'
                    if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
                    else: all_embed = embed_np
                    alllabels.append(lab)
                    ldict['iq'] = f.split('.')[0]
                    labelsdictall.append(ldict)

            # if 'None' in labset: 
            #     lab = 'None'
            #     if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
            #     else: all_embed = embed_np
            #     alllabels.append(lab)
            #     ldict['iq'] = f.split('.')[0]
            #     labelsdictall.append(ldict)

            elif 'Hard None' in labset: 
                lab = 'None'
                if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
                else: all_embed = embed_np
                alllabels.append(lab)
                ldict['iq'] = f.split('.')[0]
                labelsdictall.append(ldict)
            # else: 
            #     lab = ','.join(tmp) 
            #     print('Dropping label {}'.format(lab))


    return all_embed, alllabels, labelsdictall

def mergedataALL(videodir, files):

    def finddeeplabel(lablist):
        maxlen = 0
        for labs in lablist:
            if len(labs.split(':'))>maxlen:
                maxlen = len(labs.split(':'))
                bestlab = labs
        return bestlab

    all_embed = []    
    alllabels = []    
    labelsdictall = []
    allathlete = []
    alldescrp = []
    labdetails = []
    for i, f in enumerate(files):
        if f.endswith('.pkl'):
            embed_np, finallabels, labelsdict = pickle.load(open(videodir+f,'rb'))
        else: continue
        cnt = 0
        for i, (seglabels, ldict) in enumerate(zip(finallabels, labelsdict)):
            # print(seglabels)
            tmp = []
            descript = []
            athlete = []

            for label in seglabels:
                labstr = ''
                athstr = ''
                desc = ''

                if label == 'None':
                    tmp.append('None')
                elif label == 'Hard None':
                    tmp.append('Hard None')
                elif label == 'Easy None':
                    tmp.append('Easy None')
                else:
                    lablist =  label.split(',')
                    for labcats in lablist:
                        labtypes = labcats.split(':')
                        if labtypes[0].strip() == 'Category':
                            labstr=labtypes[1].strip()
                        elif labtypes[0].strip() == 'Move':
                            labstr+=':'+labtypes[1].strip()
                        elif labtypes[0].strip() == 'Athlete':
                            athstr = labtypes[1].strip()
                        else:
                            if labtypes[0].strip() == '<New Tag>': 
                                print('Dropped {}'.format(labtypes[0]))
                            
                            elif len(labtypes)==2: desc = labtypes[1].strip()
                            else: 
                                desc += ','+labtypes[0].strip()
                                # print('Appended {}'.format(desc))
                        # elif labtypes[0].strip() == 'Description':
                        #     desc = labtypes[1].strip()
                        # else: print("Dropping {} --> {}".format(lablist, labtypes[0].strip()))

                if len(labstr)>0: tmp.append(labstr)
                descript.append(desc)
                athlete.append(athstr)

            
            
            labfreq = Counter()
            lablevel = dict()
            hasMove = False
            hasNone = False
            lab = ','.join(tmp)
            for labs in tmp:
                if 'Powermoves' in labs: 
                    labfreq['powermove']+=1
                    if 'powermove' in lablevel: lablevel['powermove'].append(labs)
                    else:lablevel['powermove'] = [labs]
                    hasMove = True
                elif 'Toprocks' in labs: 
                    labfreq['toprock']+=1
                    if 'toprock' in lablevel: lablevel['toprock'].append(labs)
                    else:lablevel['toprock'] = [labs]
                    hasMove = True
                elif 'Footwork' in labs: 
                    labfreq['footwork']+=1
                    if 'footwork' in lablevel: lablevel['footwork'].append(labs)
                    else:lablevel['footwork'] = [labs]
                    hasMove = True
                elif 'Hard None' in labs or 'Easy None' in labs: hasNone = True # Only include Hard and Easy None
                
            if hasMove: # Most Freq one
                maxcnt = 0
                for move in labfreq:
                    if labfreq[move]>maxcnt:
                        lab = move
                        tmp = [finddeeplabel(lablevel[move])] # Just the first one!
                        maxcnt = labfreq[move]
            elif hasNone: lab = 'Hard None'

            if len(tmp) == 0: lab = 'None'

            alllabels.append(lab)
            labdetails.append(tmp)
            allathlete.append(athlete)
            alldescrp.append(descript)
            
            ldict['iq'] = f.split('.')[0]
            labelsdictall.append(ldict)

        if len(all_embed): all_embed = np.vstack((all_embed, embed_np))
        else: all_embed = embed_np
 
    return all_embed, alllabels, labelsdictall, labdetails, allathlete, alldescrp

def filterdata(all_x, all_y, all_dict, all_level_y, all_ath, all_desc):
    filt_x, filt_y, filt_ydict, filt_level_y, filt_ath, filt_desc =[], [], [], [], [], [] 
    for x, y, ydict, y_level, y_ath, y_desc in zip(all_x, all_y, all_dict, all_level_y, all_ath, all_desc):
        if y in ['Hard None','footwork', 'toprock','powermove']:
            filt_y.append(y)
            if len(filt_x): filt_x = np.vstack((filt_x, x))
            else: filt_x = x
            filt_ydict.append(ydict)
            filt_level_y.append(y_level)
            filt_ath.append(y_ath)
            filt_desc.append(y_desc)
    return filt_x, filt_y, filt_ydict, filt_level_y, filt_ath, filt_desc


def partition(filt_x_br, filt_y_br, filt_ydict_br, filt_level_y_br, filt_ath_br, filt_desc_br):
    labelmap = {'None': 0,  'powermove': 1, 'footwork' : 2, 'toprock': 3}
    filterlabels = list(labelmap.keys())
    altlabels_br = [[l[0] for l in filt_level_y_br]]
    y_br = [labelmap[v] for v in filt_y_br]

    trainX_br,trainy_br,trainlabel_br,testX_br,testy_br,testlabel_br,trainaltlabel_br,testaltlabel_br = partitiondata(filt_x_br, y_br, filt_y_br, altlabels_br, seed = 42, ratio = 0.8, filterlabels = filterlabels)
    trainaltlabel_br = [trainaltlabel_br[0].tolist()]
    testaltlabel_br = [testaltlabel_br[0].tolist()]
    trainlabel_br = trainlabel_br.tolist()
    testlabel_br = testlabel_br.tolist()
    trainy_br = trainy_br.tolist()
    testy_br = testy_br.tolist()

    return (trainX_br,trainy_br,trainlabel_br,testX_br,testy_br,testlabel_br,trainaltlabel_br,testaltlabel_br)


#trainX_br,trainy_br,trainlabel_br,testX_br,testy_br,testlabel_br,trainaltlabel_br,testaltlabel_br = partitiondata(filt_x_br, y_br, filt_y_br, altlabels_br, seed = seed, ratio = 0.8, filterlabels = filterlabels)
#trainaltlabel_br = [trainaltlabel_br[0].tolist()]
#testaltlabel_br = [testaltlabel_br[0].tolist()]
#trainlabel_br = trainlabel_br.tolist()
#testlabel_br = testlabel_br.tolist()
#trainy_br = trainy_br.tolist()
#testy_br = testy_br.tolist()
#
#
#alldata = (trainX_br,trainy_br,trainlabel_br,testX_br,testy_br,testlabel_br,trainaltlabel_br,testaltlabel_br)
#pickle.dump(alldata,open('test_{}.pkl'.format(seed),'wb'))

def merge_datasets(*datasets):
    """
    Merge multiple datasets into a single combined dataset.
    
    Args:
        *datasets: Variable number of dataset tuples, each containing:
            (trainX, trainy, trainlabel, testX, testy, testlabel, trainaltlabel, testaltlabel)
    
    Returns:
        tuple: Combined dataset (trainX, trainy, trainlabel, testX, testy, testlabel, trainaltlabel, testaltlabel)
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset required")
    
    trainX_list = []
    trainy_list = []
    trainlabel_list = []
    testX_list = []
    testy_list = []
    testlabel_list = []
    trainaltlabel_list = []
    testaltlabel_list = []
    
    for dataset in datasets:
        trainX, trainy, trainlabel, testX, testy, testlabel, trainaltlabel, testaltlabel = dataset
        
        trainX_list.append(trainX)
        trainy_list.extend(trainy)
        trainlabel_list.extend(trainlabel)
        
        testX_list.append(testX)
        testy_list.extend(testy)
        testlabel_list.extend(testlabel)
        
        trainaltlabel_list.append(trainaltlabel[0])
        testaltlabel_list.append(testaltlabel[0])
    
    # Merge numpy arrays and lists
    merged_trainX = np.vstack(trainX_list)
    merged_testX = np.vstack(testX_list)
    
    merged_trainaltlabel = [sum(trainaltlabel_list, [])]
    merged_testaltlabel = [sum(testaltlabel_list, [])]
    
    return (merged_trainX, trainy_list, trainlabel_list, 
            merged_testX, testy_list, testlabel_list, 
            merged_trainaltlabel, merged_testaltlabel)