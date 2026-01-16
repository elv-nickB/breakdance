from functools import partial
import os
from pathlib import Path
import torch
import torch.nn as nn
from ray import init
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import argparse

from src.model import MLP
from src.utils import partitiondata
from src.training import BinHingeTrainer_wRAY
import time
import numpy as np

def load_data(data_dir = './data', datafile = None, seed = 10000, ratio = 0.7):
    with open(os.path.join(data_dir, datafile),'rb') as f:
        return pickle.load(f)


def raytrain(config, data_dir=None, datafile = None, C_test_ratio = 1.0, seed = 10000):
 
    net = MLP(input_dim=1024,
                 num_output=4, 
                 n_feat_hidden_layers = int(config['n_feat_hidden_layers']),
                 n_class_hidden_layers = int(config['n_class_hidden_layers']), 
                 feature_scale = int(config['feature_scale']), 
                 class_scale = int(config['class_scale']), 
                 hidden_resnet=False)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    labelmap = {'None': 0,  'powermove': 1, 'footwork' : 2, 'toprock': 3}

    trainX,trainy,trainlabel,testX,testy,testlabel,trainaltables, _ = load_data(data_dir=data_dir, datafile=datafile, seed = seed)
    
    trainer = BinHingeTrainer_wRAY(trainX, trainy, trainlabel, alllabelscont = trainaltables,
                            Xval = testX, yval = testy, labelsval = testlabel, 
                            net=net, device = torch.device(device), 
                            labelmap = labelmap)

    net, results = trainer.training(lamda = float(config['lamda']), 
                                    learning_rate = float(config['lr']), 
                                    verbose = 10, 
                                    optimalgo = 'AdamW', 
                                    batch_size = 50, 
                                    scheduler = -1, 
                                    amsgrad = False, 
                                    smooth=False, 
                                    n_epochs = int(config['epochs']), 
                                    C_ratio = float(config['C_ratio']), 
                                    Ccont = float(config['Ccont']),
                                    addContrast = True, 
                                    multi_level = True, 
                                    batch_size_c = 100, C_test_ratio = C_test_ratio)


def raytune(num_samples=10, max_num_epochs=1000, cpu_parallel = 5, gpus_per_trial=2, data_dir = '/home/elv-sauptik/PROJECTS/breakdance/data', C_test_ratio = 1.0, datafile = None, seed = 10000):
    config = {
        "lamda":tune.uniform(1e-8, 1e-2),
        "C_ratio": tune.loguniform(1e-3, 1e2),
        "Ccont": tune.uniform(1e-6, 5.),
        "lr": tune.loguniform(3e-5, 3e-2),
        "epochs": tune.choice([200, 500, 1000]), #200, 500, 1000
        "n_feat_hidden_layers": tune.choice([1, 2, 3]),
        "n_class_hidden_layers":tune.choice([1, 2, 3]),
        "feature_scale": tune.choice([1, 2, 4]),
        "class_scale": tune.choice([1, 2, 4])
    }

    scheduler = ASHAScheduler(
        metric="max_val_acc",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(raytrain, data_dir=data_dir, datafile = datafile, C_test_ratio = C_test_ratio, seed = seed),
        resources_per_trial={"cpu": cpu_parallel, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        raise_on_failed_trial=False,
        storage_path='/home/elv-nickb/ray_results2',
        # run_config=RunConfig(log_to_file=("my_stdout.log", "my_stderr.log")),
    )

    best_trial = result.get_best_trial("max_val_acc", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['max_val_acc']}")

    return result, best_trial


def costsensitive_score(tstlabels, scores, C_test_ratio = 1.):
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        p = 0
        n = 0
        r = C_test_ratio
        for gt,pred in zip(tstlabels, scores):
            if gt ==1.:
                p+=1
                if gt == pred: tp+=1
                else: fn+=1
            else:
                n+=1
                if gt == pred: tn+=1
                else: fp+=1
        
        Ptp = tp/p 
        Ptn = tn/n
        Pfp = fp/n 
        Pfn = fn/p
        acc = (r*Ptp+Ptn)/(r+1)
        return acc, Ptp, Ptn, Pfp, Pfn, tp, tn, fp, fn, p, n
    



def modelpred(bestmodel, testX,testy, C_test_ratio = 1.0):
    tstX = torch.tensor(testX)
    scores, _, _ = bestmodel(tstX) 
    scores = scores.detach().cpu().numpy()

    scores[np.where(scores>=0)]= 1.
    scores[np.where(scores<0)]= 0.

    scores = [v[0] for v in scores]

    acc, Ptp, Ptn, Pfp, Pfn, tp, tn, fp, fn, p, n = costsensitive_score(testy, scores, C_test_ratio = C_test_ratio) 
    print(' Bal. Acc = {},\n TP_rate ={} ({}/{}),  FN_rate = {} ({}/{}),\n TN_rate = {}({}/{}),  FP_rate = {} ({}/{})'.format(acc, 
                                                                                                                        Ptp, tp, p,
                                                                                                                        Pfn,fn, p,
                                                                                                                        Ptn,tn, n,
                                                                                                                        Pfp, fp,n))





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Download Tags')
    parser.add_argument('-num_samples',default = 100, type = int, help = "No. of trials") 
    parser.add_argument('-cpu_parallel',default = 10, type = int, help = "No. of CPU") 
    parser.add_argument('-gpus_per_trial',default = 0.1, type = float, help = "No. of GPUs") 
    parser.add_argument('-C_test_ratio',default = 1.0, type = float, help = "Cost ratio test-time") 
    parser.add_argument('-seed',default = 10000, type = int, help = "Data Seed") 
    
    parser.add_argument('-datapath', default = '/home/elv-nickb/breakdance-train/datasets', type = str, help = "The downloaded processed dict.")
    parser.add_argument('-data', default = 'windmilldata_1n2_corrected_wNone.pkl', type = str, help = "The downloaded processed dict.")
    parser.add_argument('-result_file', default = '/home/elv-nickb/ray_results2', type = str, help = "The downloaded processed dict.")
    
    parser.add_argument('-portno', default = 8265, type = int, help = "The UI port")

    args = parser.parse_args()

    result, best_trial = raytune(num_samples=args.num_samples, cpu_parallel = args.cpu_parallel, gpus_per_trial=args.gpus_per_trial, C_test_ratio = args.C_test_ratio, data_dir = args.datapath, datafile = args.data, seed = args.seed)
    best_checkpoint = result.get_best_checkpoint(best_trial, metric="max_val_acc", mode="max")
    config = best_trial.config
    print('The Model Config n_feat_hidden_layers = {}, n_class_hidden_layers = {}, class_scale = {}'.format(config['n_feat_hidden_layers'], config['n_class_hidden_layers'], config['class_scale']))
    print('Checkpoint= {}'.format(best_checkpoint))

    pickle.dump(result,open('./results/{}.pkl'.format(args.result_file),'wb'))
    
    bestmodel = MLP(input_dim=1024,
                 num_output=4, 
                 n_feat_hidden_layers = int(config['n_feat_hidden_layers']),
                 n_class_hidden_layers = int(config['n_class_hidden_layers']), 
                 feature_scale = int(config['feature_scale']), 
                 class_scale = int(config['class_scale']), 
                 hidden_resnet=False)

    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            checkpoint_state = pickle.load(fp)
        bestmodel.load_state_dict(checkpoint_state["net_state_dict"])
        print('saving')
        torch.save(bestmodel.state_dict(), 'output_model.ckpt')
    
    trainX,trainy,trainlabel,testX,testy,testlabel,trainaltables,testaltables = load_data(data_dir=args.datapath, datafile=args.data, seed = args.seed)
    modelpred(bestmodel, trainX,trainy, C_test_ratio = args.C_test_ratio)
    modelpred(bestmodel, testX,testy, C_test_ratio = args.C_test_ratio)

    """
    ###################################################################
    datadir = '/home/elv-sauptik/PROJECTS/ImageBind/tmp/'
    X, y, time_stamps = [],[],[]

    availablefiles = os.listdir(datadir)

    filelist = ['iq__2dGwZjUx5L9Y8AB3AFcLgsDcctz7',
    'iq__2e1LNgn6L22A7JJLrtohRoB1Xcqt',
    'iq__2ezw1MdsoKJzDqd41cPgnhG9Huox',
    'iq__2G322kku9xJU7nJfCyyG8Q32jYUn',
    'iq__3cMv2KNA3348Ba9JBf9n4giBWseU',
    'iq__3eggDxNe72f28KCjHHdUWvKoFmig',
    'iq__3tAoGTC9bAVMhY5ohrh3MPuafd8L',
    'iq__3yCJgTw3fWcXHESTycx35z1HhkLm',
    'iq__49UhnKG6knwcRKzRyPxbGDmrMzEE',
    'iq__5UJDhkbPNBB2rQrh9d9vJbco8AE'
    ]

    for i, fname in enumerate(filelist):
        file = fname+'.pkl' 
        if file in availablefiles:
            print('Opening File {}.{}'.format(i, file))
            all_embed, all_labels, all_labels_dict = pickle.load(open(datadir+file,'rb'))
        
        else:
            print('Missing file {}'.format(file)) 
            continue

        for X_embed, labels, time_stamp in zip(all_embed, all_labels, all_labels_dict):
            # if any(item not in ["None","<New Tag>"] for item in labels):
            if len(X): X = np.vstack((X, X_embed))
            else: X = X_embed

            y.append(labels)
            time_stamp_cpy =  time_stamp.copy()
            time_stamp_cpy['iq'] = file.split('.')[0]
            time_stamps.append(time_stamp_cpy)

    Xtst, tstlabel = X, y 
    ytst = []
    for label in tstlabel:
        labellist = label[0].split(':')
        if len(labellist)>=2 and labellist[1] == 'Windmill': ytst.append(1)
        else: ytst.append(0)
    
    modelpred(bestmodel, Xtst, ytst, C_test_ratio = args.C_test_ratio)
    """