from sklearn.metrics import balanced_accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.data import BreakdanceDataLoader, getMCContrastdata, BalancedBatchSampler, getMCContrastdata_ver2, getlevelLabels
from ray.train import Checkpoint, get_checkpoint
from src.loss import MCHingeLoss, FMContrastiveSim, HingeLoss
import numpy as np
from ray import train
import tempfile
from pathlib import Path
import ray.cloudpickle as pickle
import copy
    
class MCHingeAdvancedTrainerMultLevel_RAY(object):
    def __init__(self,
                 X, y, labels, altlabels, labelmap,# File name or the actual file
                 Xval = None, yval = None, labelsval = None, altlabelsval = [],
                 Xtst = None, ytst = None, labeltst = None, altlabelstst = [],
                 Xadd = None, yadd = None, labeladd = None, altlabelsadd = [],
                 model = None,
                 ratio_within = 1.0,
                 ratio_between = [], 
                #  best_model_fname = 'MC_model_brace',
                 device = torch.device('cpu')): # Assume only HPO no NAS
        

        self.X = X
        self.y = y
        self.labels = labels
        self.K = len(labelmap)
        self.labelmap = labelmap
        self.altlabels = altlabels 
        self.r_w = ratio_within
        self.r_b = ratio_between if len(ratio_between) else [1. for _ in range(self.K)]
        # self.best_model_fname = best_model_fname

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
            self.altlabelsval = altlabelsval
        
        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst
            self.altlabelstst = altlabelstst


        if Xadd is not None:
            self.Xadd = Xadd 
            self.yadd = yadd 
            self.labeladd = labeladd
            self.altlabelsadd = altlabelsadd


        if model is not None: self.net =  model.to(device)
        self.device = device

        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels,self.altlabels)
        ratios = self.train_data.getCostRatios()
        print('Prior Probability Training {}'.format(ratios))

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None
        if Xadd is not None: self.add_data = BreakdanceDataLoader(self.Xadd,self.yadd,self.labeladd, self.altlabelsadd)
        else: self.add_data = None


    def evaluate(self,datloader,testmodel = None): 
        if not testmodel: testmodel = self.net
        testmodel.eval()
        def perclass_bal_costsensitive(y_true, y_pred, r_w = 1.0, r_b = []): # r = Ctp/Ctn
            if len(r_b) == 0: r_b = [1.0 for _ in range(self.K)]
            bal_acc = 0
            for k in range(self.K):
                TP, TN, FP = 0, 0, 0
                preds = y_pred[np.where(y_true==k)]
                pos_pred = y_true[np.where(y_pred==k)]

                TP = len(preds[np.where(preds==k)])
                FN = len(preds[np.where(preds!=k)])
                FP = len(pos_pred[np.where(pos_pred!=k)])
                Prec = TP/(TP+FP) if TP+FP >0 else 0
                Recall = TP/(TP+FN) if TP+FN>0 else 0
                bal_acc += r_b[k]*(r_w*Prec+Recall)/(r_w+1)
            totwt = sum(r_b)
            bal_acc = bal_acc/totwt
            return bal_acc

        
        idx_label_score = []
        self.net.eval()

        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy()
                outputs, _, _ = testmodel(valX)
                outputs = np.argmax(outputs.data.cpu().numpy(), axis = 1)
                idx_label_score += list(zip(valy.tolist(),outputs.tolist(), vallabels))

        y_true, y_pred, labels = zip(*idx_label_score)     # Need original ratings for test!


        bal_acc = balanced_accuracy_score(y_true, y_pred) # accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labelmap)

        labels = np.array(labels)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        class_bal_metric = perclass_bal_costsensitive(y_true, y_pred, r_w = self.r_w, r_b = self.r_b)
    
        return class_bal_metric, bal_acc, report



    # Here we will do 1. Labeled, 2. Transductive, 3. Contrastive

    def training(
        self,
        lamda = 1e-2, 
        optimalgo = 'SGD',
        learning_rate = 5e-5,
        batch_size = 50, 
        batch_contrast = 50, 
        Ccont = 0.0, 
        margin = 0.0, 
        C_ratio = [], 
        n_epochs = 50000, 
        scheduler = -1, 
        amsgrad = False, 
        pos_labels = ['None', 'footwork', 'powermove', 'toprock'],
        verbose = 100, addContrast=True, multi_level = True
    ): 
            
        C_ratio = C_ratio if len(C_ratio) else self.r_b
        print('#######################################')
        print('Training with C_ratio = {}'.format(C_ratio))
        print('#######################################')
        K = self.K
        if optimalgo == 'SGD': optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
        if optimalgo == 'AdamW': optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
        else: optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!
            
        if scheduler>0: 
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        
        
        # Model CheckPoint
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                self.net.load_state_dict(checkpoint_state["net_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0


        balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
        train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
        train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring
        
        if self.val_data is not None: 
            val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
        
        if self.test_data is not None: 
            test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

        if addContrast:
            balanced_batch_sampler_cont = BalancedBatchSampler(self.train_data, K, n_samples = batch_contrast, class_id = [float(x) for x in range(K)])
            cont_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler_cont)

        class_weights = torch.FloatTensor(C_ratio).cuda()
        mcloss = MCHingeLoss()
        # contrast_loss = FMContrastiveSim()  ## Only one level here!! ?? Will this help??
        # contrast_loss = nn.CosineEmbeddingLoss(margin=margin)

        contrast_loss_det = FMContrastiveSim()
        contrast_loss_0 = FMContrastiveSim()
        contrast_loss_1 = FMContrastiveSim()


        max_val_acc = 0
        best_epoch_trn = 0
        best_epoch_val_report= None
        best_epoch_trn_report = None
        best_model = copy.deepcopy(self.net)
        qa_val_acc = 0


        for epoch in range(n_epochs+1):
            loss_epoch = 0.0
            n_batches = 0
            
            for _, data in enumerate(train_loader):
                X, y, labels_pos, alllabels_pos = data
                X = X.to(self.device)
                y = y.to(self.device)
                
                outputs, feat_pos,_ = self.net(X)
                optimizer.zero_grad()
                loss = mcloss(outputs, y, K, C_ratio = class_weights)

                data_cont = next(iter(cont_loader))
                X_cont, _, _, cont_labels = data_cont
                X_cont=X_cont.to(self.device)
                _, feat_cont,_ = self.net(X_cont)


                labels_det = getlevelLabels(cont_labels[0],level = -1)
                labels_0 = getlevelLabels(cont_labels[0],level = 0)
                labels_1 = getlevelLabels(cont_labels[0],level = 1)


                X_pivot_d, X_ref_d, rating_d, label_d  = getMCContrastdata_ver2(feat_cont, labels_det, n_samp = batch_contrast)
                X_pivot_0, X_ref_0, rating_0, label_0  = getMCContrastdata_ver2(feat_cont, labels_0, n_samp = batch_contrast)
                X_pivot_1, X_ref_1, rating_1, label_1  = getMCContrastdata_ver2(feat_cont, labels_1, n_samp = batch_contrast)
            

                rating_d = torch.tensor(rating_d).to(self.device)
                rating_0 = torch.tensor(rating_0).to(self.device)
                rating_1 = torch.tensor(rating_1).to(self.device)
                

                if multi_level:
                    con_loss_det = contrast_loss_det(X_pivot_d, X_ref_d, rating_d, level = 0., maxlevel=3., margin = margin)
                    con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 1., maxlevel=3., margin = margin)
                    con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 2., maxlevel=3., margin = margin)
                    
                else:
                    con_loss_det = contrast_loss_det(X_pivot_d, X_ref_d, rating_d, level = 0., maxlevel=1., margin = margin)
                    con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=1., margin = margin)
                    con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 0., maxlevel=1., margin = margin)  # Single level's
                    
            
                con_loss = con_loss_det+con_loss_0+con_loss_1

                l2_reg = torch.tensor(0.).to(self.device)

                for name, param in self.net.named_parameters(): l2_reg += torch.norm(param)**2
                
                classloss = loss + lamda * l2_reg + Ccont * con_loss   # Equal Weights to different levels!!
                classloss.backward()
                optimizer.step()

                loss_epoch += classloss.item()
                n_batches += 1

            if epoch%verbose==0:
                print('------------------------------------- ')
                print('Epoch = {}'.format(epoch))
                
                trn_class_bal_metric, trn_acc, trn_report = self.evaluate(train_loader_scoring)
                print('Train Loss = {}, Label loss ={}, Cont. Loss = {}(lDet = {},l0 = {}, l1= {})'.format(classloss.item(), 
                                                                                                            loss.item(), 
                                                                                                            con_loss.item(), 
                                                                                                            con_loss_det.item(),
                                                                                                            con_loss_0.item(),
                                                                                                            con_loss_1.item()))
                print('Tr. Bal. Metric = {}, Train Acc = {}'.format(trn_class_bal_metric, trn_acc))
                
                if self.val_data is not None:
                    val_class_bal_metric, val_acc, val_report = self.evaluate(val_loader)
                    if val_acc>=max_val_acc:
                        max_val_acc = val_acc
                        best_epoch = epoch
                        best_epoch_trn = trn_acc
                        best_epoch_val_report = val_report
                        best_epoch_trn_report = trn_report
                        best_model.load_state_dict(self.net.state_dict())
                        qa_val_class_bal_metric, qa_val_acc, qa_val_report = self.evaluate(val_loader, best_model)

                    print('Val Bal. Metric = {}, Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_class_bal_metric, val_acc, best_epoch, max_val_acc, best_epoch_trn))
                    print('QA :: Val Acc = {}'.format(qa_val_acc))
                # if self.test_data is not None:
                #     tst_acc, _ = self.evaluate(test_loader)
                #     print('Test Acc = {}'.format(tst_acc))
                
                checkpoint_data = {
                    "epoch": epoch,
                    "net_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }


            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"train_loss": classloss.item(), 
                    "lab_loss": loss.item(),
                    "cont_loss": con_loss.item() if con_loss else 1000,
                    "val_acc": val_acc,
                    "max_val_acc": max_val_acc,
                    "val_class_bal_metric": val_class_bal_metric},
                    checkpoint=checkpoint,
                )

            # if epoch == n_epochs: # Final Best Result
            #     returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                
            #     if self.val_data is not None:
            #         returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
            #     if self.test_data is not None: # Need to modify
            #         returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
        print('Finished Training') 
        return best_model, self.net
    
class MCEntropyAdvancedTrainer_RAY(object):
    def __init__(self,
                 X, y, labels, altlabels, labelmap,# File name or the actual file
                 Xval = None, yval = None, labelsval = None, altlabelsval = [],
                 Xtst = None, ytst = None, labeltst = None, altlabelstst = [],
                 Xadd = None, yadd = None, labeladd = None, altlabelsadd = [],
                 model = None,
                 ratio_within = 1.0,
                 ratio_between = [], 
                #  best_model_fname = 'MC_model_brace',
                 device = torch.device('cpu')): # Assume only HPO no NAS
        

        self.X = X
        self.y = y
        self.labels = labels
        self.K = len(labelmap)
        self.labelmap = labelmap
        self.altlabels = altlabels 
        self.r_w = ratio_within
        self.r_b = ratio_between if len(ratio_between) else [1. for _ in range(self.K)]
        # self.best_model_fname = best_model_fname

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
            self.altlabelsval = altlabelsval
        
        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst
            self.altlabelstst = altlabelstst


        if Xadd is not None:
            self.Xadd = Xadd 
            self.yadd = yadd 
            self.labeladd = labeladd
            self.altlabelsadd = altlabelsadd


        if model is not None: self.net =  model.to(device)
        self.device = device

        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels,self.altlabels)
        ratios = self.train_data.getCostRatios()
        print('Prior Probability Training {}'.format(ratios))

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None
        if Xadd is not None: self.add_data = BreakdanceDataLoader(self.Xadd,self.yadd,self.labeladd, self.altlabelsadd)
        else: self.add_data = None


    def evaluate(self,datloader,testmodel = None): 
        if not testmodel: testmodel = self.net
        testmodel.eval()
        def perclass_bal_costsensitive(y_true, y_pred, r_w = 1.0, r_b = []): # r = Ctp/Ctn
            if len(r_b) == 0: r_b = [1.0 for _ in range(self.K)]
            bal_acc = 0
            for k in range(self.K):
                TP, TN, FP = 0, 0, 0
                preds = y_pred[np.where(y_true==k)]
                pos_pred = y_true[np.where(y_pred==k)]

                TP = len(preds[np.where(preds==k)])
                FN = len(preds[np.where(preds!=k)])
                FP = len(pos_pred[np.where(pos_pred!=k)])
                Prec = TP/(TP+FP) if TP+FP >0 else 0
                Recall = TP/(TP+FN) if TP+FN>0 else 0
                bal_acc += r_b[k]*(r_w*Prec+Recall)/(r_w+1)
            totwt = sum(r_b)
            bal_acc = bal_acc/totwt
            return bal_acc

        
        idx_label_score = []
        self.net.eval()

        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy()
                outputs, _, _ = testmodel(valX)
                outputs = np.argmax(outputs.data.cpu().numpy(), axis = 1)
                idx_label_score += list(zip(valy.tolist(),outputs.tolist(), vallabels))

        y_true, y_pred, labels = zip(*idx_label_score)     # Need original ratings for test!


        bal_acc = balanced_accuracy_score(y_true, y_pred) # accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labelmap)

        labels = np.array(labels)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        class_bal_metric = perclass_bal_costsensitive(y_true, y_pred, r_w = self.r_w, r_b = self.r_b)
    
        return class_bal_metric, bal_acc, report



    # Here we will do 1. Labeled, 2. Transductive, 3. Contrastive

    def training(self,
                      lamda = 1e-2, 
                      optimalgo = 'SGD',
                      learning_rate = 5e-5,
                      batch_size = 50, 
                      batch_contrast = 50, 
                      Ccont = 0.0, 
                      margin = 0.0, 
                      C_ratio = [], 
                      n_epochs = 50000, 
                      scheduler = -1, 
                      amsgrad = False, 
                      pos_labels = ['None', 'footwork', 'powermove', 'toprock'],
                      verbose = 100): 
            
            C_ratio = C_ratio if len(C_ratio) else self.r_b
            print('#######################################')
            print('Training with C_ratio = {}'.format(C_ratio))
            print('#######################################')
            K = self.K
            if optimalgo == 'SGD': optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
            if optimalgo == 'AdamW': optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
            else: optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!
                
            if scheduler>0: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)
            
            
            # Model CheckPoint
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_epoch = checkpoint_state["epoch"]
                    self.net.load_state_dict(checkpoint_state["net_state_dict"])
                    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            else:
                start_epoch = 0


            balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
            train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
            train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring
            
            
            if self.val_data is not None: 
                val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
            
            if self.test_data is not None: 
                test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

            # if addContrast:
            #     balanced_batch_sampler_cont = BalancedBatchSampler(self.train_data, K, n_samples = batch_size_c, class_id = [float(x) for x in range(K)])
            #     cont_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler_cont)

            class_weights = torch.FloatTensor(C_ratio).cuda()
            mcloss = nn.CrossEntropyLoss(weight=class_weights)
            contrast_loss = FMContrastiveSim()  ## Only one level here!! ?? Will this help??
   

            max_val_acc = 0
            best_epoch_trn = 0
            best_epoch_val_report= None
            best_epoch_trn_report = None
            best_model = copy.deepcopy(self.net)
            qa_val_acc = 0


            for epoch in range(n_epochs+1):
                loss_epoch = 0.0
                n_batches = 0
                
                for _, data in enumerate(train_loader):
                    X, y, labels_pos, alllabels_pos = data
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    # print("Pos labels = {}".format(labels_pos))

                    outputs, feat_pos,_ = self.net(X)
                    optimizer.zero_grad()
                    loss = mcloss(outputs, y)

                    X_pivot_0, X_ref_0, rating_0, label_0  = getMCContrastdata(feat_pos, labels_pos, [], [], n_samp = batch_contrast, pos_labels = pos_labels)

                    rating_0 = torch.tensor(rating_0).to(self.device)
                    con_loss = contrast_loss(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=1., margin = margin)
                    
                    l2_reg = torch.tensor(0.).to(self.device)

                    for name, param in self.net.named_parameters(): l2_reg += torch.norm(param)**2
                    
                    classloss = loss + lamda * l2_reg + Ccont * con_loss   # Equal Weights to different levels!!
                    classloss.backward()
                    optimizer.step()

                    loss_epoch += classloss.item()
                    n_batches += 1

                if epoch%verbose==0:
                    print('------------------------------------- ')
                    print('Epoch = {}'.format(epoch))
                    
                    trn_class_bal_metric, trn_acc, trn_report = self.evaluate(train_loader_scoring)
                    print('Train Loss = {}, Label loss ={}, Cont. Loss = {}'.format(classloss.item(), loss.item(), con_loss.item()))
                    print('Tr. Bal. Metric = {}, Train Acc = {}'.format(trn_class_bal_metric, trn_acc))
                    
                    if self.val_data is not None:
                        val_class_bal_metric, val_acc, val_report = self.evaluate(val_loader)
                        if val_acc>=max_val_acc:
                            max_val_acc = val_acc
                            best_epoch = epoch
                            best_epoch_trn = trn_acc
                            best_epoch_val_report = val_report
                            best_epoch_trn_report = trn_report
                            best_model.load_state_dict(self.net.state_dict())
                            qa_val_class_bal_metric, qa_val_acc, qa_val_report = self.evaluate(val_loader, best_model)

                        print('Val Bal. Metric = {}, Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_class_bal_metric, val_acc, best_epoch, max_val_acc, best_epoch_trn))
                        print('QA :: Val Acc = {}'.format(qa_val_acc))
                    # if self.test_data is not None:
                    #     tst_acc, _ = self.evaluate(test_loader)
                    #     print('Test Acc = {}'.format(tst_acc))
                    
                    checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }

        
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {"train_loss": classloss.item(), 
                        "lab_loss": loss.item(),
                        "cont_loss": con_loss.item() if con_loss else 1000,
                        "val_acc": val_acc,
                        "max_val_acc": max_val_acc,
                        "val_class_bal_metric": val_class_bal_metric},
                        checkpoint=checkpoint,
                    )

                # if epoch == n_epochs: # Final Best Result
                #     returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                    
                #     if self.val_data is not None:
                #         returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
                #     if self.test_data is not None: # Need to modify
                #         returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
            print('Finished Training') 
            return best_model, self.net

class MCHingeAdvancedTrainer_RAY(object):
    def __init__(self,
                 X, y, labels, altlabels, labelmap,# File name or the actual file
                 Xval = None, yval = None, labelsval = None, altlabelsval = [],
                 Xtst = None, ytst = None, labeltst = None, altlabelstst = [],
                 Xadd = None, yadd = None, labeladd = None, altlabelsadd = [],
                 model = None,
                 ratio_within = 1.0,
                 ratio_between = [], 
                #  best_model_fname = 'MC_model_brace',
                 device = torch.device('cpu')): # Assume only HPO no NAS
        

        self.X = X
        self.y = y
        self.labels = labels
        self.K = len(labelmap)
        self.labelmap = labelmap
        self.altlabels = altlabels 
        self.r_w = ratio_within
        self.r_b = ratio_between if len(ratio_between) else [1. for _ in range(self.K)]
        # self.best_model_fname = best_model_fname

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
            self.altlabelsval = altlabelsval
        
        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst
            self.altlabelstst = altlabelstst


        if Xadd is not None:
            self.Xadd = Xadd 
            self.yadd = yadd 
            self.labeladd = labeladd
            self.altlabelsadd = altlabelsadd


        if model is not None: self.net =  model.to(device)
        self.device = device

        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels,self.altlabels)
        ratios = self.train_data.getCostRatios()
        print('Prior Probability Training {}'.format(ratios))

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None
        if Xadd is not None: self.add_data = BreakdanceDataLoader(self.Xadd,self.yadd,self.labeladd, self.altlabelsadd)
        else: self.add_data = None


    def evaluate(self,datloader,testmodel = None): 
        if not testmodel: testmodel = self.net
        testmodel.eval()
        def perclass_bal_costsensitive(y_true, y_pred, r_w = 1.0, r_b = []): # r = Ctp/Ctn
            if len(r_b) == 0: r_b = [1.0 for _ in range(self.K)]
            bal_acc = 0
            for k in range(self.K):
                TP, TN, FP = 0, 0, 0
                preds = y_pred[np.where(y_true==k)]
                pos_pred = y_true[np.where(y_pred==k)]

                TP = len(preds[np.where(preds==k)])
                FN = len(preds[np.where(preds!=k)])
                FP = len(pos_pred[np.where(pos_pred!=k)])
                Prec = TP/(TP+FP) if TP+FP >0 else 0
                Recall = TP/(TP+FN) if TP+FN>0 else 0
                bal_acc += r_b[k]*(r_w*Prec+Recall)/(r_w+1)
            totwt = sum(r_b)
            bal_acc = bal_acc/totwt
            return bal_acc

        
        idx_label_score = []
        self.net.eval()

        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy()
                outputs, _, _ = testmodel(valX)
                outputs = np.argmax(outputs.data.cpu().numpy(), axis = 1)
                idx_label_score += list(zip(valy.tolist(),outputs.tolist(), vallabels))

        y_true, y_pred, labels = zip(*idx_label_score)     # Need original ratings for test!


        bal_acc = balanced_accuracy_score(y_true, y_pred) # accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labelmap)

        labels = np.array(labels)
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        class_bal_metric = perclass_bal_costsensitive(y_true, y_pred, r_w = self.r_w, r_b = self.r_b)
    
        return class_bal_metric, bal_acc, report



    # Here we will do 1. Labeled, 2. Transductive, 3. Contrastive

    def training(self,
                      lamda = 1e-2, 
                      optimalgo = 'SGD',
                      learning_rate = 5e-5,
                      batch_size = 50, 
                      batch_contrast = 50, 
                      Ccont = 0.0, 
                      margin = 0.0, 
                      C_ratio = [], 
                      n_epochs = 50000, 
                      scheduler = -1, 
                      amsgrad = False, 
                      pos_labels = ['None', 'footwork', 'powermove', 'toprock'],
                      verbose = 100): 
            
            C_ratio = C_ratio if len(C_ratio) else self.r_b
            print('#######################################')
            print('Training with C_ratio = {}'.format(C_ratio))
            print('#######################################')
            K = self.K
            if optimalgo == 'SGD': optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
            if optimalgo == 'AdamW': optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
            else: optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!
                
            if scheduler>0: 
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
            
            
            # Model CheckPoint
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_epoch = checkpoint_state["epoch"]
                    self.net.load_state_dict(checkpoint_state["net_state_dict"])
                    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            else:
                start_epoch = 0


            balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
            train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
            train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring
            
            
            if self.val_data is not None: 
                val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
            
            if self.test_data is not None: 
                test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

            # if addContrast:
            #     balanced_batch_sampler_cont = BalancedBatchSampler(self.train_data, K, n_samples = batch_size_c, class_id = [float(x) for x in range(K)])
            #     cont_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler_cont)

            class_weights = torch.FloatTensor(C_ratio).cuda()
            mcloss = MCHingeLoss()
            contrast_loss = FMContrastiveSim()  ## Only one level here!! ?? Will this help??
   

            max_val_acc = 0
            best_epoch_trn = 0
            best_epoch_val_report= None
            best_epoch_trn_report = None
            best_model = copy.deepcopy(self.net)
            qa_val_acc = 0


            for epoch in range(n_epochs+1):
                loss_epoch = 0.0
                n_batches = 0
                
                for _, data in enumerate(train_loader):
                    X, y, labels_pos, alllabels_pos = data
                    X = X.to(self.device)
                    y = y.to(self.device)
                    outputs, feat_pos,_ = self.net(X)
                    optimizer.zero_grad()
                    loss = mcloss(outputs, y, K, C_ratio = class_weights)

                    X_pivot_0, X_ref_0, rating_0, label_0  = getMCContrastdata(feat_pos, labels_pos, [], [], n_samp = batch_contrast, pos_labels = pos_labels)

                    rating_0 = torch.tensor(rating_0).to(self.device)
                    con_loss = contrast_loss(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=1., margin = margin)
                    
                    l2_reg = torch.tensor(0.).to(self.device)

                    for name, param in self.net.named_parameters(): l2_reg += torch.norm(param)**2
                    
                    classloss = loss + lamda * l2_reg + Ccont * con_loss   # Equal Weights to different levels!!
                    classloss.backward()
                    optimizer.step()

                    loss_epoch += classloss.item()
                    n_batches += 1

                if epoch%verbose==0:
                    print('------------------------------------- ')
                    print('Epoch = {}'.format(epoch))
                    
                    trn_class_bal_metric, trn_acc, trn_report = self.evaluate(train_loader_scoring)
                    print('Train Loss = {}, Label loss ={}, Cont. Loss = {}'.format(classloss.item(), loss.item(), con_loss.item()))
                    print('Tr. Bal. Metric = {}, Train Acc = {}'.format(trn_class_bal_metric, trn_acc))
                    
                    if self.val_data is not None:
                        val_class_bal_metric, val_acc, val_report = self.evaluate(val_loader)
                        if val_acc>=max_val_acc:
                            max_val_acc = val_acc
                            best_epoch = epoch
                            best_epoch_trn = trn_acc
                            best_epoch_val_report = val_report
                            best_epoch_trn_report = trn_report
                            best_model.load_state_dict(self.net.state_dict())
                            qa_val_class_bal_metric, qa_val_acc, qa_val_report = self.evaluate(val_loader, best_model)

                        print('Val Bal. Metric = {}, Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_class_bal_metric, val_acc, best_epoch, max_val_acc, best_epoch_trn))
                        print('QA :: Val Acc = {}'.format(qa_val_acc))
                    # if self.test_data is not None:
                    #     tst_acc, _ = self.evaluate(test_loader)
                    #     print('Test Acc = {}'.format(tst_acc))
                    
                    checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }

        
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {"train_loss": classloss.item(), 
                        "lab_loss": loss.item(),
                        "cont_loss": con_loss.item() if con_loss else 1000,
                        "val_acc": val_acc,
                        "max_val_acc": max_val_acc,
                        "val_class_bal_metric": val_class_bal_metric},
                        checkpoint=checkpoint,
                    )

                # if epoch == n_epochs: # Final Best Result
                #     returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                    
                #     if self.val_data is not None:
                #         returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
                #     if self.test_data is not None: # Need to modify
                #         returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
            print('Finished Training') 
            return best_model, self.net


from sklearn.metrics import balanced_accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torch import nn
import numpy as np
from ray import train, tune
import math
from ray.train import Checkpoint, get_checkpoint
import tempfile
from pathlib import Path
import ray.cloudpickle as pickle
import copy


class BinHingeTrainer_wRAY(object):  # 2 -Class Problem Cost-sensitive
    def __init__(self,
                 X, y, labels, alllabelscont = [], # File name or the actual file
                 Xval = None, yval = None, labelsval = None, 
                 Xtst = None, ytst = None, labeltst = None,
                 Xuniv = None,
                 Xsemi = None,
                 net = None,
                 writer = False,
                 device = torch.device('cpu'), labelmap = None, tuner = None): # Assume only HPO no NAS
        
        
        self.X = X
        self.y = y
        self.labels = labels
        self.alllabelscont = alllabelscont
        self.K = len(set(self.y))
        if labelmap:
            self.labelmap = [None for i in range(self.K)]
            for k in labelmap:
                self.labelmap[labelmap[k]] = k
            # print(self.labelmap)
        else: self.labelmap = [i for i in range(self.K)]

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
        
        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst

        if Xuniv is not None: 
            self.Xuniv = Xuniv
            self.yuniv = -2*np.ones(Xuniv.shape[0])
            self.labeluniv = ['Univ' for _ in range(Xuniv.shape[0])]

        if Xsemi is not None: 
            self.Xsemi = Xsemi
            self.ysemi = -3*np.ones(Xsemi.shape[0])
            self.labelsemi = ['Semi' for _ in range(Xuniv.shape[0])]
        

        if net is not None: self.net =  net.to(device)
        self.device = device
        # if writer: self.writer = SummaryWriter()
        # else: self.writer = None
        
        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels,self.alllabelscont)
        # ratios = self.train_data.getCostRatios()
        

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None

        if Xuniv is not None: self.univ_data = BreakdanceDataLoader(self.Xuniv,self.yuniv,self.labeluniv)
        else: self.univ_data = None
        if Xsemi is not None: self.semi_data = BreakdanceDataLoader(self.Xsemi,self.ysemi,self.labelsemi)
        else: self.semi_data = None

    def costsensitive_score(self,tstlabels, scores):
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        p = 0
        n = 0
        r = self.C_test_ratio
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
        Pfp = fp/p 
        Pfn = fn/n
        acc = (r*Ptp+Ptn)/(r+1)
        return acc, Ptp, Ptn, Pfp, Pfn


    def evaluate(self,datloader, testmodel = None): 
        idx_label_score = []
        if not testmodel: testmodel = self.net
        testmodel.eval()

        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy() # 0, 1
                

                outputs, _, _ = testmodel(valX) # -1, 1
                outputs = outputs.data.cpu().numpy()
                idx_label_score += list(zip(valy.tolist(),
                                            outputs.tolist()))

        tstlabels, scores = zip(*idx_label_score)
        tstlabels = np.array(tstlabels)
        scores = np.array(scores)

        scores[np.where(scores>=0)]= 1.
        scores[np.where(scores<0)]= 0.

        scores = [v[0] for v in scores]

        # acc = balanced_accuracy_score(tstlabels, scores)#accuracy_score(y_true, y_pred)
        # print('Bal Acc = {}'.format(acc))
        acc, Ptp, Ptn, Pfp, Pfn = self.costsensitive_score(tstlabels, scores)
        # print('CS acc = {}, Ptp = {}, Ptn = {}, Pfp={}, Pfn={}'.format(acc, Ptp, Ptn, Pfp, Pfn))
        report = classification_report(tstlabels, scores, target_names=self.labelmap)
        
        report+='''\n
        Ptp = {}
        Ptn = {}
        '''.format(Ptp,Ptn)
        return acc, report


    def training(self,
                      lamda = 1e-6, 
                      optimalgo = 'SGD',
                      learning_rate = 3e-5,
                      batch_size = 8, 
                      batch_size_u = 50, 
                      batch_size_s = 50, 
                      batch_size_c = 50, 
                      batch_size_t = 25,
                      Cu = 0.0, 
                      G = 0.0, 
                      Cs = 0.0, 
                      addtrans=False, 
                      Ct = 0.0,
                      addContrast=False, 
                      Ccont = 0.0, 
                      margin = 0.0, 
                      n_epochs = 100, 
                      scheduler = -1, 
                      amsgrad = False, 
                      verbose = 100, 
                      smooth=False, 
                      C_ratio = 1.0, 
                      multi_level = False, dotune=False,C_test_ratio = 1.0): 
            
            returndict = dict() # will be used for HPO
            self.C_test_ratio = C_test_ratio
            K = self.K
            if optimalgo == 'SGD':
                optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
            if optimalgo == 'AdamW':
                optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
            else:
                optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!

            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_epoch = checkpoint_state["epoch"]
                    self.model.load_state_dict(checkpoint_state["model_state_dict"])
                    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            else:
                start_epoch = 0
            

                
            # if scheduler>0: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)

            balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
            train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
            # train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring

            
            if self.val_data is not None: 
                if addtrans:
                    balanced_batch_sampler_val = BalancedBatchSampler(self.val_data, 2, n_samples = batch_size_t, class_id = [float(x) for x in range(2)])
                    val_loader_trans = DataLoader(self.val_data, batch_sampler=balanced_batch_sampler_val)
                    transloss = HingeLoss()
                val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
            
            if self.test_data is not None: 
                test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
            
            if self.univ_data is not None:
                univ_loader = DataLoader(self.univ_data, batch_size=batch_size_u, shuffle=True)
                univloss = HingeLoss()
            
            if self.semi_data is not None:
                semi_loader = DataLoader(self.semi_data, batch_size=batch_size_s, shuffle=True)
                semiloss = HingeLoss()

            if addContrast:
                balanced_batch_sampler_cont = BalancedBatchSampler(self.train_data, K, n_samples = batch_size_c, class_id = [float(x) for x in range(K)])
                cont_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler_cont)


            posloss = HingeLoss()
            negloss = HingeLoss()
            

            ratios = self.train_data.getCostRatios()
            # print('Pos Samples = {}, Neg Samples = {}'.format(ratios[1],ratios[0]))

            ratios_test = self.val_data.getCostRatios()
            # print('Pos Samples (Test) = {}, Neg Samples (Test) = {}'.format(ratios_test[1],ratios_test[0]))

            if C_ratio<0: C_ratio = ratios[0]/ratios[1]
                # print('Setting C_ratio = {}'.format(C_ratio))

            max_val_acc = 0
            best_epoch = 0
            best_epoch_trn = 0
            best_epoch_val_report= None
            best_epoch_trn_report = None
            best_model = copy.deepcopy(self.net)
            qa_val_acc = 0

            for epoch in range(start_epoch,n_epochs+1):
                loss_epoch = 0.0
                n_batches = 0
                
                for it, data in enumerate(train_loader):
                    # print('Running iteration ={}/{}'.format(it,len(train_loader)))
                    X, y, labels, _ = data
                    Xpos = X[y==1].to(self.device)
                    Xneg = X[y==0].to(self.device)
                    ypos = np.ones(Xpos.shape[0])
                    yneg = -np.ones(Xneg.shape[0])
                    # y = y.to(self.device)
                    ypos = torch.from_numpy(ypos).to(self.device).float()
                    yneg = torch.from_numpy(yneg).to(self.device).float()

                    outputpos,_,_ = self.net(Xpos)
                    outputneg,_,_ = self.net(Xneg)

                    optimizer.zero_grad()
                    loss = C_ratio*posloss(outputpos,ypos,smooth = smooth) + negloss(outputneg,yneg,smooth = smooth)
                    
                    l2_reg = torch.tensor(0.).to(self.device)

                    for name, param in self.net.named_parameters():
                        l2_reg += torch.norm(param)**2
                    
                    classloss = loss + lamda * l2_reg

                    uloss = None
                    if self.univ_data is not None:
                        data_univ = next(iter(univ_loader))
                        XU, _,_, _ = data_univ
                        XU=XU.to(self.device)
                        outU,_,_ = self.net(XU)
                        outU_rep = torch.cat(2*[outU])
                        yU = torch.from_numpy(np.array([1. for _ in range(outU.shape[0])]+[-1. for _ in range(outU.shape[0])]))
                        yU = yU.long().cuda()
                        uloss = univloss(outU_rep,yU, smooth = smooth,  margin = -G)
                        classloss+=Cu*uloss
                    
                    sloss = None
                    if self.semi_data is not None:
                        data_semi = next(iter(semi_loader))
                        Xs, _,_, _ = data_semi
                        Xs=Xs.to(self.device)
                        outS,_,_ = self.net(Xs)
                        outS_rep = torch.cat(2*[outS])
                        yS = torch.from_numpy(np.array([1. for _ in range(outS.shape[0])]+[-1. for _ in range(outS.shape[0])]))
                        yS = yS.long().cuda()
                        sloss = semiloss(outS_rep,yS, nonconvex = True)
                        classloss+=Cs*sloss
                    
                    tloss = None
                    if addtrans:
                        data_trans = next(iter(val_loader_trans))
                        XT, _,_, _ = data_trans
                        XT=XT.to(self.device)
                        outT,_,_ = self.net(XT)
                        outT_rep = torch.cat(2*[outT])
                        yT = torch.from_numpy(np.array([1. for _ in range(outT.shape[0])]+[-1. for _ in range(outT.shape[0])]))
                        yT = yT.long().cuda()
                        tloss = transloss(outT_rep,yT, nonconvex = True)
                        classloss+=Ct*tloss
                        
                    con_loss = None
                    if addContrast:
                        contrast_loss_det = FMContrastiveSim()
                        contrast_loss_0 = FMContrastiveSim()
                        contrast_loss_1 = FMContrastiveSim()
                        contrast_loss_2 = FMContrastiveSim()

                        
                        data_cont = next(iter(cont_loader))
                        X_cont, _, _, cont_labels = data_cont
                        X_cont=X_cont.to(self.device)
                        _, feat_cont,_ = self.net(X_cont)

                        labels_det = getlevelLabels(cont_labels[0],level = -1)
                        labels_0 = getlevelLabels(cont_labels[0],level = 0)
                        labels_1 = getlevelLabels(cont_labels[0],level = 1)
                        labels_2 = getlevelLabels(cont_labels[0],level = 2)


                        X_pivot_d, X_ref_d, rating_d, label_d  = getMCContrastdata_ver2(feat_cont, labels_det, n_samp = batch_size_c)
                        X_pivot_0, X_ref_0, rating_0, label_0  = getMCContrastdata_ver2(feat_cont, labels_0, n_samp = batch_size_c)
                        X_pivot_1, X_ref_1, rating_1, label_1  = getMCContrastdata_ver2(feat_cont, labels_1, n_samp = batch_size_c)
                        X_pivot_2, X_ref_2, rating_2, label_2  = getMCContrastdata_ver2(feat_cont, labels_2, n_samp = batch_size_c)

                        rating_d = torch.tensor(rating_d).to(self.device)
                        rating_0 = torch.tensor(rating_0).to(self.device)
                        rating_1 = torch.tensor(rating_1).to(self.device)
                        rating_2 = torch.tensor(rating_2).to(self.device)

                        if multi_level:
                            con_loss_det = contrast_loss_det(X_pivot_d, X_ref_d, rating_d, level = 0., maxlevel=4., margin = margin)
                            con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 1., maxlevel=4., margin = margin)
                            con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 2., maxlevel=4., margin = margin)
                            con_loss_2 = contrast_loss_2(X_pivot_2, X_ref_2, rating_2, level = 3., maxlevel=4., margin = margin)  # Multilevel
                        else:
                            con_loss_det = contrast_loss_det(X_pivot_d, X_ref_d, rating_d, level = 0., maxlevel=1., margin = margin)
                            con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=1., margin = margin)
                            con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 0., maxlevel=1., margin = margin)  # Single level's
                            con_loss_2 = contrast_loss_2(X_pivot_2, X_ref_2, rating_2, level = 0., maxlevel=1., margin = margin)  # Single level's
                        
                        con_loss = con_loss_det+con_loss_0+con_loss_1+con_loss_2
                        classloss+=Ccont*con_loss

                    classloss.backward()
                    optimizer.step()

                    loss_epoch += classloss.item()
                    n_batches += 1

                # print('------------------------------------- ')
                # print('Epoch = {}'.format(epoch))
                if epoch%verbose==0:
                    trn_acc, trn_report = self.evaluate(train_loader_scoring)
                    print('Train Loss = {}, Train Acc = {}, Train Hinge ={}'.format(classloss.item(), trn_acc, loss.item()))


                    if self.univ_data is not None: 
                        print('Univ Loss = {} \n'.format(uloss.item()))

                    if self.semi_data is not None:
                        print('Semi Loss = {} \n'.format(sloss.item()))

                    if addtrans: 
                        print('Transd Loss = {} \n'.format(tloss.item()))

                    if addContrast: 
                        print('Contrast Loss = {}, Level -1 = {}, Level 0 = {}, Level 1 = {}, Level 2 = {}\n'.format(con_loss.item(), con_loss_det.item(),con_loss_0.item(),con_loss_1.item(),con_loss_2.item()))
                
                if self.val_data is not None:
                    val_acc, val_report = self.evaluate(val_loader)
                    if val_acc>=max_val_acc and trn_acc>=val_acc: #Typical Learning!! No Caveats!!
                        max_val_acc = val_acc
                        best_epoch = epoch
                        best_epoch_trn = trn_acc
                        best_epoch_val_report = val_report
                        best_epoch_trn_report = trn_report
                        best_model.load_state_dict(self.net.state_dict())
                        qa_val_acc, qa_val_report = self.evaluate(val_loader, best_model)
                        # best_model = copy.deepcopy(self.net)
                    
                    if epoch%verbose==0:
                        print('Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_acc, best_epoch, max_val_acc, best_epoch_trn))
                        print('QA :: Val Acc = {}'.format(qa_val_acc))


                    if self.test_data is not None:
                        tst_acc, _ = self.evaluate(test_loader)
                        print('Test Acc = {}'.format(tst_acc))
                        
                
                    checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
        
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {"train_loss": classloss.item(), 
                        "lab_loss": loss.item(),
                        "univ_loss": uloss.item() if uloss else 1000,
                        "semi_loss": sloss.item() if sloss else 1000,
                        "trans_loss": tloss.item() if tloss else 1000,
                        "cont_loss": con_loss.item() if con_loss else 1000,
                        "val_acc": val_acc,
                        "max_val_acc": max_val_acc},
                        checkpoint=checkpoint,
                    )


                if epoch == n_epochs: # Final Best Result
                    returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                    
                    if self.val_data is not None:
                        returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
                    if self.test_data is not None:
                        returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
            
            torch.save(best_model.state_dict(), './models_ckpt/best_model.ckpt')
            print('Finished Training') 
            return best_model, self.net, returndict






class BinHingeTrainer(object):  # 2 -Class Problem Cost-sensitive
    def __init__(self,
                 X, y, labels, alllabelscont = [], # File name or the actual file
                 Xval = None, yval = None, labelsval = None, 
                 Xtst = None, ytst = None, labeltst = None,
                 Xuniv = None,
                 Xsemi = None,
                 model = None,
                 writer = False,
                 device = torch.device('cpu'), labelmap = None): # Assume only HPO no NAS
        
        self.X = X
        self.y = y
        self.labels = labels
        self.alllabelscont = alllabelscont
        self.K = len(set(self.y))
        if labelmap:
            self.labelmap = [None for i in range(self.K)]
            for k in labelmap:
                self.labelmap[labelmap[k]] = k
            # print(self.labelmap)
        else: self.labelmap = [i for i in range(self.K)]

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
        
        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst

        if Xuniv is not None: 
            self.Xuniv = Xuniv
            self.yuniv = -2*np.ones(Xuniv.shape[0])
            self.labeluniv = ['Univ' for _ in range(Xuniv.shape[0])]

        if Xsemi is not None: 
            self.Xsemi = Xsemi
            self.ysemi = -3*np.ones(Xsemi.shape[0])
            self.labelsemi = ['Semi' for _ in range(Xuniv.shape[0])]
        

        if model is not None: self.net =  model.to(device)
        self.device = device
        # if writer: self.writer = SummaryWriter()
        # else: self.writer = None
        
        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels,self.alllabelscont)
        # ratios = self.train_data.getCostRatios()
        

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None

        if Xuniv is not None: self.univ_data = BreakdanceDataLoader(self.Xuniv,self.yuniv,self.labeluniv)
        else: self.univ_data = None
        if Xsemi is not None: self.semi_data = BreakdanceDataLoader(self.Xsemi,self.ysemi,self.labelsemi)
        else: self.semi_data = None



    def evaluate(self,datloader): 
        idx_label_score = []
        self.net.eval()

        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy() # 0, 1
                

                outputs, _, _ = self.net(valX) # -1, 1
                outputs = outputs.data.cpu().numpy()
                idx_label_score += list(zip(valy.tolist(),
                                            outputs.tolist()))

        tstlabels, scores = zip(*idx_label_score)
        tstlabels = np.array(tstlabels)
        scores = np.array(scores)

        scores[np.where(scores>=0)]= 1.
        scores[np.where(scores<0)]= 0.

        acc = balanced_accuracy_score(tstlabels, scores)#accuracy_score(y_true, y_pred)
        report = classification_report(tstlabels, scores, target_names=self.labelmap)
        return acc, report


    def training(self,
                      lamda = 1e-2, 
                      optimalgo = 'SGD',
                      learning_rate = 5e-5,
                      batch_size = 8, batch_size_u = 50, batch_size_s = 50, batch_size_c = 50, batch_size_t = 25,
                      Cu = 0.0, G = 0.0, Cs = 0.0, 
                      addtrans=False, Ct = 0.0,
                      addContrast=False, Ccont = 0.0, margin = 0.5, 
                      n_epochs = 50000, scheduler = -1, amsgrad = False, verbose = 100, smooth=False, C_ratio = -1, multi_level = False, nni=False, tune=False): 
            
            returndict = dict() # will be used for HPO
            intermediate = dict()
            intermediate['train_loss']=[]
            intermediate['trn_acc']=[]
            intermediate['train_hinge']=[]
            
            K = self.K
            if optimalgo == 'SGD':
                optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
            if optimalgo == 'AdamW':
                optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
            else:
                optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!
                
            if scheduler>0: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)

            balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
            train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
            # train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring

            
            if self.val_data is not None: 
                if addtrans:
                    balanced_batch_sampler_val = BalancedBatchSampler(self.val_data, 2, n_samples = batch_size_t, class_id = [float(x) for x in range(2)])
                    val_loader_trans = DataLoader(self.val_data, batch_sampler=balanced_batch_sampler_val)
                    transloss = HingeLoss()
                    intermediate['tloss'] = []
                val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
                intermediate['val_acc'] = []
            
            if self.test_data is not None: 
                test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
                intermediate['tst_acc'] = []
            
            if self.univ_data is not None:
                univ_loader = DataLoader(self.univ_data, batch_size=batch_size_u, shuffle=True)
                univloss = HingeLoss()
                intermediate['uloss'] = []
            
            if self.semi_data is not None:
                semi_loader = DataLoader(self.semi_data, batch_size=batch_size_s, shuffle=True)
                semiloss = HingeLoss()
                intermediate['sloss'] = []
            if addContrast:
                balanced_batch_sampler_cont = BalancedBatchSampler(self.train_data, K, n_samples = batch_size_c, class_id = [float(x) for x in range(K)])
                cont_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler_cont)
                intermediate['con_loss']=[]

            posloss = HingeLoss()
            negloss = HingeLoss()
            

            ratios = self.train_data.getCostRatios()
            print('Pos Samples = {}, Neg Samples = {}'.format(ratios[1],ratios[0]))

            ratios_test = self.val_data.getCostRatios()
            print('Pos Samples (Test) = {}, Neg Samples (Test) = {}'.format(ratios_test[1],ratios_test[0]))

            if C_ratio<0: 
                C_ratio = ratios[0]/ratios[1]
                print('Setting C_ratio = {}'.format(C_ratio))

            max_val_acc = 0
            best_epoch_trn = 0
            best_epoch_val_report= None
            best_epoch_trn_report = None

            for epoch in range(n_epochs+1):
                loss_epoch = 0.0
                n_batches = 0
                
                for it, data in enumerate(train_loader):
                    # print('Running iteration ={}/{}'.format(it,len(train_loader)))
                    X, y, labels, _ = data
                    Xpos = X[y==1].to(self.device)
                    Xneg = X[y==0].to(self.device)
                    ypos = np.ones(Xpos.shape[0])
                    yneg = -np.ones(Xneg.shape[0])
                    # y = y.to(self.device)
                    ypos = torch.from_numpy(ypos).to(self.device).float()
                    yneg = torch.from_numpy(yneg).to(self.device).float()

                    outputpos,_,_ = self.net(Xpos)
                    outputneg,_,_ = self.net(Xneg)

                    optimizer.zero_grad()
                    loss = C_ratio*posloss(outputpos,ypos,smooth = smooth) + negloss(outputneg,yneg,smooth = smooth)
                    
                    l2_reg = torch.tensor(0.).to(self.device)

                    for name, param in self.net.named_parameters():
                        l2_reg += torch.norm(param)**2
                    
                    classloss = loss + lamda * l2_reg

                    if self.univ_data is not None:
                        data_univ = next(iter(univ_loader))
                        XU, _,_, _ = data_univ
                        XU=XU.to(self.device)
                        outU,_,_ = self.net(XU)
                        outU_rep = torch.cat(2*[outU])
                        yU = torch.from_numpy(np.array([1. for _ in range(outU.shape[0])]+[-1. for _ in range(outU.shape[0])]))
                        yU = yU.long().cuda()
                        uloss = univloss(outU_rep,yU, smooth = smooth,  margin = -G)
                        classloss+=Cu*uloss
                    
                    if self.semi_data is not None:
                        data_semi = next(iter(semi_loader))
                        Xs, _,_, _ = data_semi
                        Xs=Xs.to(self.device)
                        outS,_,_ = self.net(Xs)
                        outS_rep = torch.cat(2*[outS])
                        yS = torch.from_numpy(np.array([1. for _ in range(outS.shape[0])]+[-1. for _ in range(outS.shape[0])]))
                        yS = yS.long().cuda()
                        sloss = semiloss(outS_rep,yS, nonconvex = True)
                        classloss+=Cs*sloss

                    if addtrans:
                        data_trans = next(iter(val_loader_trans))
                        XT, _,_, _ = data_trans
                        XT=XT.to(self.device)
                        outT,_,_ = self.net(XT)
                        outT_rep = torch.cat(2*[outT])
                        yT = torch.from_numpy(np.array([1. for _ in range(outT.shape[0])]+[-1. for _ in range(outT.shape[0])]))
                        yT = yT.long().cuda()
                        tloss = transloss(outT_rep,yT, nonconvex = True)
                        classloss+=Ct*tloss
                        

                    if addContrast:
                        contrast_loss_det = FMContrastiveSim()
                        contrast_loss_0 = FMContrastiveSim()
                        contrast_loss_1 = FMContrastiveSim()
                        contrast_loss_2 = FMContrastiveSim()

                        
                        data_cont = next(iter(cont_loader))
                        X_cont, _, _, cont_labels = data_cont
                        X_cont=X_cont.to(self.device)
                        _, feat_cont,_ = self.net(X_cont)

                        labels_det = getlevelLabels(cont_labels[0],level = -1)
                        labels_0 = getlevelLabels(cont_labels[0],level = 0)
                        labels_1 = getlevelLabels(cont_labels[0],level = 1)
                        labels_2 = getlevelLabels(cont_labels[0],level = 2)


                        X_pivot_d, X_ref_d, rating_d, label_d  = getMCContrastdata_ver2(feat_cont, labels_det, n_samp = batch_size_c)
                        X_pivot_0, X_ref_0, rating_0, label_0  = getMCContrastdata_ver2(feat_cont, labels_0, n_samp = batch_size_c)
                        X_pivot_1, X_ref_1, rating_1, label_1  = getMCContrastdata_ver2(feat_cont, labels_1, n_samp = batch_size_c)
                        X_pivot_2, X_ref_2, rating_2, label_2  = getMCContrastdata_ver2(feat_cont, labels_2, n_samp = batch_size_c)

                        rating_d = torch.tensor(rating_d).to(self.device)
                        rating_0 = torch.tensor(rating_0).to(self.device)
                        rating_1 = torch.tensor(rating_1).to(self.device)
                        rating_2 = torch.tensor(rating_2).to(self.device)

                        if multi_level:
                            con_loss_det = contrast_loss_det(X_pivot_d, X_ref_d, rating_d, level = 0., maxlevel=4., margin = margin)
                            con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 1., maxlevel=4., margin = margin)
                            con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 2., maxlevel=4., margin = margin)
                            con_loss_2 = contrast_loss_2(X_pivot_2, X_ref_2, rating_2, level = 3., maxlevel=4., margin = margin)  # Multilevel
                        else:
                            con_loss_det = contrast_loss_det(X_pivot_d, X_ref_d, rating_d, level = 0., maxlevel=1., margin = margin)
                            con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=1., margin = margin)
                            con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 0., maxlevel=1., margin = margin)  # Single level's
                            con_loss_2 = contrast_loss_2(X_pivot_2, X_ref_2, rating_2, level = 0., maxlevel=1., margin = margin)  # Single level's
                        
                        con_loss = con_loss_det+con_loss_0+con_loss_1+con_loss_2
                        classloss+=Ccont*con_loss

                    classloss.backward()
                    optimizer.step()

                    loss_epoch += classloss.item()
                    n_batches += 1

                if epoch%verbose==0:
                    print('------------------------------------- ')
                    print('Epoch = {}'.format(epoch))
                    
                    trn_acc, trn_report = self.evaluate(train_loader_scoring)
                    print('Train Loss = {}, Train Acc = {}, Train Hinge ={}'.format(classloss.item(), trn_acc, loss.item()))
                    intermediate['train_loss'].append(classloss.item())
                    intermediate['trn_acc'].append(trn_acc)
                    intermediate['train_hinge'].append(loss.item())

                    if self.univ_data is not None: 
                        print('Univ Loss = {} \n'.format(uloss.item()))
                        intermediate['uloss'].append(uloss.item())
                    if self.semi_data is not None:
                        print('Semi Loss = {} \n'.format(sloss.item()))
                        intermediate['sloss'].append(sloss.item())
                    if addtrans: 
                        print('Transd Loss = {} \n'.format(tloss.item()))
                        intermediate['tloss'].append(tloss.item())
                    if addContrast: 
                        print('Contrast Loss = {}, Level -1 = {}, Level 0 = {}, Level 1 = {}, Level 2 = {}\n'.format(con_loss.item(), con_loss_det.item(),con_loss_0.item(),con_loss_1.item(),con_loss_2.item()))
                        intermediate['con_loss'].append(con_loss.item())
                    
                    if self.val_data is not None:
                        val_acc, val_report = self.evaluate(val_loader)
                        if val_acc>=max_val_acc:
                            max_val_acc = val_acc
                            best_epoch = epoch
                            best_epoch_trn = trn_acc
                            best_epoch_val_report = val_report
                            best_epoch_trn_report = trn_report
                        print('Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_acc, best_epoch, max_val_acc, best_epoch_trn))
                        intermediate['val_acc'].append(val_acc)
                        
                        if nni:
                            import nni
                            nni.report_intermediate_result(val_acc)



                    if self.test_data is not None:
                        tst_acc, _ = self.evaluate(test_loader)
                        print('Test Acc = {}'.format(tst_acc))
                        intermediate['tst_acc'].append(tst_acc)

                if epoch == n_epochs: # Final Best Result
                    returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                    
                    if self.val_data is not None:
                        returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
                    if self.test_data is not None:
                        returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
            return self.net, returndict, intermediate






class MCEntropyTrainer(object):
    def __init__(self,
                 X, y, labels, # File name or the actual file
                 Xval = None, yval = None, labelsval = None, 
                 Xtst = None, ytst = None, labeltst = None,
                 model = None,
                 writer = False,
                 device = torch.device('cpu'), labelmap = None): # Assume only HPO no NAS
        
        self.X = X
        self.y = y
        self.labels = labels
        self.K = len(set(self.y))
        if labelmap:
            self.labelmap = [None for i in range(self.K)]
            for k in labelmap:
                self.labelmap[labelmap[k]] = k
            # print(self.labelmap)
        else:
            self.labelmap = [i for i in range(self.K)]

        

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
        

        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst
        
        
        if model is not None: self.net =  model.to(device)
        self.device = device
        # if writer: self.writer = SummaryWriter()
        # else: self.writer = None
        
        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels)
        ratios = self.train_data.getCostRatios()
        print('Prior Probability Training {}'.format(ratios))

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None



    def evaluate(self,datloader): 
        idx_label_score = []
        self.net.eval()
        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy()
                outputs, _, _ = self.net(valX)
                outputs = np.argmax(outputs.data.cpu().numpy(), axis = 1)
                idx_label_score += list(zip(valy.tolist(),
                                            outputs.tolist(), vallabels))

        y_true, y_pred, labels = zip(*idx_label_score)   # Need original ratings for test!
        acc = balanced_accuracy_score(y_true, y_pred)#accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labelmap)
        # print(report)
        return acc, report


    def training(self,
                      lamda = 1e-2, 
                      optimalgo = 'SGD',
                      learning_rate = 5e-5,
                      batch_size = 8, batch_size_trans = 5,
                      n_epochs = 50000, scheduler = -1, amsgrad = False, verbose = 100): 
            
            returndict = dict() # will be used for HPO
            
            K = self.K
            if optimalgo == 'SGD':
                optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
            if optimalgo == 'AdamW':
                optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
            else:
                optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!
                
            if scheduler>0: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)

            balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
            train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
            # train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring

            
            if self.val_data is not None: 
                balanced_batch_sampler_val = BalancedBatchSampler(self.val_data, K, n_samples = batch_size_trans, class_id = [float(x) for x in range(K)])
                val_loader_trans = DataLoader(self.val_data, batch_sampler=balanced_batch_sampler_val)

                val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
            
            if self.test_data is not None: 
                test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

            mcloss = nn.CrossEntropyLoss()

            max_val_acc = 0
            best_epoch_trn = 0
            best_epoch_val_report= None
            best_epoch_trn_report = None

            for epoch in range(n_epochs+1):
                loss_epoch = 0.0
                n_batches = 0
                
                for _, data in enumerate(train_loader):
                    X, y, labels, _ = data
                    X = X.to(self.device)
                    y = y.to(self.device)

                    outputs,_,_ = self.net(X)
                    optimizer.zero_grad()
                    loss = mcloss(outputs, y)
                    
                    l2_reg = torch.tensor(0.).to(self.device)

                    for name, param in self.net.named_parameters():
                        l2_reg += torch.norm(param)**2
                    
                    classloss = loss + lamda * l2_reg
                    classloss.backward()
                    optimizer.step()

                    loss_epoch += classloss.item()
                    n_batches += 1

                if epoch%verbose==0:
                    print('------------------------------------- ')
                    print('Epoch = {}'.format(epoch))
                    
                    trn_acc, trn_report = self.evaluate(train_loader_scoring)
                    print('Train Loss = {}, Train Acc = {}, Train Hinge ={}'.format(classloss.item(), trn_acc, loss.item()))
                    
                    if self.val_data is not None:
                        val_acc, val_report = self.evaluate(val_loader)
                        if val_acc>=max_val_acc:
                            max_val_acc = val_acc
                            best_epoch = epoch
                            best_epoch_trn = trn_acc
                            best_epoch_val_report = val_report
                            best_epoch_trn_report = trn_report
                        print('Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_acc, best_epoch, max_val_acc, best_epoch_trn))



                    if self.test_data is not None:
                        tst_acc, _ = self.evaluate(test_loader)
                        print('Test Acc = {}'.format(tst_acc))

                if epoch == n_epochs: # Final Best Result
                    returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                    
                    if self.val_data is not None:
                        returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
                    if self.test_data is not None:
                        returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
            return self.net, returndict



class MCEntropyAdvancedTrainer(object):
    def __init__(self,
                 X, y, labels, altlabels,# File name or the actual file
                 Xval = None, yval = None, labelsval = None, altlabelsval = [],
                 Xtst = None, ytst = None, labeltst = None, altlabelstst = [],
                 Xadd = None, yadd = None, labeladd = None, altlabelsadd = [],
                 model = None,
                 writer = False,
                 device = torch.device('cpu'), labelmap = None): # Assume only HPO no NAS
        
        self.X = X
        self.y = y
        self.labels = labels
        
        self.K = len(set(y))
        if labelmap:
            self.labelmap = [None for i in range(self.K)]
            for k in labelmap:
                self.labelmap[labelmap[k]] = k
        else:
            self.labelmap = [i for i in range(self.K)]


        self.altlabels = altlabels 

        if Xval is not None: 
            self.Xval = Xval
            self.yval = yval
            self.labelsval = labelsval
            self.altlabelsval = altlabelsval
        

        if Xtst is not None: 
            self.Xtst = Xtst
            self.ytst = ytst
            self.labeltst = labeltst
            self.altlabelstst = altlabelstst

        if Xadd is not None:
            self.Xadd = Xadd 
            self.yadd = yadd 
            self.labeladd = labeladd
            self.altlabelsadd = altlabelsadd


        if model is not None: self.net =  model.to(device)
        self.device = device
        # if writer: self.writer = SummaryWriter()
        # else: self.writer = None

        self.train_data = BreakdanceDataLoader(self.X,self.y,self.labels,self.altlabels)
        ratios = self.train_data.getCostRatios()
        print('Prior Probability Training {}'.format(ratios))

        if Xval is not None: self.val_data = BreakdanceDataLoader(self.Xval,self.yval,self.labelsval)
        else: self.val_data = None
        if Xtst is not None: self.test_data = BreakdanceDataLoader(self.Xtst,self.ytst,self.labeltst)
        else: self.test_data = None
        if Xadd is not None: self.add_data = BreakdanceDataLoader(self.Xadd,self.yadd,self.labeladd, self.altlabelsadd)
        else: self.add_data = None


    def evaluate(self,datloader): 
        idx_label_score = []
        self.net.eval()
        with torch.no_grad():
            for data in datloader:
                valX,valy,vallabels, _ = data
                valX = valX.to(self.device)
                valy = valy.cpu().data.numpy()
                outputs, _, _ = self.net(valX)
                outputs = np.argmax(outputs.data.cpu().numpy(), axis = 1)
                idx_label_score += list(zip(valy.tolist(),
                                            outputs.tolist(), vallabels))

        y_true, y_pred, labels = zip(*idx_label_score)   # Need original ratings for test!
        acc = balanced_accuracy_score(y_true, y_pred)#accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labelmap)
        return acc, report



    # Here we will do 1. Labeled, 2. Transductive, 3. Contrastive

    def training(self,
                      lamda = 1e-2, 
                      optimalgo = 'SGD',
                      learning_rate = 5e-5,
                      batch_size = 8, batch_size_trans = 5, batch_contrast = 50, 
                      multi_level = False,
                      delta = 1e-6, margin = 0., gamma = -1,
                      n_epochs = 50000, scheduler = -1, amsgrad = False, verbose = 100): 
            
            returndict = dict() # will be used for HPO
            
            K = self.K
            if optimalgo == 'SGD':
                optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
            if optimalgo == 'AdamW':
                optimizer = torch.optim.AdamW(self.net.parameters(), lr = learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)
            else:
                optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.0, 0.999), amsgrad=amsgrad)  # may set the momentum too!
                
            if scheduler>0: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)

            balanced_batch_sampler = BalancedBatchSampler(self.train_data, K, n_samples = batch_size, class_id = [float(x) for x in range(K)])
            balanced_batch_sampler_neg = BalancedBatchSampler(self.train_data, 1, n_samples = batch_size, class_id = [0])


            train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler)
            train_loader_scoring = DataLoader(self.train_data, batch_size=batch_size, shuffle=False) # Will be used for scoring
            neg_train_loader = DataLoader(self.train_data, batch_sampler=balanced_batch_sampler_neg)  # Only the negative samples

            if self.add_data is not None:  # Additional non class samples
                L = len(set(self.yadd))
                balanced_batch_sampler_neg_add = BalancedBatchSampler(self.add_data, L, n_samples = batch_contrast, class_id = [float(x) for x in range(L)])
                neg_train_loader_add = DataLoader(self.add_data, batch_sampler=balanced_batch_sampler_neg_add)  # Only the negative samples

            
            if self.val_data is not None: 
                balanced_batch_sampler_val = BalancedBatchSampler(self.val_data, K, n_samples = batch_size_trans, class_id = [float(x) for x in range(K)])
                val_loader_trans = DataLoader(self.val_data, batch_sampler=balanced_batch_sampler_val)
                val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
            
            if self.test_data is not None: 
                test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

            mcloss = nn.CrossEntropyLoss()
            contrast_loss_0 = FMContrastiveSim()
            contrast_loss_1 = FMContrastiveSim()
            contrast_loss_2 = FMContrastiveSim()  # Do not need multiple copies. Just to be safe different graph

            max_val_acc = 0
            best_epoch_trn = 0
            best_epoch_val_report= None
            best_epoch_trn_report = None

            for epoch in range(n_epochs+1):
                loss_epoch = 0.0
                n_batches = 0
                
                for _, data in enumerate(train_loader):
                    X, y, labels_pos, alllabels_pos = data
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    # print("Pos labels = {}".format(labels_pos))

                    outputs,feat_pos,_ = self.net(X)
                    optimizer.zero_grad()
                    loss = mcloss(outputs, y)

                    # Contrastive Loss
                    # data_neg = next(iter(neg_train_loader))
                    # X_neg, y_neg, labels_neg, alllabels_neg = data_neg

                    if self.add_data is not None:
                        data_neg_add = next(iter(neg_train_loader_add))
                        X_neg, y_neg, labels_neg, alllabels_neg = data_neg_add
                        # if len(labels_neg):
                        #     X_neg_1, y_neg_1, labels_neg_1, alllabels_neg_1 = data_neg_add
                        #     X_neg = torch.cat((X_neg,X_neg_1),0)
                        #     y_neg = torch.cat((y_neg,y_neg_1),0)
                        #     labels_neg+=labels_neg_1
                        #     alllabels_neg[0]+=alllabels_neg_1[0]
                        #     alllabels_neg[1]+=alllabels_neg_1[1]
                    


                        # print('X shape = {}, y shape = {}, labels_neg = {}, alllabels_neg = {}'.format(X_neg.shape, y_neg.shape, labels_neg, alllabels_neg))



                    # print("Neg labels = {}".format(labels_neg))
                    X_neg = X_neg.to(self.device)
                    y_neg = y_neg.to(self.device)

                 
                    _,feat_neg,_ = self.net(X_neg)  
                    X_pivot_0, X_ref_0, rating_0, label_0  = getMCContrastdata(feat_pos, labels_pos, feat_neg, labels_neg, n_samp = batch_contrast, pos_labels = ['Power:Windmill:Munchmill/Babymill','Power:Windmill:Barrel Windmill', 'Power:Windmill:Bellymill', 'Power:Windmill:Windmill','None'])
                    X_pivot_1, X_ref_1, rating_1, label_1  = getContrastdata(feat_pos, alllabels_pos[1], feat_neg, alllabels_neg[1], n_samp = batch_contrast, pos_labels = ['Power:Windmill'])
                    X_pivot_2, X_ref_2, rating_2, label_2  = getContrastdata(feat_pos, alllabels_pos[0], feat_neg, alllabels_neg[0], n_samp = batch_contrast, pos_labels = ['Power']) 

                    rating_0 = torch.tensor(rating_0).to(self.device)
                    rating_1 = torch.tensor(rating_1).to(self.device)
                    rating_2 = torch.tensor(rating_2).to(self.device)


                    if multi_level:
                        con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=3., margin = margin)
                        con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 1., maxlevel=3., margin = margin)
                        con_loss_2 = contrast_loss_2(X_pivot_2, X_ref_2, rating_2, level = 2., maxlevel=3., margin = margin)  # Multilevel
                    else:
                        con_loss_0 = contrast_loss_0(X_pivot_0, X_ref_0, rating_0, level = 0., maxlevel=1., margin = margin)
                        con_loss_1 = contrast_loss_1(X_pivot_1, X_ref_1, rating_1, level = 0., maxlevel=1., margin = margin)  # Single level's
                        con_loss_2 = contrast_loss_2(X_pivot_2, X_ref_2, rating_2, level = 0., maxlevel=1., margin = margin)  # Single level's
                    
                    con_loss = con_loss_0+con_loss_1+con_loss_2

                    l2_reg = torch.tensor(0.).to(self.device)

                    for name, param in self.net.named_parameters():
                        l2_reg += torch.norm(param)**2
                    
                    classloss = loss + lamda * l2_reg + delta * con_loss   # Equal Weights to different levels!!
                    classloss.backward()
                    optimizer.step()

                    loss_epoch += classloss.item()
                    n_batches += 1

                if epoch%verbose==0:
                    print('------------------------------------- ')
                    print('Epoch = {}'.format(epoch))
                    
                    trn_acc, trn_report = self.evaluate(train_loader_scoring)
                    print('Train Loss = {}, Train Acc = {}, Train Hinge ={}, Train Contrastive (level0, level1, level2)= {}({}, {}, {})'.format(classloss.item(), 
                                                                                                                        trn_acc, 
                                                                                                                        loss.item(), 
                                                                                                                        con_loss.item(), con_loss_0.item(), 
                                                                                                                        con_loss_1.item(), 
                                                                                                                        con_loss_2.item()))
                    
                    if self.val_data is not None:
                        val_acc, val_report = self.evaluate(val_loader)
                        if val_acc>=max_val_acc:
                            max_val_acc = val_acc
                            best_epoch = epoch
                            best_epoch_trn = trn_acc
                            best_epoch_val_report = val_report
                            best_epoch_trn_report = trn_report
                        print('Val Acc = {}. Best Val/Trn Acc @{} = {}/{}'.format(val_acc, best_epoch, max_val_acc, best_epoch_trn))

                    if self.test_data is not None:
                        tst_acc, _ = self.evaluate(test_loader)
                        print('Test Acc = {}'.format(tst_acc))

                if epoch == n_epochs: # Final Best Result
                    returndict['trn_acc'], returndict['trn_report']  = best_epoch_trn, best_epoch_trn_report
                    
                    if self.val_data is not None:
                        returndict['val_acc'],returndict['val_report'] = max_val_acc, best_epoch_val_report
                    if self.test_data is not None: # Need to modify
                        returndict['tst_acc'],  returndict['tst_report']  = self.evaluate(test_loader)
            return self.net, returndict