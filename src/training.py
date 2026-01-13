from sklearn.metrics import balanced_accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader
from src.data import BreakdanceDataLoader, getMCContrastdata, BalancedBatchSampler, getMCContrastdata_ver2, getlevelLabels
from ray.train import Checkpoint, get_checkpoint
from src.loss import MCHingeLoss, FMContrastiveSim
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
        #torch.save(best_model.state_dict(), '/home/elv-sauptik/PROJECTS/breakdance/models_ckpt/{}.ckpt'.format('best_model'))
        print('Finished Training') 
        return best_model, self.net