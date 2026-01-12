from src.model import MLP
from src.training import MCHingeAdvancedTrainer_RAY, MCHingeAdvancedTrainerMultLevel_RAY
import torch
import torch.nn as nn
import pickle as pkl
import numpy as np


labelmap = {'None': 0, 'powermove': 1, 'footwork' : 2, 'toprock': 3}
#net = MLP(input_dim=1024,
#                num_output=4, 
#                n_feat_hidden_layers = 3,
#                n_class_hidden_layers = 2, 
#                feature_scale = 2, 
#                class_scale = 2, 
#                hidden_resnet=False)

net = MLP(input_dim=1024,
            num_output=4, 
            n_feat_hidden_layers = 2,
            n_class_hidden_layers = 1, 
            feature_scale = 2, 
            class_scale = 2, 
            hidden_resnet=False)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.001)

net.apply(init_weights)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
net.to(device)

trainX,trainy,trainlabel,testX,testy,testlabel,trainaltlabel,testaltlabel = pkl.load(open('redbull_frague_denver_brazil_legacy_merged_OCT27_5000.pkl','rb'))
#trainX = np.vstack([trainX, trainX])
#trainy = trainy + trainy
#trainlabel = trainlabel + trainlabel
#trainaltlabel = [trainaltlabel[0] + trainaltlabel[0]]
#
#testX = np.vstack([testX] * 7)
#testy = testy * 7
#testlabel = testlabel * 7
#testaltlabel = [testaltlabel[0] * 7]

trainer = MCHingeAdvancedTrainerMultLevel_RAY(trainX, trainy, trainlabel, trainaltlabel, labelmap,
                        Xval = testX, yval = testy, labelsval = testlabel, 
                        model=net, device = torch.device(device)) #MCEntropyAdvancedTrainer_RAY. MCHingeAdvancedTrainer_RAY

bestmodel,finalmodel = trainer.training(lamda = 0., 
                      optimalgo = 'AdamW',
                      learning_rate = 3e-5,
                      batch_size = 20, 
                      batch_contrast = 200, 
                      Ccont = 0.0, 
                      margin = 0.0, 
                      C_ratio = [1., 1., 1., 1.], 
                      n_epochs = 5000, 
                      scheduler = 1, 
                      amsgrad = False, 
                      pos_labels = ['None', 'footwork', 'powermove', 'toprock'],
                      verbose = 10)