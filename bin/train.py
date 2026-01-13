from src.model import MLP
from src.training import MCHingeAdvancedTrainerMultLevel_RAY
import torch
import torch.nn as nn
import pickle as pkl


labelmap = {'None': 0, 'powermove': 1, 'footwork' : 2, 'toprock': 3}

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

trainX,trainy,trainlabel,testX,testy,testlabel,trainaltlabel,testaltlabel = pkl.load(open('out.pkl','rb'))

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
                      n_epochs = 150, 
                      scheduler = 1, 
                      amsgrad = False, 
                      pos_labels = ['None', 'footwork', 'powermove', 'toprock'],
                      verbose = 10)


torch.save(bestmodel.state_dict(), 'best_weights.ckpt')