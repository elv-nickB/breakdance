
import torch
import torch.nn as nn
import pickle as pkl
import argparse

from src.model import MLP
from src.training import MCHingeAdvancedTrainerMultLevel_RAY

def main():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--dataset', type=str, default='config.yaml', help='Path to dataset pickle file')
    parser.add_argument('--output', type=str, default='output', help='Output model path')
    parser.add_argument('--num_epochs', type=int, default=250, help='Number of training epochs')
    args = parser.parse_args()

    labelmap = {'None': 0, 'powermove': 1, 'footwork' : 2, 'toprock': 3}

    net = MLP(
        input_dim=1024,
        num_output=4, 
        n_feat_hidden_layers = 2,
        n_class_hidden_layers = 1, 
        feature_scale = 2, 
        class_scale = 2, 
        hidden_resnet=False
    )

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

    trainX, trainy, trainlabel, testX, testy, testlabel, trainaltlabel, testaltlabel = pkl.load(open(args.dataset,'rb'))

    trainer = MCHingeAdvancedTrainerMultLevel_RAY(
        trainX, trainy, trainlabel, trainaltlabel, labelmap,
        Xval = testX, yval = testy, labelsval = testlabel, 
        model=net, device = torch.device(device)
    )

    bestmodel, _ = trainer.training(
        lamda = 0., 
        optimalgo = 'AdamW',
        learning_rate = 3e-5,
        batch_size = 100, 
        batch_contrast = 200, 
        Ccont = 0.0, 
        margin = 0.0, 
        C_ratio = [1., 1., 1., 1.], 
        n_epochs = args.num_epochs, 
        scheduler = 1, 
        amsgrad = False, 
        pos_labels = ['None', 'footwork', 'powermove', 'toprock'],
        verbose = 10,
        multi_level=True
    )

    torch.save(bestmodel.state_dict(), args.output)

if __name__ == '__main__':
    main()