from torch import nn
import torch.nn.functional as F

def linearReluBlock(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f),
                         nn.BatchNorm1d(out_f),
                        nn.ReLU(True))

def linearGeluBlock(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f),
                         nn.LayerNorm(out_f),
                         nn.GELU())


def linearLeakyReluBlock(in_f, out_f):

    return nn.Sequential(nn.Linear(in_f, out_f),
                        #  nn.BatchNorm1d(out_f),
                        nn.LeakyReLU(True))

class MLP(nn.Module):   # Video Features are ImageBind ish!!
    """
    Multi-layer perceptron with single hidden layer.
    """
    def __init__(self,
                 input_dim=1024,
                 num_output=1, 
                 n_feat_hidden_layers = 1,
                 n_class_hidden_layers=1, 
                 feature_scale = 2, class_scale = 2, hidden_resnet = False):
        
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_output = num_output
        self.hidden_resnet = hidden_resnet
    
        # Mapping from query --> decision space
        inout_dims = [input_dim]
        for i in range(n_feat_hidden_layers): inout_dims.append(int(inout_dims[i]//feature_scale))
        feat_hidden_blocks = [linearLeakyReluBlock(inout_dims[i], inout_dims[i+1]) for i in range(len(inout_dims)-1)]
        self.feat_hidden_layers = nn.Sequential(*feat_hidden_blocks)

        inout_dims = [inout_dims[-1]]
        for i in range(n_class_hidden_layers): inout_dims.append(int(inout_dims[i]//class_scale))
        class_hidden_blocks = [linearLeakyReluBlock(inout_dims[i], inout_dims[i+1]) for i in range(len(inout_dims)-1)]
        self.class_hidden_layers = nn.Sequential(*class_hidden_blocks)
        self.output =  nn.Linear(inout_dims[-1], self.num_output)

    def forward(self, in_data):
        hidden_features = self.feat_hidden_layers(in_data)
        if self.hidden_resnet: hidden_features = hidden_features+in_data # Scale should be equal to 1
        class_feat = self.class_hidden_layers(hidden_features)
        # if feat: return hidden_features
        out = self.output(class_feat)
        return out, hidden_features, in_data