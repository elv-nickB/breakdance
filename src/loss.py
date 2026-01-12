

from torch.nn.functional import cosine_similarity
import torch
import numpy as np

def constructdelta(y,K,margin = 1.0):
    a = y.cpu().numpy()
    # c = [v[0] for v in a]
    b = margin*np.ones( ( a.size, K ) )
    b[ np.arange(a.size), a ] = 0.0
    return torch.from_numpy(b).float().cuda()


# Loss Function -- See https://github.com/sauptikdhar/DOC3/blob/main/Table2/CIFAR10_DOC3.ipynb
class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    
    def forward(self, ypred, ytrue, margin = 1.0, smooth = False, nonconvex = False):
        ypred = ypred.squeeze()
        if nonconvex:
            out = torch.mean(torch.clamp(margin - (ytrue * ypred), min = 0.0, max = margin))
            out = 2*out - 1
            return out
        if smooth:
            loss = torch.nn.Softplus()
            out = torch.mean(loss(margin - (ytrue * ypred)))
        else:
            out = torch.mean(torch.relu(margin - (ytrue * ypred)))
        return out


def constructCosts(y,C_ratio):
    y = y.cpu().numpy()
    Cwts = []
    for l in y:
        Cwts.append(C_ratio[l]) 
    Cwts = torch.tensor(Cwts).float().cuda()
    return Cwts



class MCHingeLoss(torch.nn.Module): # The C&S loss on Training Samples
    def __init__(self):
        super(MCHingeLoss,self).__init__()
    
    def forward(self,ypred,ytrue,K, margin = 1.0, nonconvex = False, C_ratio = []):
        if len(C_ratio) == 0: C_ratio = [1. for _ in range(K)]
        assert len(C_ratio) == K

        ytrue = ytrue.long()
        ypred_k = ypred.gather(1, ytrue.view(-1,1))
        e_il = constructdelta(ytrue,K,margin)
        loss_per_sample,_ = torch.max(e_il + ypred, dim = 1)
        if margin > 0.0 and nonconvex:
            loss_per_sample = torch.clamp(loss_per_sample - ypred_k.flatten(), min = 0.0, max = margin)
        else:
            loss_per_sample = loss_per_sample - ypred_k.flatten()
        
        Cwts = constructCosts(ytrue,C_ratio)
        loss_per_sample = Cwts*loss_per_sample/K # Scale
        loss = torch.mean(loss_per_sample) 
        if nonconvex:
            loss = loss - ((K-1)/K**2)
        return loss 




class QRloss(torch.nn.Module): # The C&S loss on Training Samples
    def __init__(self):
        super(QRloss,self).__init__()
    
    def forward(self,query,feature, rating):
        rating = rating.reshape(-1)
        return torch.mean(rating*torch.norm(query-feature,dim =1))


class FMContrastiveSim(torch.nn.Module): 
    def __init__(self):
        super(FMContrastiveSim,self).__init__()
    
    def forward(self, feature_a, feature_b, rating, level = 0., maxlevel=3., margin = 0.3): # level = 0,1,2, maxlevel = len(level) - level 0 = Barrel vs belly etc.
        sim = cosine_similarity(feature_a, feature_b)
        C_pos = 1/(level+1)
        C_neg = 1/(maxlevel-level) #1/(level+1)
        score = (C_pos/2)*(1+rating)*(1.-sim) + (C_neg/2)*(1-rating)*torch.relu(sim-margin)
        return torch.mean(score)
