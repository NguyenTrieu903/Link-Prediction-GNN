from torch import nn
import torch
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv, APPNP
# from utils import reverse, sparse_bmm, sparse_cat, add_zero, edge_list
import time
from TwoWL.utils import *

class WLNet(torch.nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 layer1=2,
                 layer2=1,
                 layer3=1,
                 dp0_0 = 0.0,
                 dp0_1 = 0.0,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 ln0=True,
                 ln1=True,
                 ln2=True,
                 ln3=True,
                 ln4=True,
                 act0=False,
                 act1=False,
                 act2=False,
                 act3=True,
                 act4=True,
                 ):
        super(WLNet, self).__init__()

        self.use_feat = use_feat
        self.feat = feat
        use_affine = False

        relu_lin = lambda a, b, dp, lnx, actx: Seq([
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()])
        if feat is not None:
            self.lin1 = nn.Sequential(
                nn.Dropout(dp0_0),
                relu_lin(feat.shape[1], hidden_dim_1, dp0_1, ln0, act0)
            )

        Convs = lambda a, b, dp, lnx, actx: Seq([
            SAGEConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()])

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(max_x + 1, hidden_dim_1),
                                             torch.nn.Dropout(p=dp1))
        #self.embedding = nn.Embedding(max_x + 1, latent_size_1)

        self.nconvs = nn.ModuleList([Convs(hidden_dim_1, hidden_dim_1, dp2, ln1, act1)] +
                                    [Convs(hidden_dim_1, hidden_dim_1, dp2, ln2, act2) for _ in range(layer1 - 1)]
                                    )

        input_edge_size = hidden_dim_1

        self.h_1 = Seq([relu_lin(input_edge_size + 1, hidden_dim_2, dp3, ln3, act3)] +
                       [relu_lin(hidden_dim_2, hidden_dim_2, dp3, ln3, act3) for _ in range(layer2 - 1)])

        self.g_1 = Seq([relu_lin(hidden_dim_2 * 2 + input_edge_size + 1, hidden_dim_2, dp3, ln4, act4)] +
                       [relu_lin(hidden_dim_2, hidden_dim_2, dp3, ln4, act4) for _ in range(layer3 - 1)])

        self.lin_dir = torch.nn.Linear(hidden_dim_2, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, ei, pos, ei2=None, test=False):
        edge_index = ei
        n = x.shape[0]
        if self.use_feat:
            x = self.feat
            x = self.lin1(x)
        else:
            x = self.embedding(x)
        # x = F.relu(self.nlin1(x))

        for conv in self.nconvs:
            x = conv(x, edge_index)
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        x = x.reshape(n, n, -1)
        eim = torch.zeros((n * n,), device=x.device)
        eim[edge_index[0] * n + edge_index[1]] = 1
        eim = eim.reshape(n, n, 1)
        x = torch.cat((x, eim), dim=-1)
        x = mataggr(x, self.h_1, self.g_1)
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1)
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x

class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

def mataggr(A, h, g):
    '''
    A (n, n, d). n is number of node, d is latent dimension
    h, g are mlp
    '''
    B = h(A)
    #C = f(A)
    n, d = A.shape[0], A.shape[1]
    vec_p = (torch.sum(B, dim=1, keepdim=True)).expand(-1, n, -1)
    vec_q = (torch.sum(B, dim=0, keepdim=True)).expand(n, -1, -1)
    D = torch.cat([A, vec_p, vec_q], -1)
    return g(D)
