from torch import nn
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm

from TwoWL.utils import *


class LocalWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_node_feat,
                 node_feat,
                 channels_1wl=256,
                 channels_2wl=32,
                 depth1=1,
                 depth2=1,
                 dp_lin0=0.7,
                 dp_lin1=0.7,
                 dp_emb=0.5,
                 dp_1wl0=0.5,
                 dp_2wl=0.5,
                 dp_1wl1=0.5,
                 act0=True,
                 act1=True,
                 ):
        super().__init__()

        use_affine = False

        relu_lin = lambda a, b, dp, lnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())

        relu_conv = lambda insize, outsize, dp, act: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])

        self.max_x = max_x
        self.use_node_feat = use_node_feat
        self.node_feat = node_feat
        if use_node_feat:
            self.lin1 = nn.Sequential(
                nn.Dropout(dp_lin0),
                relu_lin(node_feat.shape[-1], channels_1wl, dp_lin1, True, False)
            )
        else:
            self.emb = nn.Sequential(nn.Embedding(max_x + 1, channels_1wl),
                                     GraphNorm(channels_1wl),
                                     Dropout(p=dp_emb, inplace=True))

        self.conv1s = nn.ModuleList(
            [relu_conv(channels_1wl, channels_1wl, dp_1wl0, act0) for _ in range(depth1 - 1)] +
            [relu_conv(channels_1wl, channels_2wl, dp_1wl1, act1)])

        self.conv2s = nn.ModuleList(
            [relu_conv(channels_2wl, channels_2wl, dp_2wl, True) for _ in range(depth2)])
        self.conv2s_r = nn.ModuleList(
            [relu_conv(channels_2wl, channels_2wl, dp_2wl, True) for _ in range(depth2)])
        self.pred = nn.Linear(channels_2wl, 1)

    def forward(self, x, edge1, pos, idx=None, ei2=None, test=False):
        edge2, edge2_r = reverse(ei2)

        x = self.lin1(self.node_feat) if self.use_node_feat else self.emb(x).squeeze()
        for conv1 in self.conv1s:
            x = conv1(x, edge1)

        x = x[pos[:, 0]] * x[pos[:, 1]]
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)
        x = x[idx]
        mask = torch.cat(
            [torch.ones([1, x.shape[0] // 2], dtype=bool),
             torch.zeros([1, x.shape[0] // 2], dtype=bool)]).t().reshape(-1)
        x = x[mask] * x[~mask]
        x = self.pred(x)
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
    # C = f(A)
    n, d = A.shape[0], A.shape[1]
    vec_p = (torch.sum(B, dim=1, keepdim=True)).expand(-1, n, -1)
    vec_q = (torch.sum(B, dim=0, keepdim=True)).expand(n, -1, -1)
    D = torch.cat([A, vec_p, vec_q], -1)
    return g(D)
