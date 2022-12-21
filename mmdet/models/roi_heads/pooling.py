import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SpatialAttention(nn.Module):

    def __init__(self, insize, outsize):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding='same')
        self.proj = nn.Linear(insize, outsize)
        self.tanh = nn.Tanh()
    
    def forward(self, feature_maps):
        # input batch_size, channel, w, h
        meanp = feature_maps.mean(dim=1, keepdim=True)
        maxp, _ = feature_maps.max(dim=1, keepdim=True)
        cp = torch.cat([meanp, maxp], dim=1)

        att = self.conv(cp).reshape(feature_maps.size(0), -1)
        att = torch.softmax(att, dim=-1)
        att = att.reshape(feature_maps.size(0), 1, feature_maps.size(2), feature_maps.size(3))

        agg = torch.sum((feature_maps * att).reshape(feature_maps.size(0), feature_maps.size(1), -1), dim=-1)
        agg = self.tanh(self.proj(agg))
        return agg

class MultiStageGeM(nn.Module):

    def __init__(self, insize, outsize, ptype='GeM') -> None:
        super(MultiStageGeM, self).__init__()
        if ptype == 'GeM':
            self.pooling = GeMPooling(insize, outsize)
        elif ptype == 'MEX':
            self.pooling = MEXPooling(insize, outsize)
        elif ptype == 'AlphaMEX':
            self.pooling = AlphaMEXPooling(insize, outsize)
        elif ptype == 'adaptiveS':
            # only for single pooling
            self.pooling = AdaptivePooling(insize, outsize)
        elif ptype == 'adaptive':
            self.pooling = AdaptivePooling(insize, outsize, proj=True)
        
    def forward(self, x):
        # x should be B C H*W
        # C should be equal to insize
        return self.pooling(x)

class AdaptivePooling(nn.Module):

    def __init__(self, insize, outsize, proj=False) -> None:
        super(AdaptivePooling, self).__init__()
        if proj:
            self.proj = nn.Linear(insize, outsize)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.p = proj
    
    def forward(self, x):
        
        x = self.pooling(x)
        x = x.squeeze(-1)
        if self.p:
            x = self.proj(x)

        return x

class GeMPooling(nn.Module):

    def __init__(self, insize, outsize) -> None:
        super(GeMPooling, self).__init__()
        self.proj = nn.Linear(insize, outsize)
        self.p = nn.Parameter(torch.Tensor([3]))
        self.minimumx = nn.Parameter(torch.Tensor([1e-6]), requires_grad=False)
    
    def forward(self, x):
        xpower = torch.pow(torch.maximum(x, self.minimumx), self.p)
        gem = torch.pow(xpower.mean(dim=-1, keepdim=False), 1.0 / self.p)
        gem = self.proj(gem)
        return gem

class MEXPooling(nn.Module):

    def __init__(self, insize, outsize) -> None:
        super(MEXPooling, self).__init__()
        self.proj = nn.Linear(insize, outsize)
        self.beta = nn.Parameter(torch.FloatTensor([3.0]))
    
    def forward(self, x):
        # xexp = torch.exp(self.beta * x)
        # logExp = torch.log(xexp.mean(dim=-1, keepdim=False)) * (1.0 / self.beta)
        # original was commented, the new one has stable gradient as logsumexp
        logExp = (torch.logsumexp(self.beta * x, dim=-1) + math.log(1 / x.size(-1)))  * (1.0 / self.beta)
        logExp = self.proj(logExp)
        return logExp

class AlphaMEXPooling(nn.Module):

    def __init__(self, insize, outsize) -> None:
        super(AlphaMEXPooling, self).__init__()
        self.proj = nn.Linear(insize, outsize)
        self.alpha = nn.Parameter(torch.FloatTensor([1.0]))
        self.minimumx = nn.Parameter(torch.Tensor([1e-6]), requires_grad=False)
    
    def forward(self, x):
        # alpha = torch.sigmoid(self.alpha)
        # base = alpha / (1 - alpha)
        # logPow = torch.log(basePow.mean(dim=-1, keepdim=False)) * (1.0 / torch.log(base))
        # logPow = self.proj(logPow)
        # change to more stable one, actually beta in MEX is ln(alpha / (1 - alpha))
        alpha = torch.sigmoid(self.alpha)
        base = torch.log(alpha / (1 - alpha + self.minimumx))
        logExp = (torch.logsumexp(base * x, dim=-1) + math.log(1 / x.size(-1)))  * (1.0 / torch.log(base))
        logExp = self.proj(logExp)
        return logExp

class SelfAttend(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.h1(seqs).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output