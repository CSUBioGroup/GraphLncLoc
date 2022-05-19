from torch.nn import functional as F
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch
from .MLP import *
import sys



class lncRNALocalizer:
    def __init__(self, weightPath, in_dim=128, hidden_dim=64, n_classes=4, map_location=None,
                 device=torch.device("cpu")):
        stateDict = torch.load(weightPath, map_location=map_location)
        self.id2lab = ['Cytoplasm', 'Nucleus', 'Exosome', 'Ribosome']
        self.kmers2id = np.load(r'data/lncRNA_lib_data/kmers2id.npy', allow_pickle=True).item()
        self.node_features = np.load(r'data/lncRNA_lib_data/node_features.npy', allow_pickle=True)
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        self.classify = MLP(inSize=hidden_dim, outSize=n_classes).to(device)
        self.moduleList = nn.ModuleList([self.conv1, self.conv2, self.classify])

        for idx, module in enumerate(self.moduleList):
            module.load_state_dict(stateDict[idx])
            module.eval()

        self.device = device

    def predict(self, RNA):
        graph = self.transform(RNA)
        graph = graph.to(self.device)
        g, h = graph, graph.ndata['attr']
        h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            res = self.classify(hg)
            return {k: v for k, v in zip(self.id2lab, F.softmax(res, dim=1).data.numpy()[0])}

    def item2graph(self, idseq, embedding):
        newidSeq = []
        old2new = {}
        count = 0
        for eachid in idseq:
            if eachid not in old2new:
                old2new[eachid] = count
                count += 1
            newidSeq.append(old2new[eachid])
        counter_uv = Counter(list(zip(newidSeq[:-1], newidSeq[1:])))
        graph = dgl.graph(list(counter_uv.keys()))
        weight = torch.FloatTensor(list(counter_uv.values()))
        norm = dglnn.EdgeWeightNorm(norm='both')
        norm_weight = norm(graph, weight)
        graph.edata['weight'] = norm_weight
        node_features = embedding[list(old2new.keys())][:]
        graph.ndata['attr'] = torch.tensor(node_features)
        return graph

    def transform(self, RNA):
        RNA = ''.join([i if i in 'ATCG' else '' for i in RNA.replace('U', 'T')])
        idseq = [self.kmers2id[RNA[i:i + 4]] for i in range(len(RNA) - 4 + 1)]
        return self.item2graph(idseq, self.node_features)


from collections import Counter


def vote_predict(localizers, RNA):
    num = len(localizers)
    res = Counter({})
    for localizer in localizers:
        res += Counter(localizer.predict(RNA))
    return {k: res[k] / num for k in res.keys()}
