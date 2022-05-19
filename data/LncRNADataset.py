import os
import pickle
from collections import Counter
import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
import numpy as np
from dgl.data.utils import save_info, load_info
from dgl.nn.pytorch import EdgeWeightNorm
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from utils.config import *

params = config()


class LncRNADataset(DGLDataset):
    """
        url : str
            The url to download the original dataset.
        raw_dir : str
            Specifies the directory where the downloaded data is stored or where the downloaded data is stored. Default: ~/.dgl/
        save_dir : str
            The directory where the finished dataset will be saved. Default: the value specified by raw_dir
        force_reload : bool
            If or not to re-import the dataset. Default: False
        verbose : bool
            Whether to print progress information.
        """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(LncRNADataset, self).__init__(name='lncrna',
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose
                                            )
        print('***Executing init function***')
        print('Dataset initialization is completed!\n')

    def process(self):
        # Processing of raw data into plots, labels
        print('***Executing process function***')
        self.kmers = params.k
        # Open files and load data
        print('Loading the raw data...')
        with open(self.raw_dir, 'r') as f:
            data = []
            for i in tqdm(f):
                data.append(i.strip('\n').split('\t'))

        # Get labels and k-mer sentences
        k_RNA, rawLab = [[i[1][j:j + self.kmers] for j in range(len(i[1]) - self.kmers + 1)] for i in data], [i[2] for i
                                                                                                              in data]

        # Get the mapping variables for label and label_id
        print('Getting the mapping variables for label and label id...')
        self.lab2id, self.id2lab = {}, []
        cnt = 0
        for lab in tqdm(rawLab):
            if lab not in self.lab2id:
                self.lab2id[lab] = cnt
                self.id2lab.append(lab)
                cnt += 1
        self.classNum = cnt

        # Get the mapping variables for kmers and kmers_id
        print('Getting the mapping variables for kmers and kmers id...')
        self.kmers2id, self.id2kmers = {"<EOS>": 0}, ["<EOS>"]
        kmersCnt = 1
        for rna in tqdm(k_RNA):
            for kmers in rna:
                if kmers not in self.kmers2id:
                    self.kmers2id[kmers] = kmersCnt
                    self.id2kmers.append(kmers)
                    kmersCnt += 1
        self.kmersNum = kmersCnt

        # Get the ids of RNAsequence and label
        self.k_RNA = k_RNA
        self.labels = t.tensor([self.lab2id[i] for i in rawLab])
        self.idSeq = np.array([[self.kmers2id[i] for i in s] for s in self.k_RNA], dtype=object)

        self.vectorize()

        # Construct and save the graph
        self.graphs = []
        for eachseq in self.idSeq:
            newidSeq = []
            old2new = {}
            count = 0
            for eachid in eachseq:
                if eachid not in old2new:
                    old2new[eachid] = count
                    count += 1
                newidSeq.append(old2new[eachid])
            counter_uv = Counter(list(zip(newidSeq[:-1], newidSeq[1:])))
            graph = dgl.graph(list(counter_uv.keys()))
            weight = t.FloatTensor(list(counter_uv.values()))
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph, weight)
            graph.edata['weight'] = norm_weight
            node_features = self.vector['embedding'][list(old2new.keys())][:]
            graph.ndata['attr'] = t.tensor(node_features)
            self.graphs.append(graph)
            

    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graphs)

    def save(self):
        # Save the processed data to `self.save_path`
        print('***Executing save function***')
        save_graphs(self.save_dir + ".bin", self.graphs, {'labels': self.labels})
        # Save additional information in the Python dictionary
        info_path = self.save_dir + "_info.pkl"
        info = {'kmers': self.kmers, 'kmers2id': self.kmers2id, 'id2kmers': self.id2kmers, 'lab2id': self.lab2id,
                'id2lab': self.id2lab}
        save_info(info_path, info)

    def load(self):
        # Import processed data from `self.save_path`
        print('***Executing load function***')
        self.graphs, label_dict = load_graphs(self.save_dir + ".bin")
        self.labels = label_dict['labels']
        info_path = self.save_dir + "_info.pkl"
        info = load_info(info_path)
        self.kmers, self.kmers2id, self.id2kmers, self.lab2id, self.id2lab = info['kmers'], info['kmers2id'], info[
            'id2kmers'], info['lab2id'], info['id2lab']

    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        print('***Executing has_cache function***')
        graph_path = self.save_dir + ".bin"
        info_path = self.save_dir + "_info.pkl"
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def vectorize(self, method="word2vec", feaSize=params.d, window=5, sg=1,
                  workers=8, loadCache=True):
        self.vector = {}
        print('\n***Executing vectorize function***')
        if os.path.exists(f'checkpoints/Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl') and loadCache:
            with open(f'checkpoints/Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl', 'rb') as f:
                if method == 'kmers':
                    tmp = pickle.load(f)
                    self.vector['encoder'], self.kmersFea = tmp['encoder'], tmp['kmersFea']
                else:
                    self.vector['embedding'] = pickle.load(f)
            print(f'Load cache from checkpoints/Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl!')
            return
        if method == 'word2vec':
            doc = [i + ['<EOS>'] for i in self.k_RNA]
            model = Word2Vec(doc, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
            word2vec = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            for i in range(self.kmersNum):
                word2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = word2vec
        elif method == 'kmers':
            enc = OneHotEncoder(categories='auto')
            enc.fit([[i] for i in self.kmers2id.values()])
            feaSize = len(self.kmers2id)-1
            kmers = np.zeros((len(self.labels), feaSize))
            bs = 50000
            print('Getting the kmers vector...')
            for i, t in enumerate(self.idSeq):
                for j in range((len(t) + bs - 1) // bs):
                    kmers[i] += enc.transform(np.array(t[j * bs:(j + 1) * bs]).reshape(-1, 1)).toarray().sum(
                        axis=0)
            kmers = kmers[:, 1:]
            feaSize -= 1
            # Normalized
            kmers = (kmers - kmers.mean(axis=0)) / kmers.std(axis=0)
            self.vector['encoder'] = enc
            self.kmersFea = kmers

        # Save k-mer vectors
        with open(f'checkpoints/Node_feature/{method}_k{self.kmers}_d{feaSize}.pkl', 'wb') as f:
            if method == 'kmers':
                pickle.dump({'encoder': self.vector['encoder'], 'kmersFea': self.kmersFea}, f, protocol=4)
            else:
                pickle.dump(self.vector['embedding'], f, protocol=4)

