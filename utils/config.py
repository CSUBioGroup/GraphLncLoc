import torch as t


# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        # k value of k-mer
        self.k = 4
        # Dimension of word2vec word vector, i.e. node feature dimension
        self.d = 128
        # Parameters of the hidden layer of the graph convolutional networks
        self.hidden_dim = 64
        # Number of sample categories
        self.n_classes = 4
        # Set random seeds
        self.seed = 10
        # Training parameters
        self.batchSize = 8
        self.num_epochs = 1000
        self.lr = 0.003
        self.kFold = 5
        self.savePath = f"checkpoints/dglmodel/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}/"
        self.device = t.device("cuda:0")

    def set_seed(self,s):
        self.seed=s
        self.savePath = f"checkpoints/dglmodel/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}_s{self.seed}/"
