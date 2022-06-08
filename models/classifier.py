import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
from dgl.dataloading import GraphDataLoader
import numpy as np
import torch as t
import os
from utils.config import *
from utils.FocalLoss import *
from models.MLP import *
import random

params = config()

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

seed_array=[983, 523, 697, 689, 523]    # Set model seed

    
class BaseClassifier:
    def __init__(self):
        pass

    def cv_train(self, dataset, batchSize=32, num_epochs=10, lr=0.001, kFold=5, savePath='checkpoints/', earlyStop=100,
                 seed=10,device=t.device('cpu')):
        splits = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=10)
        fold_best = []
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset[:][0],dataset[:][1])):
            savePath2 = savePath + f"fold{fold + 1}"
            setup_seed(seed_array[fold])
            self.reset_parameters()
            best_f = 0.0
            print('>>>>>>Fold{}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)

            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = GraphDataLoader(dataset, batch_size=batchSize, sampler=train_sampler, num_workers=4)
            test_loader = GraphDataLoader(dataset, batch_size=batchSize, sampler=test_sampler, num_workers=4)
            optimizer = t.optim.Adam(self.moduleList.parameters(), lr=lr)
            # lossfn = nn.CrossEntropyLoss()
            lossfn = FocalLoss(gamma=2)
            best_record = {'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'train_f': 0,
                           'test_f': 0, 'train_pre': 0, 'test_pre': 0, 'train_rec': 0, 'test_rec': 0,
                           'train_roc': 0, 'test_roc': 0}
            nobetter = 0

            for epoch in range(num_epochs):
                train_loss, train_acc, train_f, train_pre, train_rec, train_roc = self.train_epoch(train_loader,
                                                                                                   lossfn,
                                                                                                   optimizer,
                                                                                                   device)

                test_loss, test_acc, test_f, test_pre, test_rec, test_roc = self.valid_epoch(test_loader, lossfn,
                                                                                             device)

                print(
                    ">>>Epoch:{} of Fold{} AVG Train Loss:{:.3f}, AVG Test Loss:{:.3f}\n"
                    "Train Acc:{:.3f} %, Train F1-score:{:.3f}, Train Precision:{:.3f}, Train Recall:{:.3f}, Train ROC:{:.3f};\n"
                    "Test Acc:{:.3f} %, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}!!!\n".format(
                        epoch + 1, fold + 1, train_loss, test_loss,
                        train_acc * 100, train_f, train_pre, train_rec, train_roc,
                        test_acc * 100, test_f, test_pre, test_rec, test_roc))

                if best_f < test_f:
                    nobetter = 0
                    best_f = test_f
                    best_record['train_loss'] = train_loss
                    best_record['test_loss'] = test_loss
                    best_record['train_acc'] = train_acc
                    best_record['test_acc'] = test_acc
                    best_record['train_f'] = train_f
                    best_record['test_f'] = test_f
                    best_record['train_pre'] = train_pre
                    best_record['test_pre'] = test_pre
                    best_record['train_rec'] = train_rec
                    best_record['test_rec'] = test_rec
                    best_record['train_roc'] = train_roc
                    best_record['test_roc'] = test_roc
                    print(f'>Bingo!!! Get a better Model with valid F1-score: {best_f:.3f}!!!')

                    self.save("%s.pkl" % savePath2, epoch + 1, best_f)
                else:
                    nobetter += 1
                    if nobetter >= earlyStop:
                        print(
                            f'Test F1-score has not improved for more than {earlyStop} steps in epoch {epoch + 1}, stop training.')
                        break
            fold_best.append(best_record)
            print(f'*****The Fold{fold + 1} is done!')
            print(
                "Finally,the best model's Test Acc:{:.3f} %, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}!!!\n\n\n".format(
                    best_record['test_acc'] * 100, best_record['test_f'], best_record['test_pre'],
                    best_record['test_rec'], best_record['test_roc']))
            os.rename("%s.pkl" % savePath2,
                      "%s_%s.pkl" % (savePath2, ("%.3f" % best_f)))
        # Print the result of 5 folds
        print('*****All folds are done!')
        print("=" * 20 + "FINAL RESULT" + "=" * 20)
        # Print table header
        row_first = ["Fold", "ACC", "F1-score", "Precision", "Recall", "ROC"]
        print("".join(f"{item:<12}" for item in row_first))
        # Print table content
        metrics = ['test_f', 'test_pre', 'test_rec', 'test_roc']
        for idx, fold in enumerate(fold_best):
            print(f"{idx + 1:<12}" + "%-.3f" % (fold['test_acc'] * 100) + "%-6s" % "%" + "".join(
                f"{fold[key]:<12.3f}" for key in metrics))
        # Print average
        avg, metrics2 = {}, ['test_acc', 'test_f', 'test_pre', 'test_rec', 'test_roc']
        for item in metrics2:
            avg[item] = 0
            for fold in fold_best:
                avg[item] += fold[item]
            avg[item] /= len(fold_best)
        print(f"%-12s" % "Average" + "%-.3f" % (avg['test_acc'] * 100) + "%-6s" % "%" + "".join(
            f"{avg[key]:<12.3f}" for key in metrics))
        print("=" * 52)

    def train_epoch(self, dataloader, loss_fn, optimizer, device):
        train_loss, train_acc, train_f, train_pre, train_rec, train_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.to_train_mode()
        pred_list = []
        label_list = []
        for _, (batched_graph, labels) in enumerate(dataloader):
            batched_graph, labels = batched_graph.to(device), labels.to(device)
            feats = batched_graph.ndata['attr']

            optimizer.zero_grad()
            output = self.calculate_y(batched_graph, feats)
            output, labels = output.to(t.float32), labels.to(t.int64)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
			
            # pred_list loads the predicted values of the training set samples
            pred_list.extend(output.detach().cpu().numpy())

            # label_list loads the true values of the training set samples (one-hot form)
            label_list.extend(labels.cpu().numpy())
        
        with t.no_grad():
            pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
            train_loss = loss_fn(pred_tensor, label_tensor)
        pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
        train_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
        train_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
        train_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro',
                                    zero_division=0)
        train_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
        train_roc = roc_auc_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                  pred_array, average='micro')
        return train_loss, train_acc, train_f, train_pre, train_rec, train_roc

    def valid_epoch(self, dataloader, loss_fn, device):
        with t.no_grad():
            valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.to_eval_mode()
            pred_list = []
            label_list = []
            for _, (batched_graph, labels) in enumerate(dataloader):
                batched_graph, labels = batched_graph.to(device), labels.to(device)
                feats = batched_graph.ndata['attr']
                output = self.calculate_y(batched_graph, feats)
                output, labels = output.to(t.float32), labels.to(t.int64)
                
                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())

                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())
            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)
            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
            valid_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
            valid_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
            valid_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro',
                                        zero_division=0)
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
            valid_roc = roc_auc_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                      pred_array, average='micro')
        return valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for idx, module in enumerate(self.moduleList):
            stateDict[idx] = module.state_dict()
        t.save(stateDict, path)
        print('Model saved in "%s".\n' % path)

    def load(self, path, map_location=None):
        parameters = t.load(path, map_location=map_location)
        for idx, module in enumerate(self.moduleList):
            module.load_state_dict(parameters[idx])
        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()


class GraphClassifier(BaseClassifier):
    def __init__(self, in_dim, hidden_dim, n_classes, device=t.device("cpu")):
        super(GraphClassifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        self.classify = MLP(inSize = hidden_dim, outSize = n_classes).to(device)
        self.moduleList = nn.ModuleList([self.conv1, self.conv2, self.classify])
        self.device = device

    def calculate_y(self, g, h):
        # Apply graph convolution networks and activation functions
        h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
        with g.local_scope():
            g.ndata['h'] = h
            # Use the average readout to get the graph representation
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)
