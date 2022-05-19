# GraphLncLoc
A graph convolutional network-based lncRNA subcellular localization predictor

# Requirements
    torch==1.9.1  
    dgl-cu102==0.7.2  
    scikit-learn==1.0.1  
    numpy==1.20.3  
    gensim==4.1.2  
    tqdm==4.62.3  

# Usage
## Simple usage
You can train the model in a very simple way by the command blow:
``python train.py >output/log.txt 2>&1`` 
## How to train your own model
Also you can use the package provided by us to train your model.

First, you need to import the package.  
```python
from data.LncRNADataset import *
from model.classifier import *
from utils.config import *
```
Second, you can train your own model by modifying the variables in *utils/config.py*, and create a configuration object by the command blow: 
```python
params = config()
```  

>In the *utils/config.py*, the meaning of the variables is explained as follows:
>>***k*** is the value of the k-mer nodes.  
>>***d*** is the dimension of vector of node features which are trained by gensim library.  
>>***hidden_dim*** is the parameters of the hidden layer of GCNs.  
>>***n_classes*** is the number of sample categories.  
>>***savePath*** is the folder where the model is saved.  
>>***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.  

Then you need to create the data object.  
```python
dataset = LncRNADataset(raw_dir='data/data.txt', save_dir=f'checkpointslgraph/k{params.k}_d{params.d}')
```
Finally, you can create the model object and start training.
```python
model = GraphClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes, device=params.device)
model.cv_train(dataset, batchSize=params.batchSize, num_epochs=params.num_epochs, lr=params.lr, kFold=params.kFold, savePath=params.savePath, device=params.device)
```

## How to do prediction
First, import the package. 
```python
from model.lncRNA_lib import *
```
Then instantiate  five model objects.
```python
mapLocation={'cuda:0':'cpu', 'cuda:1':'cpu'}
GraphLncLoc =[]
for i in range(1,6):
    GraphLncLoc.append(lncRNALocalizer(f"checkpoints/Final_model/fold{i}.pkl", map_location=mapLocation))
```
Finally, the prediction results of the five models were voted to get the final prediction results.
```python
def lncRNA_loc_predict(lncRNA):
    return vote_predict(GraphLncLoc, lncRNA.upper())

if __name__=="__main__":
    sequence="ACC...UCU"
    print(lncRNA_loc_predict(sequence))
```
## Independent test set
The *test_set.txt* in *Independent_test_set* folder is used in comparison with other existing predictors. 

## Other details
The other details can be seen in the paper and the codes.

# Citation
Min Li, Baoying Zhao, Rui Yin, Chengqian Lu, Min Zeng. GraphLncLoc: long non-coding RNA subcellular locali-zation prediction using graph convolutional networks based on sequence to graph transformation

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
