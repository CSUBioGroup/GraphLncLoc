
# Requirements
    torch==1.9.1  
    dgl-cu102==0.7.2  
    scikit-learn==1.0.1  
    numpy==1.20.3  
    gensim==4.1.2  
    tqdm==4.62.3  

# Usage
## How to do prediction
In the following, take the four-category (cytoplasm, nucleus, ribosome, exosome) prediction task as an example:

First, import the package. 
```python
from model.lncRNA_lib import *
```
Then instantiate  five model objects.
```python
mapLocation={'cuda:0':'cpu', 'cuda:1':'cpu'}
GraphLncLoc =[]
for i in range(1,6):
    GraphLncLoc.append(lncRNALocalizer(f"checkpoints/Final_model/fold{i}_5.pkl", n_classes=5, map_location=mapLocation))
```
Finally, the prediction results of the five models were voted to get the final prediction results.
```python
def lncRNA_loc_predict(lncRNA):
    return vote_predict(GraphLncLoc, lncRNA.upper())

if __name__=="__main__":
    sequence="ACC...UCU"
    print(lncRNA_loc_predict(sequence))
```
Note: 
1. You can run *predict_5*.py to perform the five-category (cytoplasm, nucleus, ribosome, exosome, cytosol) prediction task.
2. You can run *predict_dc.py* to perform the four-category (cytoplasm, nucleus, ribosome, exosome) prediction task. In particular, we treat lncRNA sequences from cytosol as if they were from cytoplasm.
