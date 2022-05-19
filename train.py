from data.LncRNADataset import *
from models.classifier import *
from utils.config import *
import datetime

starttime = datetime.datetime.now()

params = config()




dataset = LncRNADataset(raw_dir='data/data.txt', save_dir=f'checkpoints/dglgraph/k{params.k}_d{params.d}')
model = GraphClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes, device=params.device)

model.cv_train(dataset, batchSize=params.batchSize,
               num_epochs=params.num_epochs,
               lr=params.lr,
               kFold=params.kFold,
               savePath=params.savePath,
               device=params.device
               )

endtime = datetime.datetime.now()
print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')
