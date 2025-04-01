
import pytz
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import copy

from models.GTKla import GNNcor_CLAD
root_dir = "root_dir"
model_info = "GNNcor_CLAD"
results_type = "results_test"
data_dir = root_dir + "/data/my_datasets_41_cont"
seq_len = 41
results_dir = root_dir + f"/papers/{results_type}/{model_info}.csv"
pths_dir = root_dir + "/code/pths_test"
Epoch = 200
def getSeqs(file_path):
    seqs = []
    with open(file_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                continue
            seqs.append(line)
    return seqs
class cor_datasets(Dataset):
    def __init__(self, file_path):
        self.seqs = []
        self.labels = []
        self.data = []
        self.seq_index = []

        with open(file_path,"r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    self.labels.append(int(line[1:]))
                else:
                    self.seqs.append(line)
                    self.data.append(self.encode_sequence(line))
        
        self.seq_index = [i for i in range(len(self.seqs))]
    
    def encode_sequence(self, sequence):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWYXU'  
        aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
        return [aa_to_idx[aa] for aa in sequence]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.data[idx])), torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(self.seq_index[idx]))

Fold = 1
begin = 0
sum_res = {"ACC":0,"MCC":0,"AUROC":0,"PRE":0,"SEN":0,"SPE":0}
for fold in range(begin,begin+Fold):
    
    test_dataset = cor_datasets(data_dir + f'/test.fasta')  # 测试数据文件
    test_seqs = getSeqs(data_dir + f'/test.fasta')
    batch_size = 128
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    vocab_size = 22  
    d_model = 256
    model = torch.load("model_dir")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵

    # 训练循环
    results = {"ACC":0,"MCC":0,"AUROC":0,"PRE":0,"SEN":0,"SPE":0}
    best_model = copy.deepcopy(model)
    for epoch in range(Epoch):
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        with torch.no_grad():
            for data, target, seq_index in test_dataloader:
                data, target,seq_index = data.cuda(), target.cuda(), seq_index.cuda()
                probs,embs,att_matrix,att = model(data,seq_index,test_seqs)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend((probs > 0.5).cpu().numpy()) 
                y_prob.extend(probs.cpu().numpy())

            accuracy = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_prob)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            print(f'ACC: {accuracy:.4f}')
            print(f'MCC: {mcc:.4f}')
            print(f'AUROC: {roc_auc:.4f}')
