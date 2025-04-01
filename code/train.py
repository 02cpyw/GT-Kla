
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

from gihub_code.code.models.GTKla import GTKla
root_dir = "root_dir"
model_info = "GNNcor_CLAD"
results_type = "results_test"
data_dir = root_dir + "/data/my_datasets_41_cont"
seq_len = 41
results_dir = root_dir + f"/papers/{results_type}/{model_info}.csv"
pths_dir = root_dir + "/code/pths_test"
Epoch = 200
logging.basicConfig(
    filename=root_dir + '/code/logs/{}_{}.log'.format(datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),model_info),  # 日志保存的文件路径
    level=logging.INFO,  # 设置日志记录的级别（INFO及以上的级别会记录）
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志输出的格式
)
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
        """将蛋白质序列编码为整数序列"""
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
    
    train_dataset = cor_datasets(data_dir + f'/train/train_{fold+1}.fasta')  # 训练数据文件
    test_dataset = cor_datasets(data_dir + f'/test.fasta')  # 测试数据文件
    train_seqs = getSeqs(data_dir + f'/train/train_{fold+1}.fasta')
    test_seqs = getSeqs(data_dir + f'/test.fasta')
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    vocab_size = 22  
    d_model = 256
    model = GTKla(vocab_size, d_model, nhead=4, num_layers=4, max_len=seq_len).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵

    # 训练循环
    results = {"ACC":0,"MCC":0,"AUROC":0,"PRE":0,"SEN":0,"SPE":0}
    best_model = copy.deepcopy(model)
    for epoch in range(Epoch):
        model.train()
        total_loss = 0
        all_embs = []
        y_true = []
        progress_bar = tqdm(train_dataloader, desc=f"Fold {fold+1}/{Fold} Epoch {epoch + 1}/{Epoch}")
        att_matrix,diff_att2 = [],[]
        for batch_idx, (data, target, seq_index) in enumerate(progress_bar):
            data, target,seq_index = data.cuda(), target.cuda(), seq_index.cuda()
            optimizer.zero_grad()
            probs,embs,att_matrix,att = model(data,seq_index,train_seqs)
            y_true.extend(target.detach().cpu().numpy())
            all_embs.extend(embs.detach().cpu().numpy())
            loss = criterion(probs, target.float())  # 计算损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        print(f'average_loss: {total_loss / len(train_dataloader):.4f}')