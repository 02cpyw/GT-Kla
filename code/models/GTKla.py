import math
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, GraphConv
from tqdm import tqdm
from torch.nn import Parameter

def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))
class DMHA(nn.Module):
    def __init__(self, embedding_dimension, number_of_heads, layer_position):
        super().__init__()
        assert embedding_dimension % number_of_heads == 0
        self.head_count = number_of_heads
        self.dimension_per_head = embedding_dimension // number_of_heads
        self.initial_scaling_factor = lambda_init(layer_position)
        self.query_projection_first = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.query_projection_second = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.key_projection_first = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.key_projection_second = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.value_projection = nn.Linear(embedding_dimension, 2 * embedding_dimension, bias=False)
        self.final_combination_projection = nn.Linear(2 * embedding_dimension, embedding_dimension, bias=False)
        self.attention_dropout_layer = nn.Dropout(0.2)
        self.residual_dropout_layer = nn.Dropout(0.2)
        self.sublayer_normalization = nn.LayerNorm(2 * self.dimension_per_head, elementwise_affine=False)
        self.query_lambda_param_first = Parameter(torch.randn(number_of_heads, self.dimension_per_head) * 0.1)
        self.key_lambda_param_first = Parameter(torch.randn(number_of_heads, self.dimension_per_head) * 0.1)
        self.query_lambda_param_second = Parameter(torch.randn(number_of_heads, self.dimension_per_head) * 0.1)
        self.key_lambda_param_second = Parameter(torch.randn(number_of_heads, self.dimension_per_head) * 0.1)

    def forward(self, input_tensor):
        batch_size, sequence_length, embedding_size = input_tensor.shape
        query_first = self.query_projection_first(input_tensor).view(batch_size, sequence_length, self.head_count, self.dimension_per_head).transpose(1, 2)
        query_second = self.query_projection_second(input_tensor).view(batch_size, sequence_length, self.head_count, self.dimension_per_head).transpose(1, 2)
        key_first = self.key_projection_first(input_tensor).view(batch_size, sequence_length, self.head_count, self.dimension_per_head).transpose(1, 2)
        key_second = self.key_projection_second(input_tensor).view(batch_size, sequence_length, self.head_count, self.dimension_per_head).transpose(1, 2)
        value_tensor = self.value_projection(input_tensor).view(batch_size, sequence_length, self.head_count, 2 * self.dimension_per_head).transpose(1, 2)
        scaling_factor = 1.0 / math.sqrt(self.dimension_per_head)
        attention_matrix_first = torch.bmm(query_first.view(batch_size * self.head_count, sequence_length, self.dimension_per_head),
                                         key_first.transpose(-2, -1).view(batch_size * self.head_count, self.dimension_per_head, sequence_length)) * scaling_factor
        attention_matrix_second = torch.bmm(query_second.view(batch_size * self.head_count, sequence_length, self.dimension_per_head),
                                          key_second.transpose(-2, -1).view(batch_size * self.head_count, self.dimension_per_head, sequence_length)) * scaling_factor
        attention_mask = torch.tril(torch.ones(sequence_length, sequence_length, device=input_tensor.device)).unsqueeze(0).unsqueeze(0)
        attention_matrix_first = attention_matrix_first.masked_fill(attention_mask == 0, float('-inf'))
        attention_matrix_second = attention_matrix_second.masked_fill(attention_mask == 0, float('-inf'))
        attention_matrix_first = F.softmax(attention_matrix_first, dim=-1)
        attention_matrix_second = F.softmax(attention_matrix_second, dim=-1)
        lambda_value_first = torch.exp((self.query_lambda_param_first * self.key_lambda_param_first).sum(dim=-1).unsqueeze(-1).unsqueeze(-1))
        lambda_value_second = torch.exp((self.query_lambda_param_second * self.key_lambda_param_second).sum(dim=-1).unsqueeze(-1).unsqueeze(-1))
        lambda_value_combined = lambda_value_first - lambda_value_second + self.initial_scaling_factor
        combined_attention = attention_matrix_first - lambda_value_combined * attention_matrix_second
        combined_attention = self.attention_dropout_layer(combined_attention)
        output_tensor = torch.bmm(combined_attention, value_tensor.view(batch_size * self.head_count, sequence_length, 2 * self.dimension_per_head))
        normalized_output = self.sublayer_normalization(output_tensor)
        scaled_output = normalized_output * (1 - self.initial_scaling_factor)
        scaled_output = scaled_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, 2 * embedding_size)
        final_output = self.residual_dropout_layer(self.final_combination_projection(scaled_output))
        return final_output
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 2 * n_embd, bias=False)
        self.fc2 = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class Elem(nn.Module):
    def __init__(self, n_embd, n_head, attention_class, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = attention_class(n_embd, n_head, layer_idx)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(input_dim,hidden_dim)
        self.conv2 = GraphConv(hidden_dim,hidden_dim)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs,seqs):
        batch_size = inputs.shape[0]
        edge_index, node_features, edge_attr = prepare_graph_data(inputs,seqs)
        x = self.conv1(node_features, edge_index, edge_attr)
        x = self.prelu1(x)  
        x = F.dropout(x, training=self.training)  
        x = self.conv2(x, edge_index, edge_attr)
        x = self.prelu2(x) 
        x = F.dropout(x, training=self.training)  
        x = self.fc(x) 
        seq_len = x.size()[0] // batch_size
        x = x.view(batch_size, seq_len, -1) 
        return x

class CWA(nn.Module):
    def __init__(self, input_dimension, hidden_layer_size, output_task_count):
        super(CWA, self).__init__()
        
        self.value_linear = nn.Linear(input_dimension, hidden_layer_size)
        self.query_linear = nn.Linear(input_dimension, hidden_layer_size)
        self.score_linear = nn.Linear(hidden_layer_size, output_task_count)

    def forward(self, query_input, sequence_data):
        expanded_query = torch.unsqueeze(query_input, dim=1)
        combined_features = self.value_linear(sequence_data) + self.query_linear(expanded_query)
        activated_features = F.tanh(combined_features)
        attention_scores = self.score_linear(activated_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        transposed_sequence = torch.transpose(sequence_data, 1, 2)
        context_result = torch.matmul(transposed_sequence, attention_weights)
        context_result = torch.transpose(context_result, 1, 2)
        return torch.mean(context_result, 1)
	
class CONV_LSTM(nn.Module):
        def __init__(self, feature_size):
            super(CONV_LSTM, self).__init__()
            self.conv_block = nn.Sequential(
                nn.Conv1d(feature_size, feature_size, kernel_size=10),
                nn.PReLU(),
                nn.BatchNorm1d(feature_size),
                nn.Dropout()
            )
            self.lstm_block = nn.LSTM(input_size=feature_size,hidden_size=feature_size // 2,num_layers=1,batch_first=True,bidirectional=True)

        def forward(self, input_tensor):
            conv_input = input_tensor.transpose(1, 2)
            conv_output = self.conv_block(conv_input)
            lstm_input = conv_output.transpose(1, 2)
            hidden_states, (last_representation, _) = self.lstm_block(lstm_input)
            return hidden_states,last_representation
    
class position_emb(nn.Module):
	def __init__(self,d_model,max_len):
		super(position_emb, self).__init__()
		self.pos_embed = nn.Embedding(max_len, d_model)
	def forward(self, x):
		seq_len = x.size(1) 
		pos = torch.arange(seq_len, dtype=torch.long).cuda()
		pos = pos.unsqueeze(0).expand_as(x)
		embedding = self.pos_embed(pos)
		return embedding

print("<------------------ prepare cor file ------------------->")
seg2cor_file = "segfile_dir"                       
seg2cor = {}
with open(seg2cor_file,"r") as f:
    for line in f.readlines():
        coordinates = []
        corlist = line.strip().split('\t')
        for cor in corlist[1:]:
            coords = cor.strip("()").split(", ")
            coords = tuple(float(coord.strip("'")) for coord in coords) 
            coordinates.append(coords)
        seg2cor[corlist[0]] = coordinates
def prepare_graph_data(input_tensor,seqs):
    batch_size, seq_len, embedding_dim = input_tensor.shape
    node_features = input_tensor.view(-1, embedding_dim)
    edges = []
    distances = []
    for i in range(batch_size):
        seq = seqs[i]
        if seq in seg2cor:
            corlist = seg2cor[seq]
            for j in range(1,seq_len - 1):
                if seq[j-1] == 'X':
                    continue
                for k in range(j + 1, seq_len - 1): 
                    if seq[k-1] == 'X':
                        continue
                    corj = corlist[j-1]
                    cork = corlist[k-1]
                    corj = np.array(corj, dtype=np.float32)
                    cork = np.array(cork, dtype=np.float32)
                    distance = np.linalg.norm(np.array(corj) - np.array(cork))
                    if distance <= 10:
                        edges.append((i * seq_len + j, i * seq_len + k))
                        distances.append(distance)
                    else:
                        break
        else:
            for j in range(seq_len - 1):
                for k in range(j + 1, min(j + 9, seq_len)): 
                    edges.append((i * seq_len + j, i * seq_len + k))
                    distance = abs(k - j)
                    distances.append(distance)
                    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    distances = torch.tensor(distances, dtype=torch.float32)
    edge_attr = rbf(distances)
    edge_index = edge_index.cuda()
    node_features = node_features.cuda()
    edge_attr = edge_attr.cuda()

    return edge_index, node_features, edge_attr
def rbf(distance, gamma=10):
    return torch.exp(-gamma * (distance ** 2))

class GTKla(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = position_emb(d_model,max_len+2)
        n_head = 4
        n_layer = 6
        self.dmha = nn.ModuleList([Elem(d_model, n_head, DMHA, layer_idx=i + 1) for i in range(n_layer)])
        self.gnn = GNN(d_model,d_model//2,d_model)
        self.CONV_LSTM = CONV_LSTM(d_model)
        self.CWA = CWA(in_features=d_model,hidden_units=10,num_task=1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(32, 2)
        ) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seq_index,seqs):
        seqs = [seqs[i.item()] for i in seq_index]
        emb = self.emb(x)
        pos = self.pos_emb(x) 
        x = self.norm(emb + pos)
        gnn = self.gnn(x,seqs) 
        CONV_LSTM = self.CONV_LSTM(gnn) 
        hidden_states,last_representation = CONV_LSTM.unsqueeze(1) 
        cwa = self.CWA(hidden_states,last_representation)
        encoded = cwa
        for Elem in self.dmha:
            encoded = Elem(encoded)
        encoded = encoded.squeeze()
        sne = encoded
        logits = self.fc(encoded)
        logits,_ = torch.max(logits,dim=1)
        return logits,sne
    

    
    
