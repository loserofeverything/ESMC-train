import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig

# ==================
# 1. 数据集生成（模拟TCR集合数据）
# ==================
class TCRGenerator:
    """生成模拟TCR序列及标签，阳性样本包含特征motif"""
    
    def __init__(self, num_samples=300, set_size=100, classes=2):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.set_size = set_size
        self.positive_motif = 'CASS'  # 阳性特征片段
        self.num_samples = num_samples
        self.class_ratio = [0.7, 0.3] if classes==2 else [1/classes]*classes
        
    def generate_sequence(self, label):
        """生成单条TCR序列"""
        length = random.randint(12, 18)
        seq = ''.join(random.choices(self.amino_acids, k=length))
        
        # 阳性样本插入特征位点
        if label == 1 and random.random() < 0.3:
            insert_pos = random.randint(0, len(seq)-4)
            seq = seq[:insert_pos] + self.positive_motif + seq[insert_pos+4:]
        return seq

    def create_dataset(self):
        labels = np.random.choice([0,1], size=self.num_samples, p=self.class_ratio)
        data = [
            [self.generate_sequence(l) for _ in range(self.set_size)] 
            for l in labels
        ]
        return data, labels

# 生成300样本（210负/90正），每个样本包含100个TCR
tcr_data, labels = TCRGenerator(num_samples=300, set_size=100).create_dataset()

# ==================
# 2. 数据加载与预处理
# ==================
class TCRDataset(Dataset):
    def __init__(self, data, labels, subsample_size=32, model_name = "esmc_300m"):
        self.data = data
        self.labels = labels
        # self.esm_model = ESMC.from_pretrained(model_name)
        self.client = ESMC.from_pretrained(model_name).to("cuda")
        
    
    def __len__(self):
        return len(self.labels)


    
        
    def __getitem__(self, idx):
        # 随机子采样
        tcr_set = self.data[idx]
        embeddings = []
        for seq in tcr_set:
            protein = ESMProtein(sequence=seq)
            tensor = self.client.encode(protein)
            output = self.client.logits(
                tensor, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            # 使用平均池化获得定长表示
            pooled = torch.mean(output.embeddings, dim=0)
            embeddings.append(pooled.detach().cpu().numpy())         

        return np.array(embeddings), self.labels[idx]

# 数据集划分
train_data = TCRDataset(tcr_data[:200], labels[:200])
val_data = TCRDataset(tcr_data[200:], labels[200:])


class TCRClassifier(torch.nn.Module):
    def __init__(self, input_dim=960):  # ESMC-300M嵌入维度为1280
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),  # 二分类输出
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

model = TCRClassifier()

# ==================
# 4. 训练配置
# ==================
# 加权损失函数（解决类别失衡）
pos_weight = torch.tensor([len(labels)/sum(labels)-1]) 
# criterion = torch.nn.CrossEntropyLoss(weight=pos_weight)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# ==================
# 5. 训练循环
# ==================
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for embeddings, labels in loader:
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ==================
# 6. 验证评估
# ==================
def evaluate(model, loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for emb, labels in loader:
            outputs = model(emb)
            preds.extend(torch.argmax(outputs, dim=1).tolist())
            true_labels.extend(labels.tolist())
    return f1_score(true_labels, preds, average='weighted')

# 执行训练
for epoch in range(10):
    train_loss = train_epoch(model, DataLoader(train_data, batch_size=8, shuffle=True))
    val_f1 = evaluate(model, DataLoader(val_data, batch_size=16))
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val F1={val_f1:.4f}")
