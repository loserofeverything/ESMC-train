import os
import time
from dataset import *
from models import *
from plot import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
# from peft import LoraConfig, get_peft_model
from esm.tokenization import EsmSequenceTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_mapping': {'labels': [0,1,2,3]}  # 保存类别映射信息
    }, path)

# 加载模型（推理模式）
def load_for_inference(path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    # 初始化新模型实例
    model = ESMCProClassifier(num_classes=len(checkpoint['class_mapping']['labels'])) 
    model.load_state_dict(checkpoint['model_state'])
    # 加载ESMC令牌化器
    model.esmc.tokenizer = EsmSequenceTokenizer()  # 根据实际初始化方式调整
    model.eval()
    return model.to(device)


class LabelEncoder:
    def __init__(self):
        self.class_to_index = {}
        self.index_to_class = {}

    def fit(self, label_list):

        self.class_to_index = {label: index for index, label in enumerate(label_list)}
        self.index_to_class = {index: label for label, index in self.class_to_index.items()}

    def transform(self, label_list):
        return [self.class_to_index[label] for label in label_list]


def shuffle_samples(tensor):
    """
    随机打乱每个样本中数据点的排列顺序
    Args:
        tensor: 输入张量，形状为(B, 5000, 960)
    Returns:
        打乱后的张量，形状保持不变
    """
    B, N, D = tensor.shape  # B:批次大小, N:5000, D:960
    
    # 为每个样本生成随机排列索引
    # 使用同样的种子以保证结果可复现
    torch.manual_seed(42)
    indices = torch.stack([torch.randperm(N) for _ in range(B)])  # shape: (B, 5000)
    
    # 创建索引张量用于收集操作
    batch_indices = torch.arange(B).unsqueeze(1).expand(-1, N)  # shape: (B, 5000)
    
    # 使用高级索引进行打乱
    shuffled_tensor = tensor[batch_indices, indices]
    
    return shuffled_tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha)  # 注册到buffer实现设备同步
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        FL_loss = (1-pt)**self.gamma * BCE_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)  # 设备同步保证
            FL_loss = alpha[targets.long()] * FL_loss
        
        return torch.mean(FL_loss) if self.reduction=='mean' else FL_loss



def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def set_seed(seed):
    random.seed(seed)                        # 设置 Python 随机数种子
    np.random.seed(seed)                     # 设置 NumPy 随机数种子
    torch.manual_seed(seed)                  # 设置 PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)             # 设置 PyTorch GPU 随机数种子
    torch.cuda.manual_seed_all(seed)         # 如果使用多个GPU

    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法相同
    torch.backends.cudnn.benchmark = True     # 禁用CuDNN的自动优化，确保可重复性


def fill_sublists(A):
    # 找到最长子列表的长度
    max_length = max(len(sublist) for sublist in A)

    # 填充每个子列表
    for sublist in A:
        sublist.extend(['X'] * (max_length - len(sublist)))
    return A


def collate_fn(batch):
    sids, fids, cdr3s, labels, seq_lens, cloneFractions = zip(*batch)
    unique_labels = torch.tensor([tensor[0] for tensor in labels])
    sids = torch.tensor(sids)
    fids = torch.tensor(fids)

    return sids, fids, unique_labels, cdr3s

def collate_fnIS(batch):
    sids, fids, cdr3s, labels, encoded, seq_lens, cloneFractions = zip(*batch)
    # 提取每个张量的唯一值（实际上就是每个张量的第一个元素）
    encoded = torch.stack(encoded, dim=0)
    sids = torch.tensor(sids)
    labels = torch.tensor(labels)
    fids = torch.tensor(fids)
    return sids, fids, labels, encoded


def collate_fnISMeta(batch):
    sids, fids, cdr3s, labels, encoded, seq_lens = zip(*batch)
    # 提取每个张量的唯一值（实际上就是每个张量的第一个元素）
    encoded = torch.stack(encoded, dim=0)
    sids = torch.tensor(sids)
    labels = torch.tensor(labels)
    fids = torch.tensor(fids)
    return sids, fids, labels, encoded


# 示例使用
if __name__ == "__main__":

    
    label_names_sub = ['Breast','Lung','Colorectal', 'Healthy']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder()
    label_encoder.fit(label_names_sub)
    print(label_encoder.transform(label_names_sub))


    split = 5
    is_load = True
    topk = 5000
    is_start_training = True
    start_time = time.time()

    train_dataset = dataset(None, label_encoder=label_encoder, mode = 'train', \
        topk = topk, is_load=is_load, split=split, h5_root_dir="/root/autodl-tmp/12_25_balanced_long", classname='lab', \
            is_load_encoded=False, exclude_k=4)
    end_time = time.time()
    print("Time taken to load train dataset: ", end_time - start_time)
    
    start_time = time.time()
    test_dataset = dataset(None, label_encoder=label_encoder, mode = 'test', \
        topk = topk, is_load=is_load, split=split,h5_root_dir="/root/autodl-tmp/12_25_balanced_long", classname='lab', \
            is_load_encoded=False, exclude_k=4)
    end_time = time.time()
    print("Time taken to load Test dataset: ", end_time - start_time)
    
    
    
    if is_start_training:
        
        split = 4
        save_dir = "/root/autodl-tmp/ESM/exp"
        root_dir = "split{}_ESMCfinetune_ON_12_25LongNoBenignMeanPooledlab".format(split) #实验名
        save_dir = os.path.join(save_dir, root_dir)
        model_name = "{}__10_24VecAggregator_1".format(topk)
        fig_save_dir = os.path.join(save_dir, "figs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        exp_path = "modelname{}".format(model_name)
        model_save_path = os.path.join(save_dir, exp_path)
        fig_save_path = os.path.join(fig_save_dir, exp_path)
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
  
        model = ESMCProClassifier(num_classes=split)
        model = model.to(torch.bfloat16)
        model = model.to(device)
        freqs = torch.tensor([672, 898, 452, 463], dtype=torch.bfloat16)
        class_weights = 1.0 / (torch.log(1.02 + freqs))  # 示例：可根据需要调整
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW([
        {'params': model.feat_proj.parameters()},
        {'params': model.aggregator.parameters()},
        {'params': model.class_head.parameters(), 'lr': 1e-3}
        ], lr=1e-4)
        
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        

        
        train_dataset.set_seqOrNot(False)
        test_dataset.set_seqOrNot(False)
        
        train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, pin_memory=True,collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, pin_memory=True, collate_fn=collate_fn, num_workers=4)

        
        accum_steps = 8
        train_losses = []
        min_loss = float('inf')
        max_auc = 0
        max_acc = 0
        save_interval = 40
        train_auces = []
        train_acces = []
        val_auces = []
        val_acces = []
        val_losses = []
        total_epochs = 100

        # save_checkpoint(model, optimizer, 0, "/root/autodl-tmp/ESM/testsave.pth")

        for epoch in range(total_epochs):
            model.train()
            all_preds = []
            all_labels = []
            loss_sum = 0.0
            step_count = 0
            total_layers = len(model.esmc.transformer.blocks)
            
            if epoch <= 3:
                freeze_num = total_layers
                model._freeze_transformer(freeze_num)
                optimizer.param_groups[0]['lr'] = 0
                optimizer.param_groups[1]['lr'] = 0
                
            elif epoch <= 10:
                freeze_num = total_layers - 4
                model._freeze_transformer(freeze_num)
                optimizer.add_param_group({
                    'params': [p for p in model.esmc.parameters() if p.requires_grad],
                    'lr': 1e-5
                })
            else:
                freeze_num = 0
                model._freeze_transformer(freeze_num)
                optimizer.param_groups[-1]['lr'] = 1e-6
                
            
            for batch_idx, (sid, fid, label, cdr3) in tqdm.tqdm(enumerate(train_loader), \
            desc="Train epoch {}".format(epoch), total=len(train_loader)):
                
                batch_size = label.shape[0]
                label = label.to(device)
                optimizer.zero_grad()
                outputs = model(cdr3)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                step_count += 1
                        
                logits = F.softmax(outputs.float(), dim=1).cpu().detach().numpy()
                all_preds.extend(logits)
                all_labels.extend(label.cpu().numpy())
                
                
            if split == 2:
                auc_value = calculate_auc(all_preds, all_labels)
            else:
                auc_value = calculate_multiclass_auc(all_preds, all_labels)
            predicted_labels = [np.argmax(pred) for pred in all_preds]
            accuracy = np.sum(np.array(predicted_labels) == np.array(all_labels))
            accuracy = accuracy / len(all_labels)
            loss_sum /= step_count
            train_losses.append(loss_sum)
            train_auces.append(auc_value)
            train_acces.append(accuracy)
            cm = confusion_matrix(all_labels, predicted_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            cm_save_path = os.path.join(fig_save_path, 'cm')
            if not os.path.exists(cm_save_path):
                os.makedirs(cm_save_path)
            plt.savefig(os.path.join(cm_save_path, "epoch_{}_train.png".format(epoch)))
            plt.close()
            plot_roc_pr_curves(all_preds, all_labels, epoch, fig_save_path, mode="train")
            
            print("Train epoch {} Loss: {} AUC: {} Acc: {}".format(epoch, loss_sum, auc_value, accuracy))
            
            model.eval()
            all_preds = []
            loss_sum = 0
            all_labels = []
            
            with torch.no_grad():
                
                step_count = 0
                for sid, fid, label, cdr3 in tqdm.tqdm(test_loader, \
                desc="Test epoch {}".format(epoch), total=len(test_loader)):
                    batch_size = label.shape[0]
                    label = label.to(device)

                    outputs = model(cdr3)
                    loss = criterion(outputs, label)
                    loss_sum += loss.item()
                    step_count += 1
                    logits = F.softmax(outputs.float(), dim=1).cpu().detach().numpy()
                    all_preds.extend(logits)
                    all_labels.extend(label.cpu().numpy())
                    
            
            
            if split == 2:
                auc_value = calculate_auc(all_preds, all_labels)
            else:
                auc_value = calculate_multiclass_auc(all_preds, all_labels)
            
            predicted_labels = [np.argmax(pred) for pred in all_preds]
            accuracy = np.sum(np.array(predicted_labels) == np.array(all_labels))
            accuracy = accuracy / len(all_labels)
            loss_sum /= step_count
            cm = confusion_matrix(all_labels, predicted_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            cm_save_path = os.path.join(fig_save_path, 'cm')
            if not os.path.exists(cm_save_path):
                os.makedirs(cm_save_path)
            plt.savefig(os.path.join(cm_save_path, "epoch_{}_test.png".format(epoch)))
            plt.close()
            plot_roc_pr_curves(all_preds, all_labels, epoch, fig_save_path, mode="test")
            val_losses.append(loss_sum)
            val_auces.append(auc_value)
            val_acces.append(accuracy)
            
            print("val epoch {} Loss: {} AUC: {} Acc: {}".format(epoch, loss_sum, auc_value, accuracy))        
            
    
            
            
            if accuracy > max_acc:
                max_acc = accuracy
                ckp_path = os.path.join(model_save_path, "epoch_{}_acc_{}.pth".format(epoch, accuracy))
                save_checkpoint(model, optimizer, epoch, ckp_path)
                print(f"模型已保存到：{ckp_path}")
                
                
            elif epoch%save_interval==0:
                ckp_path = os.path.join(model_save_path, "epoch_{}_acc_{}.pth".format(epoch, accuracy))
                save_checkpoint(model, optimizer, epoch, ckp_path)
                print(f"模型已保存到：{ckp_path}")
            
            print("训练完成")
            # 绘制损失图
            plt.figure()
            plt.plot(range(len(train_losses)), train_losses, marker='o')
            plt.title('Training Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(fig_save_path,'training_loss.png'))  # 保存图像
            plt.close()
            
            plt.figure()
            plt.plot(range(len(train_auces)), train_auces, marker='o')
            plt.title('Training Auc per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.grid(True)
            plt.savefig(os.path.join(fig_save_path,'training_auc.png'))  # 保存图像
            plt.close()
            
            plt.figure()
            plt.plot(range(len(val_acces)), val_acces, marker='o')
            plt.title('testing Acc per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('ACC')
            plt.grid(True)
            plt.savefig(os.path.join(fig_save_path,'eval_acc.png'))  # 保存图像
            plt.close()
            
            plt.figure()
            plt.plot(range(len(train_acces)), train_acces, marker='o')
            plt.title('training Acc per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('ACC')
            plt.grid(True)
            plt.savefig(os.path.join(fig_save_path,'train_acc.png'))  # 保存图像
            plt.close()
            
            plt.figure()
            plt.plot(range(len(val_auces)), val_auces, marker='o')
            plt.title('testing Auc per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.grid(True)
            plt.savefig(os.path.join(fig_save_path,'eval_auc.png'))  # 保存图像
            plt.close()
            
            plt.figure()
            plt.plot(range(len(val_losses)), val_losses, marker='o')
            plt.title('testing Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(fig_save_path,'eval_loss.png'))  # 保存图像
            plt.close()
        
        
        
    