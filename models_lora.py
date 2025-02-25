import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from typing import Tuple, Dict, List
from Bio.Align import substitution_matrices
from torch.cuda.amp import autocast
from esm.models.esmc import ESMC
from peft import get_peft_model, LoraConfig, TaskType



class ESMCClassifier(nn.Module):
    def __init__(self, esmc_model: ESMC, num_classes: int, freeze_backbone=False):
        super().__init__()
        self.esmc = esmc_model
        
        # 冻结预训练参数
        if freeze_backbone:
            for param in self.esmc.parameters():
                param.requires_grad_(False)
        
        # 多序列特征聚合模块
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=960, num_heads=8, batch_first=True
        )
        self.context_encoder = nn.TransformerEncoderLayer(
            d_model=960, nhead=8, dim_feedforward=2048, 
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, sequence_batch: List[List[str]]):
        """
        输入：一批样本（batch_size个样本，每个样本含N条氨基酸序列）
        输出：分类logits (batch_size, num_classes)
        """
        batch_size = len(sequence_batch)
        sample_features = []
        
        # 处理每个样本
        for sequences in sequence_batch: 
            # 对单样本的多序列并行处理
            tokens = self.esmc._tokenize(sequences)
            output = self.esmc(tokens)
            seq_embeddings = output.embeddings  # (N, L, 960)
            
            # 氨基酸级别特征聚合
            seq_embeddings = seq_embeddings.mean(dim=1)  # (N, 960)
            
            # 序列间注意力聚合
            attn_output, _ = self.sequence_attention(
                seq_embeddings, seq_embeddings, seq_embeddings
            )  # (N, 960)
            
            # 上下文编码
            context_rep = self.context_encoder(attn_output)  # (N, 960)
            sample_features.append(context_rep.mean(dim=0))  # (960,)
        
        # 分类预测
        features = torch.stack(sample_features)  # (batch_size, 960)
        return self.classifier(features)
    
    
    
class oldESMCProClassifier(nn.Module):
    def __init__(self, num_classes=4, total_layers=30):
        super().__init__()
        self.esmc = ESMC.from_pretrained("esmc_300m")
        self.total_layers = total_layers
        # 冻结机制修正（根据TransformerStack结构）
        for p in self.esmc.embed.parameters():
            p.requires_grad = False
        self._freeze_transformer(0)
        
        # 特征处理层
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(self.esmc.embed.embedding_dim),
            nn.Linear(self.esmc.embed.embedding_dim, 512),
            nn.GELU()
        )
        
        # 基于transformer的多序列聚合
        # self.aggregator = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=512, nhead=2, batch_first=True
        #     ),
        #     num_layers=1
        # )
        self.class_head = nn.Linear(512, num_classes)

    def _freeze_transformer(self, freeze_layers):
        """根据TransformerStack结构冻结层"""
        # 冻结embedding层
        for block in self.esmc.transformer.blocks[:self.total_layers]:
            for p in block.parameters():
                p.requires_grad = True
        
        # 冻结指定数量的TransformerBlock
        for block in self.esmc.transformer.blocks[:freeze_layers]:
            for p in block.parameters():
                p.requires_grad = False

    def get_sequence_features(self, batch_sequences):
        """批量处理5000条序列的特征抽取"""
        batch_feats = []
        for sample_seq in batch_sequences:  # (5000,)字符串列表
            # 批量编码
            # tokens = self.esmc._tokenize(sample_seq)  # (5000, seq_len)
            # output = self.esmc(tokens)
            
            # _, _, hiddens = (
            #     output.sequence_logits,
            #     output.embeddings,
            #     output.hidden_states,
            # )
            
            # # 使用最终隐藏层
            # seq_feats = torch.mean(hiddens[-1], dim=1)  # (5000, d_model)
            
            
            # 批量编码
            tokens = self.esmc._tokenize(sample_seq)  # (5000, seq_len)
            mask = tokens != self.esmc.tokenizer.pad_token_id
            
            # TransformerStack处理
            x = self.esmc.embed(tokens)
            _, pre_norm, hiddens = self.esmc.transformer(
                x, 
                sequence_id=mask,
                affine=None,
                affine_mask=None,
                chain_id=torch.ones_like(mask)
            )
            
            # 使用最终隐藏层
            seq_feats = torch.mean(hiddens[-1], dim=1)  # (5000, d_model)
            
            
            
            batch_feats.append(self.feat_proj(seq_feats))
            
        return torch.stack(batch_feats)  # (B, 5000, 512)

    def forward(self, batch_sequences):
        features = self.get_sequence_features(batch_sequences)
        # aggregated = self.aggregator(features).mean(dim=1)
        aggregated = torch.mean(features, dim=1)
        return self.class_head(aggregated)
    
    
class ESMCProClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name="esmc_300m"):
        super().__init__()
        self.esmc = ESMC.from_pretrained(model_name)
                
        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            # 根据ESMC模型中MultiHeadAttention的具体实现修改target_modules
            target_modules=[
                "layernorm_qkv.1",  # MultiHeadAttention中的线性层
                "out_proj"          # MultiHeadAttention中的输出投影层
            ],
            inference_mode=False,
            bias="none",
        )
        
        # 将原始模型转换为LoRA模型
        self.esmc = get_peft_model(self.esmc, peft_config)
        # 冻结除了LoRA参数之外的其他参数
        for param in self.esmc.parameters():
            if not getattr(param, "is_lora", False):
                param.requires_grad = False
        
        # 特征处理层
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(self.esmc.embed.embedding_dim),
            nn.Linear(self.esmc.embed.embedding_dim, 512),
            nn.GELU()
        )

        self.class_head = nn.Linear(512, num_classes)


    def get_sequence_features(self, batch_sequences):
        """批量处理5000条序列的特征抽取"""
        batch_feats = []
        for sample_seq in batch_sequences:  # (5000,)字符串列表
            # 批量编码
            tokens = self.esmc._tokenize(sample_seq)  # (5000, seq_len)
            sequence_id = tokens != self.esmc.tokenizer.pad_token_id
            x = self.esmc.embed(tokens)
            x, _, hiddens = self.esmc.transformer(x, sequence_id=sequence_id)
            hiddens = torch.stack(hiddens, dim=0)
            
            # output = self.esmc(tokens)
            # _, _, hiddens = (
            #     output.sequence_logits,
            #     output.embeddings,
            #     output.hidden_states,
            # )
            
            
            # 使用最终隐藏层
            seq_feats = torch.mean(hiddens[-1], dim=1)  # (5000, d_model)
            batch_feats.append(self.feat_proj(seq_feats))
            
        return torch.stack(batch_feats)  # (B, 5000, 512)

    def forward(self, batch_sequences):
        features = self.get_sequence_features(batch_sequences)
        aggregated = torch.mean(features, dim=1)
        return self.class_head(aggregated)