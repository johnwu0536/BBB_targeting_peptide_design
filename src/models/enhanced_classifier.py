#!/usr/bin/env python3
"""
增强的分类器模型，整合受体蛋白序列特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np

from .classifier import PeptideClassifier
from src.utils.fasta_parser import FastaParser


class EnhancedPeptideClassifier(PeptideClassifier):
    """
    增强的肽段分类器，整合受体序列特征。
    """
    
    def __init__(self, config: Dict, device: torch.device):
        super().__init__(config, device)
        
        # 加载受体特征
        self.receptor_features = self._load_receptor_features(config)
        
        # 增强的特征维度
        self.receptor_feature_dim = 20  # 受体特征维度
        
        # 增强的分类器头
        enhanced_hidden_dim = config["classifier"]["hidden"] + self.receptor_feature_dim
        self.enhanced_classifier = nn.Sequential(
            nn.Linear(enhanced_hidden_dim, enhanced_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config["classifier"]["dropout"]),
            nn.Linear(enhanced_hidden_dim // 2, 2)  # 二分类
        )
        
        # 受体特征编码器
        self.receptor_encoder = nn.Sequential(
            nn.Linear(self._get_receptor_feature_size(), 64),
            nn.ReLU(),
            nn.Linear(64, self.receptor_feature_dim)
        )
    
    def _load_receptor_features(self, config: Dict) -> Dict:
        """加载受体特征信息。"""
        data_dir = config["paths"]["data_dir"]
        parser = FastaParser(data_dir)
        
        receptor_features = {}
        for receptor in ['TFRC', 'INSR', 'LRP1', 'ICAM1', 'NCAM1']:
            receptor_data = parser.get_receptor_sequence(receptor)
            if receptor_data:
                features = parser.extract_sequence_features(receptor_data['sequence'])
                receptor_features[receptor] = {
                    'sequence_info': receptor_data,
                    'features': features
                }
        
        print(f"[enhanced_classifier] Loaded features for {len(receptor_features)} receptors")
        return receptor_features
    
    def _get_receptor_feature_size(self) -> int:
        """计算受体特征向量的维度。"""
        # 基本特征：长度、平均疏水性、净电荷、极性比例、复杂性
        return 5
    
    def _encode_receptor_features(self, receptor_name: str) -> torch.Tensor:
        """编码受体特征为向量。"""
        if receptor_name not in self.receptor_features:
            # 返回默认特征向量
            return torch.zeros(self._get_receptor_feature_size())
        
        features = self.receptor_features[receptor_name]['features']
        physchem = features['physchem_properties']
        complexity = features['sequence_complexity']
        
        # 构建特征向量
        feature_vector = [
            features['length'] / 1000.0,  # 归一化长度
            physchem['avg_hydropathy'],
            physchem['net_charge'] / 10.0,  # 归一化电荷
            physchem['polarity_ratio'],
            complexity['complexity_score']
        ]
        
        return torch.tensor(feature_vector, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor, receptor_name: str = None) -> torch.Tensor:
        """
        前向传播，整合受体特征。
        
        Args:
            x: 输入肽段序列张量
            receptor_name: 受体名称
            
        Returns:
            分类logits
        """
        # 基础肽段特征提取
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # 使用注意力机制聚合序列信息
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 基础特征
        base_features = self.classifier(context_vector)
        
        if receptor_name:
            # 整合受体特征
            receptor_feat = self._encode_receptor_features(receptor_name).to(x.device)
            encoded_receptor = self.receptor_encoder(receptor_feat)
            
            # 融合特征
            combined_features = torch.cat([base_features, encoded_receptor], dim=-1)
            output = self.enhanced_classifier(combined_features)
        else:
            # 回退到基础分类器
            output = base_features
        
        return output
    
    def predict_with_receptor(self, sequences: List[str], receptor_name: str, 
                            device: torch.device) -> torch.Tensor:
        """
        针对特定受体预测肽段结合概率。
        
        Args:
            sequences: 肽段序列列表
            receptor_name: 受体名称
            device: 计算设备
            
        Returns:
            预测概率
        """
        self.eval()
        
        # 编码序列
        encoded_seqs = []
        for seq in sequences:
            encoded = self._encode_sequence(seq)
            encoded_seqs.append(encoded)
        
        if not encoded_seqs:
            return torch.tensor([])
        
        batch_tensor = torch.stack(encoded_seqs).to(device)
        
        with torch.no_grad():
            logits = self.forward(batch_tensor, receptor_name)
            probs = F.softmax(logits, dim=-1)
        
        return probs[:, 1]  # 返回正类概率


def create_enhanced_classifier(config: Dict, device: torch.device) -> EnhancedPeptideClassifier:
    """创建增强的分类器实例。"""
    return EnhancedPeptideClassifier(config, device)


def compare_receptor_specificity(classifier: EnhancedPeptideClassifier, 
                               sequences: List[str], device: torch.device) -> Dict:
    """
    比较肽段对不同受体的特异性。
    
    Args:
        classifier: 增强分类器
        sequences: 肽段序列列表
        device: 计算设备
        
    Returns:
        各受体的预测概率字典
    """
    results = {}
    
    for receptor in ['TFRC', 'INSR', 'LRP1', 'ICAM1', 'NCAM1']:
        probs = classifier.predict_with_receptor(sequences, receptor, device)
        results[receptor] = probs.cpu().numpy()
    
    return results


class MultiReceptorClassifier(nn.Module):
    """
    多受体分类器，同时预测肽段与多个受体的结合。
    """
    
    def __init__(self, config: Dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # 基础肽段编码器
        self.embedding_dim = config["classifier"]["embedding"]
        self.hidden_dim = config["classifier"]["hidden"]
        self.vocab_size = 21  # 20 AA + padding
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(self.hidden_dim * 2, 1)
        
        # 受体特定的分类器头
        self.receptor_heads = nn.ModuleDict({
            'TFRC': self._create_classifier_head(),
            'INSR': self._create_classifier_head(),
            'LRP1': self._create_classifier_head(),
            'ICAM1': self._create_classifier_head(),
            'NCAM1': self._create_classifier_head()
        })
    
    def _create_classifier_head(self) -> nn.Module:
        """创建受体特定的分类器头。"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config["classifier"]["dropout"]),
            nn.Linear(self.hidden_dim, 2)
        )
    
    def forward(self, x: torch.Tensor, receptor: str = None) -> torch.Tensor:
        """前向传播。"""
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        if receptor and receptor in self.receptor_heads:
            # 特定受体预测
            return self.receptor_heads[receptor](context_vector)
        else:
            # 多受体预测
            outputs = {}
            for rec, head in self.receptor_heads.items():
                outputs[rec] = head(context_vector)
            return outputs
    
    def predict_multi_receptor(self, sequences: List[str], device: torch.device) -> Dict:
        """预测肽段与所有受体的结合概率。"""
        self.eval()
        
        # 编码序列
        encoded_seqs = []
        for seq in sequences:
            encoded = self._encode_sequence(seq)
            encoded_seqs.append(encoded)
        
        if not encoded_seqs:
            return {}
        
        batch_tensor = torch.stack(encoded_seqs).to(device)
        
        with torch.no_grad():
            outputs = self.forward(batch_tensor)
        
        results = {}
        for receptor, logits in outputs.items():
            probs = F.softmax(logits, dim=-1)
            results[receptor] = probs[:, 1].cpu().numpy()  # 正类概率
        
        return results
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """编码肽段序列。"""
        encoded = []
        for aa in sequence:
            if aa in "ACDEFGHIKLMNPQRSTVWY":
                encoded.append("ACDEFGHIKLMNPQRSTVWY".index(aa) + 1)  # 1-indexed
            else:
                encoded.append(0)  # Unknown/padding
        
        # 填充到最大长度
        max_length = self.config["task"]["max_len"]
        if len(encoded) < max_length:
            encoded = encoded + [0] * (max_length - len(encoded))
        elif len(encoded) > max_length:
            encoded = encoded[:max_length]
        
        return torch.tensor(encoded, dtype=torch.long)
