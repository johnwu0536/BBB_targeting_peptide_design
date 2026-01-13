#!/usr/bin/env python3
"""
多模态分类器，整合序列、受体和结构特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)


class SeqEncoder(nn.Module):
    """序列编码器模块。"""
    
    def __init__(self, seq_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MultiModalClassifier(nn.Module):
    """多模态分类器，整合序列、受体和结构特征。"""
    
    def __init__(self, seq_dim: int, receptor_dim: int, struct_dim: int = 0, 
                 energy_dim: int = 0, hidden_dim: int = 256, num_classes: int = 2):
        """
        初始化多模态分类器。
        
        Args:
            seq_dim: 序列特征维度
            receptor_dim: 受体特征维度
            struct_dim: 结构特征维度 (0表示不使用)
            energy_dim: 能量特征维度 (0表示不使用)
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
        """
        super().__init__()
        
        self.seq_dim = seq_dim
        self.receptor_dim = receptor_dim
        self.struct_dim = struct_dim
        self.energy_dim = energy_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 序列编码器
        self.seq_encoder = SeqEncoder(seq_dim, hidden_dim)
        
        # 受体特征编码器
        self.receptor_encoder = nn.Sequential(
            nn.Linear(receptor_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 结构特征编码器 (可选)
        self.struct_encoder = None
        if struct_dim > 0:
            self.struct_encoder = nn.Sequential(
                nn.Linear(struct_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        
        # 能量特征编码器 (可选)
        self.energy_encoder = None
        if energy_dim > 0:
            self.energy_encoder = nn.Sequential(
                nn.Linear(energy_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        
        # 计算融合维度
        fusion_dim = hidden_dim * 2  # 序列 + 受体
        if struct_dim > 0:
            fusion_dim += hidden_dim
        if energy_dim > 0:
            fusion_dim += hidden_dim
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 注意力机制 (可选)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, seq_feats: torch.Tensor, receptor_feats: torch.Tensor,
                struct_feats: Optional[torch.Tensor] = None,
                energy_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            seq_feats: 序列特征 (batch_size, seq_dim)
            receptor_feats: 受体特征 (batch_size, receptor_dim)
            struct_feats: 结构特征 (batch_size, struct_dim) 或 None
            energy_feats: 能量特征 (batch_size, energy_dim) 或 None
            
        Returns:
            分类logits (batch_size, num_classes)
        """
        # 编码序列特征
        h_seq = self.seq_encoder(seq_feats)  # (batch_size, hidden_dim)
        
        # 编码受体特征
        h_rec = self.receptor_encoder(receptor_feats)  # (batch_size, hidden_dim)
        
        # 收集所有特征
        features = [h_seq, h_rec]
        
        # 编码结构特征 (如果可用)
        if self.struct_encoder is not None and struct_feats is not None:
            h_struct = self.struct_encoder(struct_feats)
            features.append(h_struct)
        
        # 编码能量特征 (如果可用)
        if self.energy_encoder is not None and energy_feats is not None:
            h_energy = self.energy_encoder(energy_feats)
            features.append(h_energy)
        
        # 特征融合
        if len(features) > 1:
            # 使用注意力机制融合特征
            features_tensor = torch.stack(features, dim=1)  # (batch_size, n_features, hidden_dim)
            
            # 自注意力
            attended_features, _ = self.attention(
                features_tensor, features_tensor, features_tensor
            )
            
            # 池化
            h_fusion = torch.mean(attended_features, dim=1)  # (batch_size, hidden_dim)
        else:
            h_fusion = features[0]
        
        # 分类
        logits = self.classifier(h_fusion)
        
        return logits
    
    def predict_proba(self, seq_feats: torch.Tensor, receptor_feats: torch.Tensor,
                     struct_feats: Optional[torch.Tensor] = None,
                     energy_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测类别概率。
        
        Args:
            seq_feats: 序列特征
            receptor_feats: 受体特征
            struct_feats: 结构特征
            energy_feats: 能量特征
            
        Returns:
            类别概率 (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(seq_feats, receptor_feats, struct_feats, energy_feats)
            return F.softmax(logits, dim=1)
    
    def save(self, path: str):
        """保存模型权重。"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'seq_dim': self.seq_dim,
                'receptor_dim': self.receptor_dim,
                'struct_dim': self.struct_dim,
                'energy_dim': self.energy_dim,
                'hidden_dim': self.hidden_dim,
                'num_classes': self.num_classes
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """
        从检查点加载模型。
        
        Args:
            path: 检查点路径
            device: 设备
            
        Returns:
            加载的模型
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        if 'config' not in checkpoint:
            raise ValueError("No config found in checkpoint")
        
        config = checkpoint['config']
        model = cls(
            seq_dim=config['seq_dim'],
            receptor_dim=config['receptor_dim'],
            struct_dim=config['struct_dim'],
            energy_dim=config['energy_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model


def create_multimodal_classifier(config: Dict, device: str = 'cpu') -> MultiModalClassifier:
    """
    创建多模态分类器。
    
    Args:
        config: 配置字典
        device: 设备
        
    Returns:
        多模态分类器
    """
    model_cfg = config["classifier"]["model"]
    
    model = MultiModalClassifier(
        seq_dim=model_cfg.get("seq_dim", 128),
        receptor_dim=model_cfg.get("receptor_dim", 10),
        struct_dim=model_cfg.get("struct_dim", 0),
        energy_dim=model_cfg.get("energy_dim", 0),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_classes=model_cfg.get("num_classes", 2)
    )
    
    model.to(device)
    logger.info(f"Created multimodal classifier with {count_parameters(model):,} parameters")
    return model


def count_parameters(model: nn.Module) -> int:
    """计算可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_receptor_features_for_classifier(receptor_name: str) -> torch.Tensor:
    """
    为分类器提取受体特征。
    
    Args:
        receptor_name: 受体名称
        
    Returns:
        受体特征张量
    """
    from src.receptors.features import get_receptor_features
    
    features = get_receptor_features(receptor_name)
    if features is None:
        # 返回默认特征
        return torch.tensor([0.0] * 10, dtype=torch.float32)
    
    # 提取关键特征
    receptor_feats = [
        features.hydropathy,
        features.charge / 100.0,  # 归一化
        features.complexity,
        features.binding_site_charge,
        features.binding_site_hydropathy,
        features.polarity_ratio,
        features.shannon_entropy / 5.0,  # 归一化
        features.length / 5000.0,  # 归一化
        float(features.charge > 0),  # 正电荷指示器
        float(features.hydropathy > 0)  # 疏水指示器
    ]
    
    return torch.tensor(receptor_feats, dtype=torch.float32)


if __name__ == "__main__":
    # 测试多模态分类器
    model = MultiModalClassifier(
        seq_dim=128,
        receptor_dim=10,
        struct_dim=5,
        energy_dim=3,
        hidden_dim=256,
        num_classes=2
    )
    
    batch_size = 4
    seq_feats = torch.randn(batch_size, 128)
    receptor_feats = torch.randn(batch_size, 10)
    struct_feats = torch.randn(batch_size, 5)
    energy_feats = torch.randn(batch_size, 3)
    
    logits = model(seq_feats, receptor_feats, struct_feats, energy_feats)
    print(f"Model output shape: {logits.shape}")
    print(f"Number of parameters: {count_parameters(model):,}")
