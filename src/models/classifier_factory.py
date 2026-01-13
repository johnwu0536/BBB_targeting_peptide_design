#!/usr/bin/env python3
"""
分类器工厂模块，提供统一的分类器创建接口。
"""

import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .classifier import PeptideClassifier
from .enhanced_classifier import EnhancedPeptideClassifier
from .multimodal_classifier import MultiModalClassifier, create_multimodal_classifier, count_parameters

# Set up logging
logger = logging.getLogger(__name__)


def build_classifier(cfg: Dict[str, Any], device: Optional[str] = None) -> nn.Module:
    """
    根据配置构建分类器实例。
    
    Args:
        cfg: 配置字典
        device: 计算设备
        
    Returns:
        分类器模型实例
    """
    classifier_cfg = cfg.get("classifier", {})
    model_type = classifier_cfg.get("type", "basic")
    
    # 提取模型配置参数
    model_params = classifier_cfg.get("model", {})
    
    # 设置默认参数
    default_params = {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "vocab_size": 21,
        "dropout": 0.3,
        "num_classes": 2
    }
    
    # 合并配置参数
    final_params = {**default_params, **model_params}
    
    # 根据类型选择分类器
    if model_type == "enhanced":
        logger.info(f"Building enhanced classifier with params: {final_params}")
        classifier = EnhancedPeptideClassifier(cfg, device)
    elif model_type == "multimodal":
        logger.info("Building multimodal classifier")
        classifier = create_multimodal_classifier(cfg, device)
    else:
        logger.info(f"Building basic classifier with params: {final_params}")
        classifier = PeptideClassifier(**final_params)
    
    # 确保模型移动到正确的设备
    if device:
        classifier = classifier.to(device)
        logger.info(f"Model moved to device: {device}")
    
    return classifier


def log_model_config(cfg: Dict[str, Any]) -> None:
    """
    记录模型配置信息。
    
    Args:
        cfg: 配置字典
    """
    classifier_cfg = cfg.get("classifier", {})
    model_cfg = classifier_cfg.get("model", {})
    
    print("[classifier_factory] Model config:")
    for key, value in model_cfg.items():
        print(f"  - {key}: {value}")
    
    # 记录分类器类型
    classifier_type = classifier_cfg.get("type", "basic")
    print(f"  - type: {classifier_type}")


def load_classifier_from_checkpoint(cfg: Dict[str, Any], checkpoint_path: str, 
                                  device: Optional[str] = None) -> nn.Module:
    """
    从检查点加载分类器。
    
    Args:
        cfg: 配置字典
        checkpoint_path: 检查点路径
        device: 计算设备
        
    Returns:
        加载的分类器模型
    """
    import torch
    
    # 构建分类器
    classifier = build_classifier(cfg, device)
    
    # 加载权重
    if device:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # 处理不同的检查点格式
    if "model_state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["state_dict"])
    else:
        classifier.load_state_dict(checkpoint)
    
    logger.info(f"Loaded classifier from {checkpoint_path}")
    return classifier
