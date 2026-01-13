#!/usr/bin/env python3
"""
测试增强的PPO + 分类器管道。
"""

import os
import sys
import yaml
import torch
import numpy as np

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.classifier_factory import build_classifier, log_model_config
from src.receptors.features import get_receptor_features, compute_receptor_compatibility, print_receptor_summary
from src.utils.encoding import clean_sequence, encode_sequence


def test_classifier_factory():
    """测试分类器工厂。"""
    print("=" * 60)
    print("Testing Classifier Factory")
    print("=" * 60)
    
    # 加载配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 测试不同分类器类型
    classifier_types = ["basic", "enhanced", "multimodal"]
    
    for classifier_type in classifier_types:
        print(f"\nTesting {classifier_type} classifier:")
        
        # 更新配置
        config["classifier"]["type"] = classifier_type
        
        try:
            # 构建分类器
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            classifier = build_classifier(config, device)
            
            # 记录配置
            log_model_config(config)
            
            # 测试前向传播
            if classifier_type == "multimodal":
                # 多模态分类器需要额外的特征
                batch_size = 2
                seq_feats = torch.randn(batch_size, 128).to(device)
                receptor_feats = torch.randn(batch_size, 10).to(device)
                logits = classifier(seq_feats, receptor_feats)
            else:
                # 基本和增强分类器使用序列输入
                batch_size = 2
                seq_len = 15
                sequences = torch.randint(0, 21, (batch_size, seq_len)).to(device)
                logits = classifier(sequences)
            
            print(f"  [OK] {classifier_type} classifier built successfully")
            print(f"  [OK] Forward pass: {logits.shape}")
            
        except Exception as e:
            print(f"  [ERROR] {classifier_type} classifier failed: {e}")


def test_receptor_features():
    """测试受体特征模块。"""
    print("\n" + "=" * 60)
    print("Testing Receptor Features")
    print("=" * 60)
    
    # 打印受体摘要
    print_receptor_summary()
    
    # 测试受体特征获取
    receptors = ["TFRC", "INSR", "LRP1", "ICAM1", "NCAM1"]
    
    for receptor in receptors:
        features = get_receptor_features(receptor)
        if features:
            print(f"[OK] {receptor}: hydropathy={features.hydropathy:.2f}, charge={features.charge:.2f}")
        else:
            print(f"[ERROR] {receptor}: not found")
    
    # 测试兼容性计算
    print("\nTesting receptor compatibility:")
    peptide_props = {
        "charge": 2.0,
        "hydropathy": -0.3,
        "complexity": 0.025
    }
    
    for receptor in receptors:
        features = get_receptor_features(receptor)
        if features:
            compatibility = compute_receptor_compatibility(peptide_props, features)
            print(f"  {receptor}: compatibility = {compatibility:.3f}")


def test_visualization():
    """测试可视化工具。"""
    print("\n" + "=" * 60)
    print("Testing Visualization Tools")
    print("=" * 60)
    
    try:
        from scripts.plot_receptor_radar import plot_receptor_radar, plot_receptor_comparison
        
        # 测试雷达图
        plot_receptor_radar("test_results/receptor_radar.png")
        print("[OK] Radar plot created successfully")
        
        # 测试比较图
        plot_receptor_comparison("test_results/receptor_comparison.png")
        print("[OK] Comparison plot created successfully")
        
    except ImportError as e:
        print(f"[ERROR] Visualization failed (matplotlib not available): {e}")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")


def test_sequence_processing():
    """测试序列处理。"""
    print("\n" + "=" * 60)
    print("Testing Sequence Processing")
    print("=" * 60)
    
    test_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",  # 标准序列
        "ACDEFGHIKLMNPQRSTVWYX",  # 包含非标准氨基酸
        "ACD",  # 太短
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",  # 太长
        "ACDEFGHIKLMNPQRSTVWY-ACD",  # 包含分隔符
    ]
    
    for seq in test_sequences:
        cleaned = clean_sequence(seq)
        encoded = encode_sequence(cleaned) if cleaned else None
        
        print(f"Original: {seq}")
        print(f"Cleaned:  {cleaned}")
        print(f"Encoded:  {encoded}")
        print("-" * 40)


def test_multimodal_features():
    """测试多模态特征提取。"""
    print("\n" + "=" * 60)
    print("Testing Multimodal Feature Extraction")
    print("=" * 60)
    
    try:
        from src.models.multimodal_classifier import extract_receptor_features_for_classifier
        
        receptors = ["TFRC", "INSR", "LRP1"]
        
        for receptor in receptors:
            features = extract_receptor_features_for_classifier(receptor)
            print(f"{receptor}: {features.tolist()}")
            
    except Exception as e:
        print(f"Multimodal feature extraction failed: {e}")


def main():
    """主测试函数。"""
    print("Enhanced PPO + Classifier Pipeline Test Suite")
    print("=" * 60)
    
    # 创建测试结果目录
    os.makedirs("test_results", exist_ok=True)
    
    # 运行测试
    test_classifier_factory()
    test_receptor_features()
    test_visualization()
    test_sequence_processing()
    test_multimodal_features()
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
