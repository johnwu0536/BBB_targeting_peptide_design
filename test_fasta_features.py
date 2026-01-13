#!/usr/bin/env python3
"""
测试FASTA序列特征提取和增强分类器功能。
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.utils.fasta_parser import FastaParser, get_all_receptor_features
from src.models.enhanced_classifier import EnhancedPeptideClassifier, compare_receptor_specificity
from src.envs.enhanced_peptide_env import create_enhanced_environment


def test_fasta_parser():
    """测试FASTA解析器功能。"""
    print("=" * 50)
    print("测试FASTA解析器")
    print("=" * 50)
    
    try:
        parser = FastaParser("data")
        
        # 测试单个受体
        tfr_data = parser.get_receptor_sequence("TFRC")
        if tfr_data:
            print(f"[PASS] TFRC序列加载成功")
            print(f"  长度: {tfr_data['length']}")
            print(f"  UniProt ID: {tfr_data['header_info'].get('uniprot_id', 'N/A')}")
            
            # 提取特征
            features = parser.extract_sequence_features(tfr_data['sequence'])
            print(f"  平均疏水性: {features['physchem_properties']['avg_hydropathy']:.2f}")
            print(f"  净电荷: {features['physchem_properties']['net_charge']:.2f}")
            print(f"  复杂性: {features['sequence_complexity']['complexity_score']:.3f}")
        else:
            print("[FAIL] TFRC序列加载失败")
        
        # 测试所有受体
        all_features = get_all_receptor_features()
        print(f"\n[PASS] 成功加载 {len(all_features)} 个受体的特征")
        
        for receptor, data in all_features.items():
            features = data['features']
            print(f"  {receptor}: 长度={data['sequence_info']['length']}, "
                  f"疏水性={features['physchem_properties']['avg_hydropathy']:.2f}, "
                  f"复杂性={features['sequence_complexity']['complexity_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] FASTA解析器测试失败: {e}")
        return False


def test_enhanced_classifier():
    """测试增强分类器功能。"""
    print("\n" + "=" * 50)
    print("测试增强分类器")
    print("=" * 50)
    
    try:
        # 加载配置
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建增强分类器
        classifier = EnhancedPeptideClassifier(config, device)
        print("[PASS] 增强分类器创建成功")
        
        # 测试序列
        test_sequences = [
            "YPRGGSV",  # 包含基序的序列
            "ACDEFGHI",  # 普通序列
            "KKKKKKK",   # 可能有毒的序列
        ]
        
        # 测试多受体特异性
        print("\n测试多受体特异性:")
        specificity_results = compare_receptor_specificity(classifier, test_sequences, device)
        
        for receptor, probs in specificity_results.items():
            print(f"  {receptor}: {[f'{p:.3f}' for p in probs]}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 增强分类器测试失败: {e}")
        return False


def test_enhanced_environment():
    """测试增强环境功能。"""
    print("\n" + "=" * 50)
    print("测试增强环境")
    print("=" * 50)
    
    try:
        # 加载配置
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建分类器（使用基础分类器作为示例）
        from src.models.classifier import PeptideClassifier
        classifier = PeptideClassifier(config, device)
        
        # 创建增强环境
        env = create_enhanced_environment(config, "TFRC", classifier)
        print("[PASS] 增强环境创建成功")
        
        # 测试环境重置
        state = env.reset()
        print(f"[PASS] 环境重置成功，状态形状: {state.shape}")
        
        # 测试几个步骤
        for step in range(3):
            action = np.random.randint(0, len(env.aa_alphabet) + 1)  # 随机动作
            next_state, reward, done, info = env.step(action)
            print(f"  步骤 {step+1}: 动作={action}, 奖励={reward:.3f}, 完成={done}, "
                  f"序列='{info['sequence']}'")
            
            if done:
                break
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 增强环境测试失败: {e}")
        return False


def analyze_receptor_features():
    """分析受体特征。"""
    print("\n" + "=" * 50)
    print("受体特征分析")
    print("=" * 50)
    
    try:
        all_features = get_all_receptor_features()
        
        print("受体特征统计:")
        for receptor, data in all_features.items():
            features = data['features']
            physchem = features['physchem_properties']
            complexity = features['sequence_complexity']
            
            print(f"\n{receptor}:")
            print(f"  序列长度: {features['length']}")
            print(f"  平均疏水性: {physchem['avg_hydropathy']:.2f}")
            print(f"  净电荷: {physchem['net_charge']:.2f}")
            print(f"  极性比例: {physchem['polarity_ratio']:.3f}")
            print(f"  复杂性分数: {complexity['complexity_score']:.3f}")
            print(f"  香农熵: {complexity['shannon_entropy']:.3f}")
        
        # 比较不同受体的特征
        print("\n受体特征比较:")
        receptors = list(all_features.keys())
        
        # 疏水性比较
        hydropathies = [all_features[r]['features']['physchem_properties']['avg_hydropathy'] for r in receptors]
        print(f"  平均疏水性范围: {min(hydropathies):.2f} - {max(hydropathies):.2f}")
        
        # 复杂性比较
        complexities = [all_features[r]['features']['sequence_complexity']['complexity_score'] for r in receptors]
        print(f"  复杂性范围: {min(complexities):.3f} - {max(complexities):.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 受体特征分析失败: {e}")
        return False


def main():
    """主测试函数。"""
    print("FASTA序列特征利用测试")
    print("=" * 60)
    
    tests = [
        ("FASTA解析器", test_fasta_parser),
        ("受体特征分析", analyze_receptor_features),
        ("增强分类器", test_enhanced_classifier),
        ("增强环境", test_enhanced_environment),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[FAIL] {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "[PASS] 通过" if success else "[FAIL] 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(tests)} 测试通过")
    
    if passed == len(tests):
        print("[SUCCESS] 所有测试通过！FASTA序列特征利用功能正常工作。")
    else:
        print("[WARNING] 部分测试失败，请检查相关功能。")


if __name__ == "__main__":
    main()
