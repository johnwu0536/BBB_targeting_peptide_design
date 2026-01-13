#!/usr/bin/env python3
"""
增强的肽段生成环境，整合受体蛋白序列特征。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import logging

from src.utils.encoding import clean_sequence, validate_sequence_length
from src.models.enhanced_classifier import EnhancedPeptideClassifier
from src.utils.fasta_parser import FastaParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='[enhanced_peptide_env] %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPeptideEnvironment:
    """增强的肽段序列生成环境，整合受体特征。"""
    
    def __init__(self, config: Dict, target_receptor: str, classifier_model: nn.Module):
        self.config = config
        self.target_receptor = target_receptor
        self.classifier = classifier_model
        self.classifier.eval()
        
        # 环境参数
        self.min_len = config["task"]["min_len"]
        self.max_len = config["task"]["max_len"]
        self.vocab_size = 21  # 20 AA + padding
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        # 奖励权重
        self.reward_weights = config["reward_weights"]
        self.physchem_constraints = config["physchem_constraints"]
        
        # 加载受体特征
        self.receptor_features = self._load_receptor_features()
        
        # 当前状态
        self.current_sequence = []
        self.done = False
        
    def _load_receptor_features(self) -> Dict:
        """加载受体特征信息。"""
        data_dir = self.config["paths"]["data_dir"]
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
        
        logger.info(f"Loaded features for {len(receptor_features)} receptors")
        return receptor_features
    
    def reset(self) -> torch.Tensor:
        """重置环境到初始状态。"""
        self.current_sequence = []
        self.done = False
        return self._get_state()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """执行动作并返回下一个状态、奖励、完成标志和信息。"""
        if self.done:
            raise ValueError("Environment is done, call reset() first")
        
        # 添加氨基酸到序列
        if action < len(self.aa_alphabet):
            aa = self.aa_alphabet[action]
            self.current_sequence.append(aa)
        
        # 检查序列是否完成
        sequence = ''.join(self.current_sequence)
        if len(sequence) >= self.max_len or action == len(self.aa_alphabet):  # 停止标记
            self.done = True
            reward = self._compute_enhanced_final_reward(sequence)
        else:
            reward = 0.0  # 中间步骤奖励
        
        next_state = self._get_state()
        info = {
            "sequence": sequence, 
            "length": len(sequence),
            "receptor": self.target_receptor
        }
        
        return next_state, reward, self.done, info
    
    def _get_state(self) -> torch.Tensor:
        """获取当前状态张量。"""
        sequence = ''.join(self.current_sequence)
        encoded = self._encode_sequence(sequence)
        return torch.tensor(encoded, dtype=torch.long)
    
    def _encode_sequence(self, sequence: str) -> List[int]:
        """编码序列为氨基酸索引。"""
        encoded = []
        for aa in sequence:
            if aa in self.aa_alphabet:
                encoded.append(self.aa_alphabet.index(aa) + 1)  # 1-indexed
            else:
                encoded.append(0)  # Unknown/padding
        
        # 填充到最大长度
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))
        
        return encoded
    
    def _compute_enhanced_final_reward(self, sequence: str) -> float:
        """计算增强的最终奖励，整合受体特征。"""
        if len(sequence) < self.min_len:
            return -1.0  # 太短序列的惩罚
        
        # 清理序列
        cleaned = clean_sequence(sequence, strict=False)
        if not cleaned:
            return -1.0
        
        # 计算奖励组件
        rewards = {}
        
        # 1. 目标受体概率（使用增强分类器）
        with torch.no_grad():
            encoded = torch.tensor(self._encode_sequence(cleaned)).unsqueeze(0)
            # 移动张量到分类器相同的设备
            device = next(self.classifier.parameters()).device
            encoded = encoded.to(device)
            
            if hasattr(self.classifier, 'predict_with_receptor'):
                # 使用增强分类器
                target_prob = self.classifier.predict_with_receptor([cleaned], self.target_receptor, device)[0].item()
            else:
                # 回退到基础分类器
                probs = self.classifier(encoded)
                target_prob = probs[0, 1].item()
            
            rewards["target_prob"] = target_prob
        
        # 2. 特异性奖励（相对于其他受体）
        rewards["specificity"] = self._compute_specificity_reward(cleaned)
        
        # 3. 理化约束
        rewards["physchem"] = self._compute_enhanced_physchem_reward(cleaned)
        
        # 4. 基序分数（基于受体特征）
        rewards["motif"] = self._compute_motif_reward(cleaned)
        
        # 5. 安全性分数
        rewards["safety"] = self._compute_safety_reward(cleaned)
        
        # 6. 受体兼容性奖励
        rewards["receptor_compatibility"] = self._compute_receptor_compatibility_reward(cleaned)
        
        # 加权求和
        total_reward = 0.0
        for component, weight in self.reward_weights.items():
            if component in rewards:
                total_reward += weight * rewards[component]
        
        return total_reward
    
    def _compute_specificity_reward(self, sequence: str) -> float:
        """计算特异性奖励（相对于其他受体）。"""
        if self.target_receptor not in self.receptor_features:
            return 0.5  # 默认值
        
        try:
            # 使用增强分类器计算多受体概率
            if hasattr(self.classifier, 'predict_with_receptor'):
                device = next(self.classifier.parameters()).device
                
                target_prob = self.classifier.predict_with_receptor([sequence], self.target_receptor, device)[0].item()
                
                # 计算对其他受体的平均概率
                other_receptors = [r for r in ['TFRC', 'INSR', 'LRP1', 'ICAM1', 'NCAM1'] if r != self.target_receptor]
                other_probs = []
                
                for receptor in other_receptors:
                    prob = self.classifier.predict_with_receptor([sequence], receptor, device)[0].item()
                    other_probs.append(prob)
                
                if other_probs:
                    avg_other_prob = np.mean(other_probs)
                    specificity = max(0, target_prob - avg_other_prob)
                    return min(specificity * 2, 1.0)  # 缩放并限制在[0,1]
            
        except Exception as e:
            logger.warning(f"Specificity computation failed: {e}")
        
        return 0.5  # 默认值
    
    def _compute_enhanced_physchem_reward(self, sequence: str) -> float:
        """计算增强的理化奖励，考虑受体特征。"""
        score = 0.0
        
        # 长度约束
        if self.min_len <= len(sequence) <= self.max_len:
            score += 0.2
        
        # 半胱氨酸惩罚
        cysteine_count = sequence.count('C')
        if cysteine_count <= self.physchem_constraints["max_cysteines"]:
            score += 0.2
        else:
            score -= 0.1 * (cysteine_count - self.physchem_constraints["max_cysteines"])
        
        # 重复惩罚（简化）
        max_repeat = max(len(sequence) - len(set(sequence)), 0)
        if max_repeat <= self.physchem_constraints["max_repeats"]:
            score += 0.2
        else:
            score -= 0.1 * (max_repeat - self.physchem_constraints["max_repeats"])
        
        # 基于受体特征的疏水性兼容性
        hydrophobicity_compatibility = self._compute_hydrophobicity_compatibility(sequence)
        score += 0.1 * hydrophobicity_compatibility
        
        return min(max(score, 0.0), 1.0)
    
    def _compute_hydrophobicity_compatibility(self, sequence: str) -> float:
        """计算肽段与受体的疏水性兼容性。"""
        if self.target_receptor not in self.receptor_features:
            return 0.5
        
        try:
            # 计算肽段的平均疏水性
            parser = FastaParser(self.config["paths"]["data_dir"])
            peptide_features = parser.extract_sequence_features(sequence)
            peptide_hydropathy = peptide_features['physchem_properties']['avg_hydropathy']
            
            # 获取受体的平均疏水性
            receptor_hydropathy = self.receptor_features[self.target_receptor]['features']['physchem_properties']['avg_hydropathy']
            
            # 计算兼容性（相似但不完全相同）
            compatibility = 1.0 - min(abs(peptide_hydropathy - receptor_hydropathy) / 4.0, 1.0)
            return compatibility
            
        except Exception as e:
            logger.warning(f"Hydrophobicity compatibility computation failed: {e}")
            return 0.5
    
    def _compute_motif_reward(self, sequence: str) -> float:
        """计算基于受体特征的基序奖励。"""
        score = 0.0
        
        # 通用基序
        if any(motif in sequence for motif in ["YPR", "THR", "TRP"]):
            score += 0.1
        
        # 受体特定基序（基于受体序列特征）
        receptor_specific_motifs = self._get_receptor_specific_motifs()
        for motif in receptor_specific_motifs:
            if motif in sequence:
                score += 0.05
        
        return min(score, 0.2)  # 限制基序奖励
    
    def _get_receptor_specific_motifs(self) -> List[str]:
        """获取受体特定的基序模式。"""
        # 简化的基序映射
        receptor_motifs = {
            'TFRC': ['RGD', 'NGR', 'KDEL'],
            'INSR': ['YMNM', 'NPXY'],
            'LRP1': ['NPXY', 'FDNPVY'],
            'ICAM1': ['LDV', 'RGD'],
            'NCAM1': ['HNK', 'LDV']
        }
        
        return receptor_motifs.get(self.target_receptor, [])
    
    def _compute_safety_reward(self, sequence: str) -> float:
        """计算安全性奖励。"""
        score = 0.1  # 基础安全性分数
        
        # 半胱氨酸惩罚（可能形成二硫键导致聚集）
        if "C" not in sequence:
            score += 0.1
        
        # 避免已知毒性模式
        toxic_patterns = ["KKKK", "RRRR", "PPPP"]  # 多聚碱性或脯氨酸重复
        if not any(pattern in sequence for pattern in toxic_patterns):
            score += 0.1
        
        return min(score, 0.3)
    
    def _compute_receptor_compatibility_reward(self, sequence: str) -> float:
        """计算受体兼容性奖励。"""
        if self.target_receptor not in self.receptor_features:
            return 0.0
        
        try:
            # 基于受体序列特征的兼容性计算
            receptor_features = self.receptor_features[self.target_receptor]['features']
            
            # 计算电荷兼容性
            charge_compatibility = self._compute_charge_compatibility(sequence, receptor_features)
            
            # 计算复杂性兼容性
            complexity_compatibility = self._compute_complexity_compatibility(sequence, receptor_features)
            
            return 0.3 * charge_compatibility + 0.2 * complexity_compatibility
            
        except Exception as e:
            logger.warning(f"Receptor compatibility computation failed: {e}")
            return 0.0
    
    def _compute_charge_compatibility(self, sequence: str, receptor_features: Dict) -> float:
        """计算电荷兼容性。"""
        parser = FastaParser(self.config["paths"]["data_dir"])
        peptide_features = parser.extract_sequence_features(sequence)
        
        peptide_charge = peptide_features['physchem_properties']['net_charge']
        receptor_charge = receptor_features['physchem_properties']['net_charge']
        
        # 电荷互补性（相反电荷有利于结合）
        charge_complementarity = 1.0 - min(abs(peptide_charge + receptor_charge) / 10.0, 1.0)
        return charge_complementarity
    
    def _compute_complexity_compatibility(self, sequence: str, receptor_features: Dict) -> float:
        """计算复杂性兼容性。"""
        parser = FastaParser(self.config["paths"]["data_dir"])
        peptide_features = parser.extract_sequence_features(sequence)
        
        peptide_complexity = peptide_features['sequence_complexity']['complexity_score']
        receptor_complexity = receptor_features['sequence_complexity']['complexity_score']
        
        # 复杂性相似性
        complexity_similarity = 1.0 - abs(peptide_complexity - receptor_complexity)
        return max(complexity_similarity, 0.0)


def create_enhanced_environment(config: Dict, target_receptor: str, classifier: nn.Module):
    """创建增强的环境实例。"""
    return EnhancedPeptideEnvironment(config, target_receptor, classifier)
