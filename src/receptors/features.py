#!/usr/bin/env python3
"""
受体特征模块，提供受体理化特征和结构信息。
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ReceptorFeatures:
    """受体特征数据类。"""
    name: str
    hydropathy: float
    charge: float
    complexity: float
    binding_site_charge: float = 0.0
    binding_site_hydropathy: float = 0.0
    length: int = 0
    polarity_ratio: float = 0.0
    shannon_entropy: float = 0.0


# 受体特征数据库
RECEPTOR_DB: Dict[str, ReceptorFeatures] = {
    "TFRC": ReceptorFeatures(
        name="TFRC",
        hydropathy=-0.22,
        charge=+0.20,
        complexity=0.031,
        binding_site_charge=+1.5,
        binding_site_hydropathy=-0.15,
        length=640,
        polarity_ratio=0.500,
        shannon_entropy=4.138
    ),
    "INSR": ReceptorFeatures(
        name="INSR",
        hydropathy=-0.42,
        charge=-11.30,
        complexity=0.022,
        binding_site_charge=-2.0,
        binding_site_hydropathy=-0.25,
        length=917,
        polarity_ratio=0.525,
        shannon_entropy=4.201
    ),
    "LRP1": ReceptorFeatures(
        name="LRP1",
        hydropathy=-0.52,
        charge=-155.30,
        complexity=0.004,
        binding_site_charge=-5.0,
        binding_site_hydropathy=-0.35,
        length=4453,
        polarity_ratio=0.513,
        shannon_entropy=4.207
    ),
    "ICAM1": ReceptorFeatures(
        name="ICAM1",
        hydropathy=-0.34,
        charge=-4.50,
        complexity=0.044,
        binding_site_charge=-1.5,
        binding_site_hydropathy=-0.20,
        length=453,
        polarity_ratio=0.497,
        shannon_entropy=4.010
    ),
    "NCAM1": ReceptorFeatures(
        name="NCAM1",
        hydropathy=-0.42,
        charge=-32.20,
        complexity=0.028,
        binding_site_charge=-3.0,
        binding_site_hydropathy=-0.30,
        length=709,
        polarity_ratio=0.535,
        shannon_entropy=4.134
    )
}


def get_receptor_features(receptor_name: str) -> Optional[ReceptorFeatures]:
    """
    获取指定受体的特征。
    
    Args:
        receptor_name: 受体名称
        
    Returns:
        受体特征对象，如果不存在则返回None
    """
    if receptor_name in RECEPTOR_DB:
        return RECEPTOR_DB[receptor_name]
    else:
        logger.warning(f"Receptor {receptor_name} not found in database")
        return None


def compute_receptor_compatibility(peptide_props: Dict, receptor_features: ReceptorFeatures) -> float:
    """
    计算肽段与受体的兼容性分数。
    
    Args:
        peptide_props: 肽段理化性质字典
        receptor_features: 受体特征对象
        
    Returns:
        兼容性分数 (0-1)
    """
    # 电荷兼容性 (互补性)
    charge_compatibility = 1.0 - min(abs(receptor_features.charge + peptide_props.get("charge", 0)) / 100.0, 1.0)
    
    # 疏水性兼容性 (相似性)
    hydropathy_compatibility = 1.0 - min(abs(receptor_features.hydropathy - peptide_props.get("hydropathy", 0)) / 2.0, 1.0)
    
    # 复杂性兼容性 (相似性)
    complexity_compatibility = 1.0 - min(abs(receptor_features.complexity - peptide_props.get("complexity", 0)) / 0.05, 1.0)
    
    # 结合位点兼容性
    binding_site_charge_compat = 1.0 - min(abs(receptor_features.binding_site_charge + peptide_props.get("charge", 0)) / 10.0, 1.0)
    binding_site_hydropathy_compat = 1.0 - min(abs(receptor_features.binding_site_hydropathy - peptide_props.get("hydropathy", 0)) / 1.0, 1.0)
    
    # 加权平均
    compatibility_score = (
        0.3 * charge_compatibility +
        0.3 * hydropathy_compatibility +
        0.2 * complexity_compatibility +
        0.1 * binding_site_charge_compat +
        0.1 * binding_site_hydropathy_compat
    )
    
    return max(0.0, min(1.0, compatibility_score))


def load_structural_annotations(struct_file: str = "data/receptor_structures.json") -> Dict[str, Dict]:
    """
    加载结构注释数据。
    
    Args:
        struct_file: 结构注释文件路径
        
    Returns:
        结构注释字典
    """
    import json
    import os
    
    if not os.path.exists(struct_file):
        logger.warning(f"Structural annotations file {struct_file} not found")
        return {}
    
    try:
        with open(struct_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load structural annotations: {e}")
        return {}


def update_receptor_features_with_structure(struct_annotations: Dict[str, Dict]):
    """
    使用结构注释更新受体特征。
    
    Args:
        struct_annotations: 结构注释字典
    """
    for receptor_name, annotations in struct_annotations.items():
        if receptor_name in RECEPTOR_DB:
            # 更新结合位点特征
            if "binding_site" in annotations:
                binding_site = annotations["binding_site"]
                RECEPTOR_DB[receptor_name].binding_site_charge = binding_site.get("charge", 0.0)
                RECEPTOR_DB[receptor_name].binding_site_hydropathy = binding_site.get("hydropathy", 0.0)
            
            logger.info(f"Updated {receptor_name} with structural annotations")


# 初始化时尝试加载结构注释
try:
    struct_annotations = load_structural_annotations()
    if struct_annotations:
        update_receptor_features_with_structure(struct_annotations)
        logger.info("Loaded structural annotations for receptors")
except Exception as e:
    logger.warning(f"Failed to load structural annotations: {e}")


def print_receptor_summary():
    """打印受体特征摘要。"""
    print("Receptor Features Summary:")
    print("-" * 80)
    for name, features in RECEPTOR_DB.items():
        print(f"{name}:")
        print(f"  Hydropathy: {features.hydropathy:.2f}")
        print(f"  Charge: {features.charge:.2f}")
        print(f"  Complexity: {features.complexity:.3f}")
        print(f"  Binding Site Charge: {features.binding_site_charge:.2f}")
        print(f"  Binding Site Hydropathy: {features.binding_site_hydropathy:.2f}")
        print()


if __name__ == "__main__":
    print_receptor_summary()
