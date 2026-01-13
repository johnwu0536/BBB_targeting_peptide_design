#!/usr/bin/env python3
"""
测试导入修复。
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.models.classifier_factory import build_classifier, log_model_config, count_parameters
    print("[OK] Import successful!")
    
    # 测试 count_parameters 函数
    import torch.nn as nn
    test_model = nn.Linear(10, 2)
    n_params = count_parameters(test_model)
    print(f"[OK] count_parameters works: {n_params} parameters")
    
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
except Exception as e:
    print(f"[ERROR] Other error: {e}")
