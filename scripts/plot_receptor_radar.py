#!/usr/bin/env python3
"""
受体特征雷达图可视化工具。
"""

import math
import matplotlib.pyplot as plt
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.receptors.features import RECEPTOR_DB


def plot_receptor_radar(out_png: str = "results/receptor_radar.png"):
    """
    绘制受体特征雷达图。
    
    Args:
        out_png: 输出图片路径
    """
    # 设置matplotlib后端为非交互式
    plt.switch_backend('Agg')
    
    # 特征标签
    labels = ["Hydropathy", "Charge", "Complexity", "Binding Site Charge", "Binding Site Hydropathy"]
    n_vars = len(labels)
    
    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})
    
    # 为每个受体绘制雷达图
    for name, feat in RECEPTOR_DB.items():
        # 归一化特征值用于雷达图
        values = [
            (feat.hydropathy + 1.0) / 2.0,  # 疏水性: [-1,1] -> [0,1]
            min(abs(feat.charge) / 200.0, 1.0),  # 电荷: 绝对值归一化
            feat.complexity * 20,  # 复杂性: 放大显示
            min(abs(feat.binding_site_charge) / 10.0, 1.0),  # 结合位点电荷
            (feat.binding_site_hydropathy + 1.0) / 2.0  # 结合位点疏水性
        ]
        
        # 闭合雷达图
        angles = [i / float(n_vars) * 2 * math.pi for i in range(n_vars)]
        values += values[:1]
        angles += angles[:1]
        
        # 绘制线条和填充
        ax.plot(angles, values, 'o-', linewidth=2, label=name, markersize=4)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置角度标签
    ax.set_xticks([i / float(n_vars) * 2 * math.pi for i in range(n_vars)])
    ax.set_xticklabels(labels)
    
    # 设置径向标签
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1)
    
    # 添加图例和标题
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("Receptor Features Radar Plot", size=16, fontweight='bold', pad=20)
    
    # 保存图片
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[plot_receptor_radar] Radar plot saved to {out_png}")
    
    plt.close()


def plot_receptor_comparison(out_png: str = "results/receptor_comparison.png"):
    """
    绘制受体特征比较图。
    
    Args:
        out_png: 输出图片路径
    """
    # 设置matplotlib后端为非交互式
    plt.switch_backend('Agg')
    
    # 特征列表
    features = ["hydropathy", "charge", "complexity", "binding_site_charge", "binding_site_hydropathy"]
    feature_names = ["Hydropathy", "Charge", "Complexity", "Binding Site Charge", "Binding Site Hydropathy"]
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 为每个特征绘制条形图
    for i, (feature, feature_name) in enumerate(zip(features, feature_names)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        receptor_names = []
        feature_values = []
        
        for name, feat in RECEPTOR_DB.items():
            receptor_names.append(name)
            feature_values.append(getattr(feat, feature))
        
        # 绘制条形图
        bars = ax.bar(receptor_names, feature_values, color=plt.cm.Set3(range(len(receptor_names))))
        ax.set_title(feature_name, fontweight='bold')
        ax.set_ylabel("Value")
        
        # 添加数值标签
        for bar, value in zip(bars, feature_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(feature_values),
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 旋转x轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 隐藏多余的子图
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    # 添加总标题
    fig.suptitle("Receptor Features Comparison", fontsize=16, fontweight='bold')
    
    # 保存图片
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[plot_receptor_comparison] Comparison plot saved to {out_png}")
    
    plt.close()


def main():
    """主函数"""
    try:
        # 绘制雷达图
        plot_receptor_radar()
        
        # 绘制比较图
        plot_receptor_comparison()
        
        print("[plot_receptor_radar] All visualizations completed successfully")
        
    except ImportError as e:
        print(f"[plot_receptor_radar] Warning: matplotlib not available: {e}")
        print("[plot_receptor_radar] Skipping visualization")
    except Exception as e:
        print(f"[plot_receptor_radar] Error during visualization: {e}")


if __name__ == "__main__":
    main()
