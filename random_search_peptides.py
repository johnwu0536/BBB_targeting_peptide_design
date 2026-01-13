#!/usr/bin/env python3
"""
Random search for BBB-peptide RL hyperparameters.

功能：
- 读取 base config.yaml
- 随机扰动若干关键参数（熵、多样性相关）
- 生成临时 config_trial_XX.yaml
- 调用 RL 训练脚本
- 从 stdout / log 中解析一个指标（占位实现），
  并把 trial + 参数 + 指标记录到 TXT 文件

用法：
    python random_search_peptides.py --trials 20 --target TFRC
"""

import os
import sys
import yaml
import random
import argparse
import subprocess
from copy import deepcopy
from datetime import datetime

LOG_TXT = "results/random_search_log.txt"


# ---------- 工具函数：嵌套 key 读写 ----------

def get_nested(d, key_path, default=None):
    """按 'a.b.c' 形式读取嵌套字典."""
    cur = d
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_nested(d, key_path, value):
    """按 'a.b.c' 形式设置嵌套字典."""
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


# ---------- 随机采样函数 ----------

def sample_log_uniform(low, high):
    import math
    log_low, log_high = math.log(low), math.log(high)
    r = random.random() * (log_high - log_low) + log_low
    return math.exp(r)


def sample_param(spec):
    """
    spec: dict, e.g.
      {"type": "uniform", "low": 0.0, "high": 0.1}
      {"type": "choice", "values": [2,3,4]}
      {"type": "log_uniform", "low": 1e-6, "high": 1e-2}
    """
    t = spec["type"]
    if t == "uniform":
        return random.uniform(spec["low"], spec["high"])
    elif t == "int":
        return random.randint(spec["low"], spec["high"])
    elif t == "choice":
        return random.choice(spec["values"])
    elif t == "log_uniform":
        return sample_log_uniform(spec["low"], spec["high"])
    else:
        raise ValueError(f"Unknown param type: {t}")


# ---------- 定义要随机的参数空间 ----------

PARAM_SPACE = {
    # 1. RL 熵系数：鼓励探索（防止序列塌缩）
    "rl.entropy_coef": {
        "type": "uniform",
        "low": 0.01,
        "high": 0.10,  # 原来 0.05，可以在 0.01-0.1 之间随机
    },

    # 2. 最大重复 run：允许 2-4 连续相同氨基酸
    "physchem_constraints.max_repeats": {
        "type": "choice",
        "values": [2, 3, 4],
    },

    # 3. 序列熵下限：提高复杂度要求
    "physchem_constraints.min_entropy": {
        "type": "uniform",
        "low": 2.0,
        "high": 4.0,
    },

    # 4. 熵不足惩罚强度
    "physchem_constraints.entropy_penalty_coef": {
        "type": "uniform",
        "low": 3.0,
        "high": 10.0,
    },

    # 5. 物化约束在 reward 中的权重
    "reward_weights.physchem": {
        "type": "uniform",
        "low": 0.2,
        "high": 0.5,
    },

    # 6. target_prob 权重同步微调，保证和 physchem 平衡
    "reward_weights.target_prob": {
        "type": "uniform",
        "low": 0.25,
        "high": 0.5,
    },

    # 7. active learning 多样性权重
    "active_selection.ucb.diversity_weight": {
        "type": "uniform",
        "low": 0.05,
        "high": 0.5,
    },

    # 8. KL 惩罚系数：避免更新过保守/过激
    "rl_enhanced.kl_penalty.beta": {
        "type": "uniform",
        "low": 0.005,
        "high": 0.05,
    },
}


# ---------- 运行单个 trial ----------

def run_trial(trial_id, base_config, args):
    """
    - 从 base_config 复制一份
    - 随机采样 PARAM_SPACE
    - 写 config_trial_X.yaml
    - 调用 RL 训练命令
    - 解析返回指标（此处占位实现）
    - 把结果写入 TXT
    """

    cfg = deepcopy(base_config)
    changed = {}

    # 1. 采样并写入 config dict
    for key, spec in PARAM_SPACE.items():
        val = sample_param(spec)
        # 一些参数要 round 一下，避免太多小数
        if isinstance(val, float):
            val = float(f"{val:.4g}")
        set_nested(cfg, key, val)
        changed[key] = val

    # 2. 生成本次 config 文件名
    cfg_name = f"config_trial_{trial_id:03d}.yaml"
    with open(cfg_name, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # 3. 要运行的命令（你可以根据需要改）
    # 选项 A：只跑 RL（建议用于快速测试）：
    cmd = f"python -m src.train_rl_ppo --config {cfg_name} --target {args.target}"

    # 选项 B：跑整条 pipeline（非常耗时），自行切换：
    # cmd = f"python pipeline.py --target {args.target} --config {cfg_name}"

    print(f"\n[Trial {trial_id}] Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # 4. 简单解析一个“指标”（你可以根据 stdout 改 parser）
    stdout = proc.stdout
    stderr = proc.stderr
    returncode = proc.returncode

    # 占位：如果你的训练里有类似 "Best reward: 1.234" 的 log，可以这样 parse
    import re
    metric = None
    match = re.search(r"Best reward[:=]\s*([-\d\.eE]+)", stdout)
    if match:
        metric = float(match.group(1))
    else:
        # 没找到就用 returncode 的倒数瞎占位（你可以改成别的）
        metric = -1.0 if returncode != 0 else 0.0

    # 5. 将结果写入 TXT
    os.makedirs(os.path.dirname(LOG_TXT), exist_ok=True)
    with open(LOG_TXT, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Trial {trial_id} | time = {datetime.now().isoformat()}\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"Return code: {returncode}\n")
        f.write(f"Parsed metric (Best reward): {metric}\n")
        f.write("Changed parameters:\n")
        for k, v in changed.items():
            origin_val = get_nested(base_config, k, "<not in base>")
            f.write(f"  - {k}: {origin_val}  ->  {v}\n")
        f.write("\nSTDOUT:\n")
        f.write(stdout[:4000] + ("\n...[truncated]\n" if len(stdout) > 4000 else "\n"))
        if stderr:
            f.write("\nSTDERR:\n")
            f.write(stderr[:4000] + ("\n...[truncated]\n" if len(stderr) > 4000 else "\n"))

    print(f"[Trial {trial_id}] Done. Metric={metric}, log appended to {LOG_TXT}")
    return metric, changed


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--target", type=str, default="LRP1")
    args = parser.parse_args()

    # 读取 base config
    if not os.path.exists(args.config):
        print(f"Base config {args.config} not found.", file=sys.stderr)
        return 1

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # 提醒：如果 YAML 里有重复 key（比如 classifier），后面的会覆盖前面的
    if isinstance(base_config, dict) and "classifier" in base_config:
        # 这里不自动修，但提醒一下
        print("[Warning] 'classifier' key appears only once in loaded config. "
              "If你在yaml里写了两次classifier, 前面的已经被覆盖。请手工合并。")

    print(f"Loaded base config from {args.config}")
    print(f"Will run {args.trials} random trials for target={args.target}")
    print(f"Log file: {LOG_TXT}")

    best_metric = None
    best_trial = None
    best_params = None

    for t in range(1, args.trials + 1):
        metric, changed = run_trial(t, base_config, args)
        if (best_metric is None) or (metric > best_metric):
            best_metric = metric
            best_trial = t
            best_params = changed

    print("\n=== Random search finished ===")
    print(f"Best trial: {best_trial} | metric={best_metric}")
    print("Best trial changed parameters:")
    for k, v in (best_params or {}).items():
        print(f"  - {k} = {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

