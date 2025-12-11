# SMAX 集成完成 ✓

## 集成总结

已成功将 SMAX 环境从 `mamba_smax` 仓库以最小化方式集成到官方 `mamba` 仓库中。

### ✅ 完成的工作

1. **环境文件** - 复制了 SMAX 环境实现
   - `env/smax/SMAX.py`
   - `env/smax/__init__.py`

2. **配置文件** - 复制了 SMAX 专用配置
   - `configs/dreamer/smax/SMAXAgentConfig.py`
   - `configs/dreamer/smax/SMAXLearnerConfig.py`
   - `configs/dreamer/smax/SMAXControllerConfig.py`

3. **核心文件修改** - 添加了 SMAX 支持并保留了所有重要功能
   - `environments.py` - 添加 SMAX 枚举
   - `configs/EnvConfigs.py` - 添加 SMAXConfig 类
   - `train.py` - 添加 SMAX 训练逻辑、日志和文件保存
   - `agent/runners/DreamerRunner.py` - 添加评估逻辑和 **pkl 数据存储**
   - `agent/learners/DreamerLearner.py` - 添加多环境支持和详细日志

### 🎯 保留的关键功能（用于论文）

#### 1. Print 输出
- 训练过程中的详细信息（episode、steps、win rate、returns、entropy）
- 评估结果（eval_win_rate、eval_returns、episode 长度）
- 模型保存路径
- Buffer 和训练状态信息

#### 2. Wandb Logging
- win/reward/scores
- returns
- eval_win_rate
- eval_returns
- Agent/Returns
- Agent/val_loss
- Agent/actor_loss

#### 3. PKL 数据存储 ⭐ **重要**
保存位置: `{results_dir}/../mamba_{map_name}_seed{seed}.pkl`

包含内容:
```python
{
    'steps': [1000, 2000, 3000, ...],           # 评估时的训练步数
    'eval_win_rates': [0.5, 0.6, 0.7, ...],     # 评估胜率/得分
    'eval_returns': [100, 120, 150, ...]        # 评估累积回报
}
```

## 🚀 使用方法

### 基本命令
```bash
python train.py --env smax --env_name 3m --n_workers 2 --seed 1 --steps 1000000 --mode online
```

### 参数说明
- `--env smax` - 使用 SMAX 环境
- `--env_name` - 地图名称（3m, 5m_vs_6m, 8m 等）
- `--n_workers` - 并行 worker 数量（建议 2-8）
- `--seed` - 随机种子（用于可复现性）
- `--steps` - 总训练步数
- `--mode` - wandb 模式
  - `online` - 在线同步到 wandb
  - `offline` - 离线保存，稍后同步
  - `disabled` - 禁用 wandb

### 示例场景

**快速测试**（禁用 wandb）:
```bash
python train.py --env smax --env_name 3m --n_workers 2 --seed 1 --steps 10000 --mode disabled
```

**正式训练**（在线 wandb）:
```bash
python train.py --env smax --env_name 5m_vs_6m --n_workers 4 --seed 42 --steps 2000000 --mode online
```

**多种子实验**:
```bash
for seed in 1 2 3 4 5; do
    python train.py --env smax --env_name 8m --n_workers 4 --seed $seed --steps 1000000 --mode online
done
```

## 📊 输出文件

### 训练过程中
- **模型检查点**: `{date}_results/smax/{env_name}/run{N}/ckpt/model_{K}Ksteps.pth`
- **代码备份**: `{date}_results/smax/{env_name}/run{N}/[agent,configs,networks,train.py]`

### 训练结束后
- **最终模型**: `{date}_results/smax/{env_name}/run{N}/ckpt/model_final.pth`
- **PKL 数据**: `{date}_results/smax/mamba_{env_name}_seed{seed}.pkl` ⭐

## 📈 论文绘图数据

使用 PKL 文件绘制学习曲线:

```python
import pickle
import matplotlib.pyplot as plt

# 加载数据
with open('1211_results/smax/mamba_3m_seed123.pkl', 'rb') as f:
    data = pickle.load(f)

# 绘制学习曲线
plt.plot(data['steps'], data['eval_win_rates'], label='Win Rate')
plt.plot(data['steps'], data['eval_returns'], label='Returns')
plt.xlabel('Training Steps')
plt.ylabel('Performance')
plt.legend()
plt.savefig('learning_curve.png')
```

## ⚙️ 配置调整

如需调整训练参数，编辑 `configs/dreamer/smax/SMAXLearnerConfig.py`:
- `MODEL_LR` - 模型学习率
- `ACTOR_LR` - Actor 学习率
- `VALUE_LR` - Critic 学习率
- `CAPACITY` - Replay buffer 容量
- `N_SAMPLES` - 训练间隔
- `MODEL_EPOCHS` - 模型训练轮数
- `PPO_EPOCHS` - PPO 训练轮数

## 🔍 验证集成

运行测试脚本:
```bash
./test_smax_integration.sh
```

应该看到所有检查项都显示 ✓。

## 📝 注意事项

1. **依赖要求**: 确保已安装 `jax`, `jaxlib`, `jaxmarl`
2. **评估频率**: 默认每 1000 步评估一次（10 个 episode）
3. **保存间隔**: 默认每 200K 步保存一次模型
4. **内存使用**: SMAX 使用 CPU 模式的 JAX 避免 CUDA 问题

## 🎓 集成原则

本次集成严格遵循以下原则:
- ✅ **最小化修改** - 只添加 SMAX 支持
- ✅ **保留官方结构** - 基于官方 mamba 代码架构
- ✅ **保持兼容性** - 不影响原有 Flatland 和 StarCraft 功能
- ✅ **完整功能** - 保留所有 print、logging 和数据存储

## 📚 更多文档

详细的集成文档请查看: `SMAX_INTEGRATION.md`

---

**集成完成时间**: 2025-12-11  
**集成方式**: 最小化 SMAX 支持  
**保留功能**: ✓ Print ✓ Logging ✓ PKL 数据存储
