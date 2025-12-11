# SMAX 集成文档

## 概述
本文档记录了将 SMAX 环境从 mamba_smax 仓库集成到官方 mamba 仓库的过程。集成采用最小化方式，只添加了必要的 SMAX 支持，同时保留了论文所需的所有 print、logging 和 pkl 数据存储功能。

## 集成的文件和修改

### 1. 新增文件

#### 环境文件
- `env/smax/SMAX.py` - SMAX 环境封装类
- `env/smax/__init__.py` - SMAX 模块初始化

#### 配置文件
- `configs/dreamer/smax/SMAXAgentConfig.py` - SMAX Agent 基础配置
- `configs/dreamer/smax/SMAXLearnerConfig.py` - SMAX Learner 配置
- `configs/dreamer/smax/SMAXControllerConfig.py` - SMAX Controller 配置

### 2. 修改的文件

#### environments.py
**修改内容：**
- 在 `Env` 枚举中添加了 `SMAX = "smax"`

```python
class Env(str, Enum):
    FLATLAND = "flatland"
    STARCRAFT = "starcraft"
    SMAX = "smax"  # 新增
```

#### configs/EnvConfigs.py
**修改内容：**
1. 添加 SMAX 导入：
```python
from env.smax.SMAX import SMAX
```

2. 更新 StarCraftConfig 以支持 seed 参数：
```python
def __init__(self, env_name, seed=23):
    self.env_name = env_name
    self.seed = seed
```

3. 添加 SMAXConfig 类：
```python
class SMAXConfig(EnvConfig):
    def __init__(self, env_name, seed, **kwargs):
        self.env_name = env_name
        self.seed = seed
        self.kwargs = kwargs

    def create_env(self):
        return SMAX(self.env_name, self.seed, **self.kwargs)
```

#### train.py
**主要修改：**

1. **导入部分**：
   - 添加了必要的库导入（os, shutil, datetime, torch, numpy, random）
   - 添加了 SMAX 相关配置导入

2. **parse_args 函数**：
   - 添加了 `--seed`、`--steps`、`--mode` 参数

3. **train_dreamer 函数**：
   - 添加了 `save_interval` 和 `save_mode` 参数

4. **get_env_info 函数**：
   - 增强了环境信息提取功能，支持连续动作空间
   - 添加了详细的打印输出

5. **prepare_smax_configs 函数**（新增）：
```python
def prepare_smax_configs(env_name):
    agent_configs = [SMAXDreamerControllerConfig(), SMAXDreamerLearnerConfig()]
    env_config = SMAXConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}
```

6. **主函数**：
   - 添加了随机种子设置（torch, numpy, random）
   - 添加了 SMAX 环境分支
   - 添加了运行目录创建和代码备份逻辑
   - 添加了 wandb 初始化配置
   - 添加了详细的日志输出

#### agent/runners/DreamerRunner.py
**关键修改：**

1. **DreamerServer 类**：
   - 添加了评估相关的 worker 和任务管理
   - 添加了 `evaluate()` 方法用于定期评估

2. **DreamerRunner 类**：
   - 添加了 `save_path` 用于保存 pkl 数据
   - 添加了 `env_type` 用于环境类型判断
   - 在 `run()` 方法中添加了：
     - 评估逻辑（每 1000 步评估一次）
     - 模型保存逻辑（可配置的保存间隔）
     - 详细的训练过程打印输出（包括 win rate、returns、entropy等）
     - pkl 数据存储（保存 steps、eval_win_rates、eval_returns）

**重要的 pkl 数据存储：**
```python
stored_dict = {
    'steps': steps,
    'eval_win_rates': eval_win_rates,
    'eval_returns': eval_ret,
}
with open(self.save_path, 'wb') as f:
    pickle.dump(stored_dict, f)
```

#### agent/learners/DreamerLearner.py
**关键修改：**

1. **导入**：
   - 添加了 `tqdm` 用于显示训练进度

2. **__init__ 方法**：
   - 添加了 `self.env_type` 保存环境类型
   - 添加了 `torch.autograd.set_detect_anomaly(True)`
   - 支持动态的 `NUM_AGENTS` 配置
   - 添加了 `self.tqdm_vis` 用于控制进度条显示
   - 添加了初始化信息打印

3. **save 方法**（新增）：
   - 保存模型并打印保存路径

4. **step 方法**：
   - 添加了 buffer size 检查的打印输出
   - 使用 tqdm 显示训练进度
   - 添加了训练开始的提示信息

5. **train_agent 方法**：
   - 添加了对 SMAX 环境的支持（与 STARCRAFT 相同的处理逻辑）
   - 修复了 target_update 的逻辑

## 保留的关键功能

### 1. Print 输出
- **train.py**: 环境信息、运行目录、配置信息
- **DreamerRunner.py**: 
  - 每个 episode 的详细信息（episode 数、steps、win rate、returns、entropy）
  - 评估结果（eval_win_rate、eval_returns、平均 episode 长度）
- **DreamerLearner.py**: 
  - 初始化信息（环境类型、agent 数量、buffer 容量）
  - 训练进度信息
  - 模型保存路径

### 2. Logging (wandb)
- **训练指标**:
  - win/reward/scores（根据环境类型）
  - returns
  - eval_win_rate
  - eval_returns
  - Agent/Returns
  - Agent/val_loss
  - Agent/actor_loss

### 3. PKL 数据存储
保存在 `{RUN_DIR}/../mamba_{map_name}_seed{seed}.pkl`，包含：
- `steps`: 评估时的训练步数
- `eval_win_rates`: 评估的胜率/得分
- `eval_returns`: 评估的累积回报

这些数据对于论文绘图至关重要。

## 使用方法

### 运行 SMAX 训练
```bash
python train.py --env smax --env_name 3m --n_workers 4 --seed 1 --steps 1000000 --mode online
```

### 参数说明
- `--env smax`: 指定使用 SMAX 环境
- `--env_name`: SMAX 地图名称（如 "3m", "5m_vs_6m", "8m" 等）
- `--n_workers`: 并行 worker 数量
- `--seed`: 随机种子
- `--steps`: 总训练步数
- `--mode`: wandb 模式（online/offline/disabled）

## 最小化集成原则

本次集成遵循以下原则：
1. **只添加 SMAX 支持**：没有引入其他环境（如 MPE、GRF、MAMuJoCo）
2. **保持官方结构**：所有修改都基于官方 mamba 的代码结构
3. **保留重要功能**：完整保留了 print、logging 和 pkl 数据存储
4. **最小化修改**：只修改必要的文件，避免不必要的代码变更

## 依赖要求

SMAX 环境需要以下依赖：
- jax
- jaxlib
- jaxmarl
- gym

请确保这些包已安装在环境中。

## 注意事项

1. **JAX CPU 模式**: SMAX 环境配置为使用 CPU 模式运行 JAX，避免在 Ray workers 中出现 CUDA 初始化问题
2. **评估频率**: 默认每 1000 步进行一次评估
3. **模型保存**: 根据 `save_interval` 参数定期保存模型
4. **数据存储**: pkl 文件在训练结束时自动保存

## 测试建议

1. 先使用小规模地图（如 "3m"）测试
2. 确认 print 输出正常
3. 检查 wandb 日志是否正确记录
4. 验证 pkl 文件是否生成并包含正确数据
