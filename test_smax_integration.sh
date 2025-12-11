#!/bin/bash
# SMAX 集成测试脚本

echo "=========================================="
echo "SMAX Integration Test"
echo "=========================================="

# 检查文件是否存在
echo -e "\n1. Checking SMAX files..."
files=(
    "env/smax/SMAX.py"
    "env/smax/__init__.py"
    "configs/dreamer/smax/SMAXAgentConfig.py"
    "configs/dreamer/smax/SMAXLearnerConfig.py"
    "configs/dreamer/smax/SMAXControllerConfig.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file NOT found"
        all_exist=false
    fi
done

# 检查关键修改
echo -e "\n2. Checking key modifications..."

# 检查 environments.py
if grep -q "SMAX = \"smax\"" environments.py; then
    echo "  ✓ environments.py: SMAX enum added"
else
    echo "  ✗ environments.py: SMAX enum NOT found"
    all_exist=false
fi

# 检查 EnvConfigs.py
if grep -q "class SMAXConfig" configs/EnvConfigs.py; then
    echo "  ✓ EnvConfigs.py: SMAXConfig class added"
else
    echo "  ✗ EnvConfigs.py: SMAXConfig class NOT found"
    all_exist=false
fi

# 检查 train.py
if grep -q "from configs.dreamer.smax.SMAXLearnerConfig import SMAXDreamerLearnerConfig" train.py; then
    echo "  ✓ train.py: SMAX imports added"
else
    echo "  ✗ train.py: SMAX imports NOT found"
    all_exist=false
fi

if grep -q "def prepare_smax_configs" train.py; then
    echo "  ✓ train.py: prepare_smax_configs function added"
else
    echo "  ✗ train.py: prepare_smax_configs function NOT found"
    all_exist=false
fi

# 检查 DreamerRunner.py
if grep -q "def evaluate" agent/runners/DreamerRunner.py; then
    echo "  ✓ DreamerRunner.py: evaluate method added"
else
    echo "  ✗ DreamerRunner.py: evaluate method NOT found"
    all_exist=false
fi

if grep -q "pickle.dump" agent/runners/DreamerRunner.py; then
    echo "  ✓ DreamerRunner.py: pkl data storage added"
else
    echo "  ✗ DreamerRunner.py: pkl data storage NOT found"
    all_exist=false
fi

# 检查 DreamerLearner.py
if grep -q "from tqdm import tqdm" agent/learners/DreamerLearner.py; then
    echo "  ✓ DreamerLearner.py: tqdm import added"
else
    echo "  ✗ DreamerLearner.py: tqdm import NOT found"
    all_exist=false
fi

if grep -q "def save" agent/learners/DreamerLearner.py; then
    echo "  ✓ DreamerLearner.py: save method added"
else
    echo "  ✗ DreamerLearner.py: save method NOT found"
    all_exist=false
fi

# 最终结果
echo -e "\n=========================================="
if [ "$all_exist" = true ]; then
    echo "✓ All checks passed!"
    echo "Integration appears to be successful."
else
    echo "✗ Some checks failed."
    echo "Please review the integration."
fi
echo "=========================================="

# 显示使用示例
echo -e "\n3. Usage example:"
echo "  python train.py --env smax --env_name 3m --n_workers 2 --seed 1 --steps 10000 --mode disabled"
echo ""
