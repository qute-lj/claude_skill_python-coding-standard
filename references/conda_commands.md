# Conda 环境管理 Bash 命令

## 运行前必做的环境检查

### 1. 检查所有可用的 conda 环境
```bash
conda env list
```

### 2. 查看当前激活的环境
```bash
conda info --envs
# 或查看当前环境详情
conda info
```

### 3. 查看指定环境的 Python 版本
```bash
conda run -n your_env_name python --version
```

### 4. 查看环境中安装的包
```bash
conda list -n your_env_name
# 或查看前20个
conda list -n your_env_name | head -20
```

## 环境使用（不使用conda activate）

### ⚠️ 重要：避免使用conda activate

**不推荐使用conda activate的原因：**
- 需要预先运行conda init，修改shell配置
- 可能导致环境冲突和依赖问题
- 在脚本中难以管理和重现

### 1. 直接使用指定环境运行 Python 脚本（推荐）
```bash
# 不需要激活环境，直接运行
conda run -n your_env_name python your_script.py

# 带参数
conda run -n your_env_name python train.py --epochs 100 --batch-size 32

# 交互式运行
conda run -n your_env_name python

# 运行Jupyter notebook
conda run -n your_env_name jupyter notebook
```

### 2. 在指定环境中安装包
```bash
# 使用conda安装到指定环境（强烈推荐）
conda install -n your_env_name numpy matplotlib scipy

# 从conda-forge安装（当主频道没有时）
conda install -n your_env_name -c conda-forge package-name

# 安装特定版本
conda install -n your_env_name numpy=1.21.0

# ⚠️ 仅当conda源中确实没有时才使用pip
conda run -n your_env_name pip install some-package-only-in-pip

# 从 requirements.txt 安装（尽量用conda环境文件替代）
conda run -n your_env_name pip install -r requirements.txt
```

## 环境创建和管理

### 1. 创建新环境
```bash
# 创建指定 Python 版本的环境
conda create -n new_env python=3.9 -y

# 创建带包的环境
conda create -n science_env python=3.9 numpy scipy matplotlib jupyter -y

# 从 environment.yml 创建
conda env create -f environment.yml
```

### 2. 克隆环境
```bash
conda create --name new_env --clone old_env
```

### 3. 导出环境
```bash
# 导出为 yml 文件
conda env export > environment.yml

# 导出为 txt 文件（仅包名）
conda list --export > requirements.txt
```

### 4. 删除环境
```bash
conda env remove -n env_name -y
```

## 常用诊断命令

### 1. 检查 conda 是否正确安装
```bash
conda --version
```

### 2. 检查 conda 配置
```bash
conda config --show
```

### 3. 清理未使用的包和缓存
```bash
conda clean --all -y
```

### 4. 更新 conda
```bash
conda update conda -y
```

## 推荐工作流程（仅使用conda run）

### 标准 workflow（推荐）
```bash
# 1. 检查环境
conda env list

# 2. 直接运行脚本（无需激活）
conda run -n my_project_env python main.py

# 3. 安装新包（优先使用conda -n）
conda install -n my_project_env new-package

# 4. 仅当conda源中没有时才使用pip
conda run -n my_project_env pip install new-package-only-in-pip

# 5. 运行Jupyter notebook
conda run -n my_project_env jupyter notebook

# 6. 交互式Python
conda run -n my_project_env python
```

## 问题排查

### 1. 如果命令未找到
```bash
# 检查 conda 是否在 PATH
which conda    # Linux/Mac
where conda    # Windows

# 确保conda已正确安装并可用
conda --version
```

### 2. 环境访问问题
```bash
# 检查环境是否存在
conda env list

# 如果环境不存在，创建它
conda create -n env_name python=3.9 -y
```

### 3. 查看 Python 路径
```bash
# 直接查看指定环境的Python路径
conda run -n env_name which python    # Linux/Mac
conda run -n env_name where python    # Windows

# 查看Python版本
conda run -n env_name python --version
```

## 最佳实践

1. **避免conda activate**：始终使用 `conda run -n env_name` 和 `conda install -n env_name`
2. **明确指定环境**：所有命令都要带 `-n env_name` 参数
3. **项目隔离**：每个项目使用独立环境
4. **版本锁定**：使用 `environment.yml` 或 `requirements.txt` 记录依赖
5. **强烈优先conda安装**：避免使用pip，除非包确实不在conda源中
6. **定期清理**：使用 `conda clean --all` 清理缓存

## 示例：完整的项目设置流程

```bash
# 1. 查看现有环境
conda env list

# 2. 创建新项目环境
conda create -n my_project python=3.9 numpy scipy matplotlib pandas -y

# 3. 验证环境
conda run -n my_project python --version

# 4. 安装项目依赖（优先使用conda环境文件）
conda env update -f environment.yml
# 或仅在必要时使用pip
conda run -n my_project pip install -r requirements.txt

# 5. 运行项目
conda run -n my_project python main.py

# 6. 导出环境（供其他开发者使用）
conda env export > environment.yml
```