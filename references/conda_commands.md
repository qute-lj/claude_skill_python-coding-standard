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

## 环境激活和使用

### 1. 激活环境（在终端执行）
```bash
# Linux/Mac
conda activate your_env_name

# Windows (CMD/PowerShell)
conda activate your_env_name
```

### 2. 直接使用指定环境运行 Python 脚本
```bash
# 不需要激活环境，直接运行
conda run -n your_env_name python your_script.py

# 带参数
conda run -n your_env_name python train.py --epochs 100 --batch-size 32

# 交互式运行
conda run -n your_env_name python
```

### 3. 在指定环境中安装包
```bash
# 安装到指定环境（无需激活）
conda install -n your_env_name numpy matplotlib

# 使用 pip
conda run -n your_env_name pip install some-package

# 从 requirements.txt 安装
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

## 快速工作流程

### 方案 1：激活后运行
```bash
# 1. 检查环境
conda env list

# 2. 激活环境
conda activate my_project_env

# 3. 运行脚本
python main.py

# 4. 安装新包（如需要）
conda install new-package
```

### 方案 2：直接运行（推荐）
```bash
# 1. 直接运行脚本
conda run -n my_project_env python main.py

# 2. 安装包
conda run -n my_project_env pip install new-package

# 3. Jupyter notebook
conda run -n my_project_env jupyter notebook
```

## 问题排查

### 1. 如果命令未找到
```bash
# 检查 conda 是否在 PATH
which conda    # Linux/Mac
where conda    # Windows

# 如果没有，初始化 conda
conda init bash
# 或
conda init cmd
```

### 2. 如果环境激活失败
```bash
# 查看 shell 配置
echo $SHELL

# 重新初始化
conda init <your-shell>
# 然后重启终端
```

### 3. 查看 Python 路径
```bash
# 在激活的环境中
which python    # Linux/Mac
where python    # Windows

# 或直接运行
conda run -n env_name which python
```

## 最佳实践

1. **始终指定环境名**：使用 `conda run -n env_name` 而不是依赖激活状态
2. **项目隔离**：每个项目使用独立环境
3. **版本锁定**：使用 `environment.yml` 或 `requirements.txt` 记录依赖
4. **定期清理**：使用 `conda clean --all` 清理缓存
5. **避免使用 pip 在 conda 环境**：优先使用 `conda install`，除非包不在 conda

## 示例：完整的项目设置流程

```bash
# 1. 查看现有环境
conda env list

# 2. 创建新项目环境
conda create -n my_project python=3.9 numpy scipy matplotlib pandas -y

# 3. 验证环境
conda run -n my_project python --version

# 4. 安装项目依赖
conda run -n my_project pip install -r requirements.txt

# 5. 运行项目
conda run -n my_project python main.py

# 6. 导出环境（供其他开发者使用）
conda env export > environment.yml
```