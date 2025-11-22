# Python 科学计算代码书写规范

> 🎯 专为科研工作者打造的专业 Python 编码标准，Claude Skill 认证

[![Skill Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/qute-lj/python-coding-standard)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 专为科学计算设计的 Python 代码规范和最佳实践，包含环境管理、输出文件规范、可视化、日志记录和数值误差分析。

## 📖 概述

本规范为科学计算 Python 代码提供全面的编码标准，涵盖从环境设置到结果输出的完整流程。作为 Claude Skill，可在需要编写专业科学计算代码时自动应用。

## 🎯 适用场景

当用户需要：
- 设置和管理 Python conda 环境
- 建立统一的输出文件管理体系
- 创建专业的科学可视化图表
- 实现规范的日志记录系统
- 分析和控制数值计算误差
- 编写符合科研标准的 Python 代码

## 📁 项目结构

```
python-coding-standard/
├── SKILL.md                    # Claude Skill 主文档
├── references/                 # 详细参考资料
│   ├── conda_commands.md      # Conda 环境管理命令
│   ├── matplotlib_examples.md  # Matplotlib 最佳实践示例
│   ├── logging_best_practices.md # 日志最佳实践指南
│   ├── loguru_guide.md        # Loguru 日志库详细指南
│   ├── latex_symbols.md       # LaTeX 数学符号速查表
│   ├── output_standards.md    # 输出文件规范指南
│   └── type_annotations_guide.md # Python 类型注解科学计算指南
├── scripts/                    # 脚本目录（已清空，不再提供 bat 脚本）
├── output/                     # 输出目录（按规范生成）
│   ├── logs/                   # 日志文件
│   ├── figures/                # 图表文件
│   ├── models/                 # 模型文件
│   ├── data/                   # 数据输出
│   └── temp/                   # 临时文件
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 环境检查与管理

**⚠️ 重要：避免使用 conda activate，推荐使用直接命令**

```bash
# 列出所有环境
conda env list

# 查看环境信息
conda info

# ✅ 推荐：直接运行（无需激活）
conda run -n your_env_name python script.py

# ✅ 推荐：在指定环境中安装包
conda install -n your_env_name numpy matplotlib

# 仅当没有其他选择时才使用 pip
conda run -n your_env_name pip install special-package
```

🔴 **避免使用**：`conda activate your_env_name` - 这可能导致环境冲突和依赖问题

详细的环境管理命令和最佳实践请参考 `references/conda_commands.md`。

### 2. 输出文件管理

建立规范的输出管理体系：

```python
import os
from datetime import datetime

def get_output_path(subdir, name, ext):
    """生成标准输出文件路径"""
    output_dir = os.path.join('output', subdir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%y%m%d%H%M')
    filename = f"{name}.{ext}"
    filepath = os.path.join(output_dir, filename)

    return filepath

# 使用示例
params = {'lr': 0.001, 'batch_size': 32, 'epochs': 100}
param_str = f"lr{params['lr']}_bs{params['batch_size']}_e{params['epochs']}"
timestamp = datetime.now().strftime('%y%m%d%H%M')

# 保存模型
model_path = get_output_path('models', f"{timestamp}_{param_str}_weights", 'pth')
logger.info(f"[SAVE] 模型文件: {model_path}")
```

### 3. 日志配置（Loguru）

```python
from loguru import logger
import sys

# 移除默认输出
logger.remove()

# 配置控制台输出
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# 配置文件输出
logger.add(
    "output/logs/{time:YYMMDD}_app.log",
    rotation="10 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG"
)

# 使用
logger.info("程序开始运行")
logger.debug(f"当前 Python 版本: {sys.version}")
logger.error("发生错误")

# 异常自动包含堆栈
try:
    risky_operation()
except Exception:
    logger.exception("操作失败")
```

### 4. Matplotlib 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 参数
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# LaTeX 公式标签
ax.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=12)
ax.set_ylabel(r'$\mathbf{Berry\ Phase\ (\pi\ units)}$')

# 添加文本
ax.text(0.5, 0.5,
        r'$\gamma(t) = \frac{|\langle m|\partial_t H|n\rangle|}{|E_m - E_n|^2}$',
        ha='center', va='center', transform=ax.transAxes)

# 保存图表（遵循输出规范）
filepath = f"output/figures/{timestamp}_{param_str}_berry_phase.png"
plt.savefig(filepath, dpi=300, bbox_inches='tight')
logger.info(f"[SAVE] 图表: {filepath}")
```

## 📋 核心规范

### 1. 环境管理
- **原则**：始终避免使用 `conda activate`，直接使用 `conda run -n` 和 `conda install -n`
- **推荐**：优先使用 `conda run -n env` 命令进行环境隔离
- **一致性**：确保团队成员使用相同的 conda 环境
- **参考**：详细命令请参考 `references/conda_commands.md`

### 2. 输出文件管理 ⭐
遵循统一的命名规范和目录结构：
- **命名格式**：`{时间戳}_{参数组}_{描述}.{扩展名}`
- **时间戳**：`YYMMDDHHMM`（简化格式）
- **目录结构**：`output/{logs,figures,models,data,temp}/`
- **日志记录**：每个输出文件都要记录到日志

### 3. 可视化规范
- **LaTeX 公式**：使用 `r'$\latex'` 格式
- **文本处理**：使用英文标签避免编码问题
- **3D 图形**：使用 `text2D` 添加 2D 注释
- **矩阵显示**：使用 Unicode 字符对齐

### 4. 日志记录
- **默认系统**：使用 Loguru 作为日志系统
- **记录内容**：关键参数、计算耗时、输出文件路径
- **日志级别**：INFO（一般信息）、DEBUG（调试）、ERROR（错误）

### 5. 类型注解
- **明确类型**：使用 Type Annotations 提高代码可读性
- **数值精度**：明确区分 float32/float64、int32/int64
- **科学计算类型**：使用 NewType 创建物理量类型
- **参考指南**：详细规范请参考 `type_annotations_guide.md`

### 6. 数值误差分析
- **容限设置**：根据实际需求设置 rtol 和 atol
- **误差评估**：计算绝对误差、相对误差、RMSE 等
- **结果记录**：在日志中记录误差分析结果

## 🔧 实用工具

### Conda 环境管理速查

```bash
# 创建环境
conda create -n science_env python=3.9 numpy scipy matplotlib -y

# 查看环境列表
conda env list

# ✅ 推荐：直接运行（无需激活）
conda run -n science_env python script.py

# ✅ 推荐：在指定环境中安装包
conda install -n science_env pandas

# 导出环境
conda env export > environment.yml

# 从文件创建环境
conda env create -f environment.yml
```

🔴 **避免使用**：`conda activate science_env` - 可能导致环境冲突

### LaTeX 数学符号速查

| 类型 | 符号 | LaTeX |
|------|------|-------|
| 希腊字母 | α | `\alpha` |
| 希腊字母 | β | `\beta` |
| 希腊字母 | γ | `\gamma` |
| 运算符 | × | `\times` |
| 运算符 | ± | `\pm` |
| 关系 | ≤ | `\leq` |
| 关系 | ≥ | `\geq` |
| 集合 | ∈ | `\in` |
| 箭头 | → | `\rightarrow` |

## 📚 详细文档

| 文档 | 描述 | 用途 |
|------|------|------|
| [SKILL.md](SKILL.md) | Claude Skill 完整指南 | AI 使用说明 |
| [output_standards.md](references/output_standards.md) | 输出文件详细规范 | 文件管理标准 |
| [loguru_guide.md](references/loguru_guide.md) | Loguru 使用指南 | 日志系统配置 |
| [matplotlib_examples.md](references/matplotlib_examples.md) | 绘图示例代码 | 可视化参考 |
| [latex_symbols.md](references/latex_symbols.md) | LaTeX 符号大全 | 数学公式编写 |
| [conda_commands.md](references/conda_commands.md) | Conda 命令参考 | 环境管理 |
| [type_annotations_guide.md](references/type_annotations_guide.md) | 类型注解科学计算指南 | 代码类型规范 |

## ✅ 代码审查清单

使用前请确认：

- [ ] 使用 `conda env list` 确认了 conda 环境
- [ ] **✅ 避免使用 `conda activate`，使用 `conda run -n env_name`**
- [ ] 输出文件遵循 `output_standards.md` 的命名规范
- [ ] 日志中记录了所有输出文件的路径
- [ ] matplotlib 使用了正确的 LaTeX 公式格式
- [ ] 避免了复杂的 LaTeX 环境
- [ ] 使用了英文标签避免编码问题
- [ ] 3D 图中使用 `text2D` 而非 `text`
- [ ] 矩阵显示使用了多行字符串和等宽字体
- [ ] 设置了适当的日志级别
- [ ] 进行了数值误差分析（如适用）
- [ ] 仅在 Python 相关任务中使用此技能

## ❓ 常见问题

### Q: 必须使用 Loguru 吗？
A: 不是强制要求，但 Loguru 是推荐的默认选择，提供更简洁的 API。

### Q: 如何处理参数过多的情况？
A: 使用简化版参数名或只保留关键参数，详细参数在日志中说明。

### Q: 时间戳精度如何选择？
A: 一般到分钟即可，同一时间多次运行可添加分钟序列号。

### Q: 这与 PEP 8 的关系？
A: 本规范专注于科学计算的特殊需求，是 PEP 8 的补充。

### Q: 为什么不再推荐使用 conda activate？
A: 为了避免环境冲突和依赖问题，提高代码的可重现性和跨平台兼容性。详细原因请参考 `references/conda_commands.md`。

### Q: 为什么不再提供 bat 脚本？
A: 为了简化项目结构并提供更好的跨平台兼容性，现在推荐直接使用 conda 命令行工具。详细命令请参考 `references/conda_commands.md`。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/new-guideline`
3. 提交更改：`git commit -m "add: new guideline for xxx"`
4. 推送分支：`git push origin feature/new-guideline`
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Conda](https://docs.conda.io/) - 包管理和环境管理
- [Matplotlib](https://matplotlib.org/) - 数据可视化
- [Loguru](https://github.com/Delgan/loguru) - 日志记录
- [NumPy](https://numpy.org/) - 数值计算基础
- [SciPy](https://scipy.org/) - 科学计算库

---

⭐ 如果这个项目对你的科研工作有帮助，请给个 Star！
