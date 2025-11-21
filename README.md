# Python 科学计算代码书写规范

[![Skill Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/python-coding-standard)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 专为科学计算设计的 Python 代码规范和最佳实践指南，包含环境管理、可视化规范、日志记录等基础设施。

## 🎯 适用场景

当你需要：
- ✅ 设置和管理 Python conda 环境
- ✅ 创建专业的科学可视化图表
- ✅ 实现规范的日志记录系统（默认使用 Loguru）
- ✅ 统一输出文件命名规范
- ✅ 正确使用 LaTeX 数学公式
- ✅ 编写符合科研标准的 Python 代码

## 📁 项目结构

```
python-coding-standard/
├── SKILL.md                    # Claude Skill 主文档
├── scripts/
│   ├── check_env.bat          # Windows 环境检查工具
│   └── run_with_env.bat       # 指定环境运行脚本
├── references/
│   ├── conda_commands.md      # Conda 环境管理命令大全
│   ├── matplotlib_examples.md  # Matplotlib 最佳实践示例
│   ├── logging_best_practices.md # 日志内容选择指南
│   ├── loguru_guide.md        # Loguru 日志库详细指南（默认）
│   ├── latex_symbols.md       # LaTeX 数学符号速查表
│   └── output_standards.md    # 输出文件规范指南
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 环境检查（Windows）

```batch
# 检查所有 conda 环境
scripts\check_env.bat
```

### 2. 运行 Python 脚本

```batch
# 使用指定环境运行
scripts\run_with_env.bat my_env my_script.py

# 或直接使用命令
conda run -n my_env python my_script.py
```

### 3. 日志配置（Loguru）

```python
from loguru import logger
import sys

# 简单配置
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("output/logs/{time:YYMMDD}_app.log", level="DEBUG")

# 使用
logger.info("程序开始")
```

### 4. Matplotlib 示例

```python
import matplotlib.pyplot as plt
import numpy as np

# 保存图表（遵循输出规范）
from datetime import datetime
import os

def save_figure(description, params=None):
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    if params:
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        filename = f"{timestamp}_{param_str}_{description}.png"
    else:
        filename = f"{timestamp}_{description}.png"

    os.makedirs('output/figures', exist_ok=True)
    filepath = os.path.join('output/figures', filename)
    plt.savefig(filepath, dpi=300)
    print(f"图表已保存: {filepath}")
    return filepath

# 使用
params = {'lr': 0.001, 'bs': 32}
save_figure('loss_curve', params)
```

## 📋 核心规范

### 输出文件管理规范 ⭐
遵循 `output_standards.md` 建立的完整输出管理体系：
- **文件命名格式**：`{时间戳}_{参数组}_{描述}.{扩展名}`
- **时间戳**：`YYMMDDHHMM`（简化格式）
- **参数组**：`lr0.001_bs32_e100`（关键参数）
- **目录结构**：`output/{logs,figures,models,data,temp}/`
- **日志记录**：每个输出文件都要在日志中记录
- **示例**：`241121_lr0.001_bs32_loss.png`

### 日志记录要点
- 使用 Loguru 作为默认日志系统
- 记录关键参数（特别是数值误差相关）
- 记录所有输出文件路径
- 使用适当的日志级别

### Matplotlib 规范
- 使用 LaTeX 公式：`r'$\formula$'`
- 矩阵显示使用 Unicode 字符对齐
- 3D 图使用 `text2D` 添加注释
- 使用英文标签避免编码问题

## 🔧 实用工具

### Conda 环境管理

```bash
# 列出所有环境
conda env list

# 创建新环境
conda create -n science_env python=3.9 numpy scipy matplotlib -y

# 直接运行（无需激活）
conda run -n science_env python script.py
```

### LaTeX 数学符号

常用符号速查：
- 希腊字母：`\alpha, \beta, \gamma`
- 运算符：`\times, \div, \pm`
- 集合：`\in, \subset, \cup, \cap`
- 箭头：`\rightarrow, \Leftarrow`

矩阵对齐（推荐）：
```python
matrix_text = 'H(t) = ⎡ αt   V ⎤\n' + \
              '       ⎣ V  -αt ⎦'
plt.text(0.5, 0.5, matrix_text, fontfamily='monospace')
```

## 📚 文档说明

| 文档 | 描述 | 适用场景 |
|------|------|----------|
| [SKILL.md](SKILL.md) | Claude Skill 主文档 | Claude AI 使用指南 |
| [conda_commands.md](references/conda_commands.md) | Conda 命令大全 | 环境管理参考 |
| [matplotlib_examples.md](references/matplotlib_examples.md) | 绘图示例代码 | 科学可视化参考 |
| [loguru_guide.md](references/loguru_guide.md) | Loguru 使用指南 | 日志系统配置 |
| [output_standards.md](references/output_standards.md) | 输出规范指南 | 文件命名标准 |

## 🏆 代码审查清单

使用前请确保：

- [ ] 使用 `check_env.bat` 确认了 conda 环境
- [ ] 输出文件遵循 `output_standards.md` 的命名规范
- [ ] 日志中记录了所有输出文件的路径
- [ ] matplotlib 使用了正确的 LaTeX 公式格式
- [ ] 矩阵显示使用了 Unicode 字符对齐
- [ ] 使用了英文标签避免编码问题
- [ ] 3D 图中使用 `text2D` 而非 `text`

## ❓ 常见问题

### Q: 如何在不同操作系统上使用？
A:
- Windows：使用提供的 `.bat` 脚本
- Linux/Mac：使用参考文档中的 bash 命令

### Q: 必须使用 Loguru 吗？
A: 不是，但 Loguru 是推荐的默认选择。传统 logging 仍然支持。

### Q: 如何贡献？
A: 欢迎 Issue 和 Pull Request！

### Q: 这与 PEP 8 的关系？
A: 本规范专注于科学计算的特定需求，是 PEP 8 的补充而非替代。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Conda](https://docs.conda.io/) - 环境管理
- [Matplotlib](https://matplotlib.org/) - 数据可视化
- [Loguru](https://github.com/Delgan/loguru) - 日志记录
- [NumPy](https://numpy.org/) - 数值计算

## 📞 联系方式

- 项目主页：[https://github.com/yourusername/python-coding-standard](https://github.com/yourusername/python-coding-standard)
- Issues：[https://github.com/yourusername/python-coding-standard/issues](https://github.com/yourusername/python-coding-standard/issues)

---

⭐ 如果这个项目对你有帮助，请给个 Star！