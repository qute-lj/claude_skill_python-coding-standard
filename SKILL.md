---
name: python-coding-standard
description: This skill provides comprehensive Python coding standards and best practices for scientific computing, including conda environment management, matplotlib visualization best practices, logging standards, and numerical error analysis guidelines. **IMPORTANT: This skill ONLY triggers for Python-related tasks.** Use this skill when writing Python code that requires professional standards for scientific visualization, proper environment setup, robust logging, or numerical accuracy analysis. Do not use for other programming languages or general Python questions that don't require scientific computing standards.
---

# Python 代码书写规范

本技能提供科学计算 Python 代码的全面编码标准和最佳实践，包括环境管理、可视化规范、日志记录和数值误差分析。

## 何时使用此技能

**重要提示：此技能仅用于 Python 相关任务！**

当用户需要：
- 设置和管理 Python conda 环境
- 创建专业的科学可视化图表
- 实现规范的日志记录系统
- 分析和控制数值计算误差
- 编写符合科研标准的 Python 代码

**不适用于：**
- 其他编程语言（Java, C++, JavaScript 等）
- 不涉及科学计算的一般 Python 问题
- 简单的 Python 脚本（如 Hello World）
- Python 基础语法教学

## 核心使用指南

### 1. 环境管理最佳实践

#### 运行前环境检查

**原则**：始终在运行 Python 代码前检查并使用正确的 conda 环境

**推荐方法**：使用 conda 命令行工具

1. **快速环境检查**：
   ```bash
   conda env list
   conda info
   ```

2. **激活并运行 Python**：
   ```bash
   # 手动操作
   conda activate your_env_name
   python your_script.py
   ```

3. **直接运行（无需激活）**：
   ```bash
   conda run -n env_name python script.py
   ```

详细的环境管理命令请参考 `references/conda_commands.md`。

### 2. 输出文件管理规范

**原则**：所有输出文件必须遵循统一的命名规范和目录结构

参考 `references/output_standards.md` 建立完整的输出管理体系：

1. **文件命名格式**：`{时间戳}_{参数组}_{描述}.{扩展名}`
   - 时间戳：`YYMMDDHHMM`（简化格式）
   - 参数组：`lr0.001_bs32_e100`（关键参数）
   - 描述：`loss`、`acc`、`weights` 等

2. **目录结构**：
   ```
   output/
   ├── logs/      # 日志文件
   ├── figures/   # 图表文件
   ├── models/    # 模型文件
   ├── data/      # 数据输出
   └── temp/      # 临时文件
   ```

3. **日志记录**：每个输出文件都要在日志中记录
   ```python
   logger.info(f"[SAVE] 图表: {filepath}")
   logger.info(f"[SAVE] 模型: {model_path}")
   ```

完整的输出管理规范请参考 `references/output_standards.md`。

### 3. Matplotlib 可视化最佳实践

#### LaTeX 公式使用规范

**核心原则**：不是所有字符都需要 LaTeX 格式，只有适合用 LaTeX 格式的公式和适宜加粗等的字体才需要使用

**LaTeX 适用场景**：
- 数学公式和物理量：`r'$E = mc^2$'`
- 特殊符号：`r'$\alpha, \beta, \gamma$'`
- 需要加粗的变量名：`r'$\mathbf{Time\ (t)}$'`
- 希腊字母和数学符号

**普通文本适用场景**：
- 简单的英文单词：`'Time'`、`'Energy'`
- 一般描述性文本
- 数值标签：`'Sample 1'`、`'Group A'`

**英文标签规范**：所有 matplotlib 的 label、title、legend 等一律使用英文，避免中文编码问题

更多matplotlib示例请参考 `references/matplotlib_examples.md`。

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 参数
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

# ✅ 正确的使用方式：LaTeX + 普通文本结合
ax.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=12)  # 变量名用 LaTeX 加粗
ax.set_ylabel(r'$\mathbf{Berry\ Phase\ (\pi\ units)}$')  # 复合概念用 LaTeX
ax.set_title('Berry Phase Evolution')  # 标题用普通英文

# ✅ 物理量定义（必须用 LaTeX）
ax.text(0.5, 0.5,
        r'$\gamma(t) = \frac{|\langle m|\partial_t H|n\rangle|}{|E_m - E_n|^2}$',
        ha='center', va='center')

# ✅ 图例使用英文
ax.legend(['Ground State', 'Excited State'], loc='best')

# ❌ 避免的使用方式
# ax.set_xlabel('时间')  # 不要用中文
# ax.text(0.5, 0.5, r'$\text{Simple Text}$')  # 简单文本不需要 LaTeX
```

LaTeX数学符号速查请参考 `references/latex_symbols.md`。

#### 文本格式化规范

```python
# 一般文本使用 \text
ax.text(0.5, 0.8, r'$\text{Ground State}$')

# 重要标签使用 \mathbf
ax.set_xlabel(r'$\mathbf{Time\ (t)}$')
ax.set_ylabel(r'$\mathbf{Berry\ Phase}$')

# 混合文本和公式
ax.text(0.5, 0.5, r'$\text{Berry Phase: } \gamma_B = \pi$')
```

#### 3D 图中的 2D 注释

```python
# 使用 text2D 添加2D文本到3D图
ax = fig.add_subplot(111, projection='3d')
ax.text2D(0.02, 0.02,
          r'$H(k) = v_F(k_x \sigma_x + k_y \sigma_y)$',
          transform=ax.transAxes,
          fontsize=10,
          ha='left', va='bottom',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
```

### 4. Logging 最佳实践

#### 使用 Loguru（默认推荐）

```python
# 安装：conda install -c conda-forge loguru
from loguru import logger

# 配置输出（遵循 output_standards.md）
logger.remove()  # 移除默认输出
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "output/logs/{time:YYMMDD}_app.log",
    rotation="10 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG"
)

# 直接使用
logger.info("程序开始运行")
logger.debug(f"当前 Python 版本: {sys.version}")
logger.error("发生错误")

# 异常自动包含堆栈
try:
    risky_operation()
except Exception:
    logger.exception("操作失败")

# 记录输出文件
filepath = save_figure("results", params)
logger.info(f"[SAVE] 图表: {filepath}")
```

详细配置请参考 `references/loguru_guide.md` 和 `references/logging_best_practices.md`。

#### 科学计算 Logging 模板

```python
def log_computation_info(func_name, params, result, time_elapsed):
    """记录计算信息（Loguru 版本）

    Args:
        func_name: 函数名称
        params: 输入参数
        result: 计算结果
        time_elapsed: 耗时
    """
    logger.info(f"函数 {func_name} 执行完成")
    logger.debug(f"输入参数: {params}")
    logger.debug(f"计算结果形状: {np.array(result).shape}")
    logger.info(f"计算耗时: {time_elapsed:.4f} 秒")
```

### 5. 数值误差分析

#### 误差评估函数

```python
import numpy as np
from typing import Union, Tuple

def analyze_numerical_error(
    computed: Union[float, np.ndarray],
    reference: Union[float, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Tuple[dict, bool]:
    """分析数值计算误差

    Args:
        computed: 计算得到的值
        reference: 参考值（解析解或高精度数值解）
        rtol: 相对误差容限
        atol: 绝对误差容限

    Returns:
        error_info: 误差信息字典
        is_acceptable: 误差是否可接受
    """
    computed = np.array(computed)
    reference = np.array(reference)

    # 计算各类误差
    abs_error = np.abs(computed - reference)
    rel_error = abs_error / (np.abs(reference) + atol)

    # 统计信息
    error_info = {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error),
        'rmse': np.sqrt(np.mean((computed - reference)**2)),
        'mae': np.mean(abs_error)
    }

    # 判断是否在容限内
    is_acceptable = np.all(abs_error <= atol + rtol * np.abs(reference))

    return error_info, is_acceptable

def log_error_analysis(error_info, is_acceptable):
    """记录误差分析结果（Loguru 版本）"""
    logger.info("数值误差分析结果:")
    logger.info(f"  最大绝对误差: {error_info['max_abs_error']:.2e}")
    logger.info(f"  平均绝对误差: {error_info['mean_abs_error']:.2e}")
    logger.info(f"  最大相对误差: {error_info['max_rel_error']:.2e}")
    logger.info(f"  RMSE: {error_info['rmse']:.2e}")

    if is_acceptable:
        logger.success("✓ 误差在可接受范围内")  # Loguru 支持 success 级别
    else:
        logger.warning("⚠ 超出误差容限")
```

## 使用 bundled 资源

### 重要：必须阅读的 References 文件

**当执行特定任务时，Claude 必须主动读取以下 references 文件：**

- `conda_commands.md` - **必须阅读**：当涉及环境管理、conda 命令、运行 Python 代码时
- `output_standards.md` - **必须阅读**：当需要保存任何输出文件（图表、数据、模型、日志）时
- `matplotlib_examples.md` - **必须阅读**：当需要创建任何可视化图表时
- `logging_best_practices.md` - **必须阅读**：当需要设置日志系统或记录信息时
- `loguru_guide.md` - **必须阅读**：当使用 Loguru 日志库时（默认推荐）
- `latex_symbols.md` - **必须阅读**：当需要在 matplotlib 中使用 LaTeX 公式时

### 使用规则

1. **任务触发读取**：根据任务类型自动读取对应的 references 文件
2. **优先级**：references 中的规范优先于通用知识
3. **完整性**：必须严格遵循 references 中的标准和示例

**重要说明**：本技能不再提供 bat 脚本文件，所有环境管理操作请直接使用 conda 命令或参考 `references/conda_commands.md` 中的详细说明。

## Claude 使用工作流程

**当使用此技能时，Claude 必须按以下步骤执行：**

### 步骤 1：任务分析
确定任务类型，识别需要哪些 references 文件：
- 环境管理 → `conda_commands.md`
- 文件输出 → `output_standards.md`
- 图表制作 → `matplotlib_examples.md`
- 日志设置 → `logging_best_practices.md` + `loguru_guide.md`
- LaTeX 公式 → `latex_symbols.md`

### 步骤 2：读取 References
根据任务类型，**主动**读取相应的 references 文件，不得跳过。

### 步骤 3：遵循标准
严格按照 references 中的规范和示例执行任务，优先级高于通用知识。

### 步骤 4：检查清单
使用下面的代码审查清单确保所有标准都被遵循。

## 实施步骤

1. **运行前检查**：使用 `conda env list` 确认环境
2. **环境激活**：使用 `conda activate` 或 `conda run -n env`
3. **导入模块**：导入必要的标准库和第三方库
4. **设置日志**：参考 `logging_best_practices.md` 配置日志系统
5. **编写代码**：遵循 `matplotlib_examples.md` 和 `latex_symbols.md` 中的规范
6. **可视化**：使用 matplotlib 最佳实践创建图表，所有标签用英文
7. **记录结果**：通过日志记录所有重要信息

## 代码审查清单

- [ ] 使用 `conda env list` 确认了 conda 环境
- [ ] matplotlib 正确使用 LaTeX 格式（仅用于公式和特殊符号）
- [ ] 所有 label、title、legend 等都使用英文
- [ ] 避免了对简单文本使用 LaTeX 格式
- [ ] 避免了复杂的 LaTeX 环境，使用 Unicode 字符对齐矩阵
- [ ] 设置了适当的日志级别，参考了 `loguru_guide.md`
- [ ] **输出文件遵循 `output_standards.md` 的命名规范**
- [ ] **日志中记录了所有输出文件的路径**
- [ ] 在 3D 图中使用 text2D 而非 text
- [ ] 矩阵显示使用了多行字符串和等宽字体
- [ ] 公式使用定义式而非具体数值

## 注意事项

- **环境管理**：优先使用 `conda run -n env` 或 conda 命令行工具，参考 `references/conda_commands.md`
- **环境一致性**：确保团队成员使用相同的 conda 环境
- **matplotlib 标签**：所有标签、标题、图例必须使用英文
- **LaTeX 使用**：只对数学公式、希腊字母、需要加粗的变量名使用 LaTeX，简单文本使用普通字符串
- **日志管理**：使用前参考 `loguru_guide.md` 配置日志系统
- **输出管理**：所有输出文件到 `output/` 目录，遵循统一命名规范
- **代码复用**：从 `references/` 复制所需代码片段