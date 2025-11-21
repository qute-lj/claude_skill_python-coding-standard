# 输出文件规范指南

## 核心原则

所有输出文件遵循统一命名规范，确保：
- 可追溯性：通过文件名了解生成条件
- 可排序性：时间戳保证按时间排序
- 可读性：关键参数一目了然
- 简洁性：命名尽可能简短

## 文件命名规范

### 标准格式
```
{时间戳}_{参数组}_{描述}.{扩展名}
```

### 1. 时间戳（简化格式）
- 格式：`YYMMDDHHMM`
- 示例：
  - `24112101` = 2024年11月21日 01分
  - `2411211430` = 2024年11月21日 14:30
  - `2412` = 2024年12月（仅月份）

### 2. 参数组（用下划线连接）
格式：`参数名1值1_参数名2值2`

#### 机器学习常见参数
- `lr0.001` - 学习率 0.001
- `bs32` - 批次大小 32
- `e100` - epochs 100
- `do0.5` - dropout 0.5
- `reg0.01` - 正则化系数 0.01

#### 数值计算常见参数
- `dt0.01` - 时间步长 0.01
- `tol1e-6` - 容差 1e-6
- `maxit1000` - 最大迭代次数 1000
- `condt1e10` - 条件数阈值
- `prec1e-12` - 数值精度

#### 数据处理参数
- `feat100` - 特征数 100
- `smp10000` - 样本数 10000
- `ratio0.8` - 比例 0.8

#### 完整参数组示例
```
lr0.001_bs32_e100          # 基础机器学习
dt0.01_tol1e-6_maxit1000   # 数值求解
feat100_smp10k_ratio0.8    # 数据处理
```

### 3. 描述部分
使用简短的英文或拼音：
- `loss` - 损失曲线
- `acc` - 准确率
- `weights` - 模型权重
- `pred` - 预测结果
- `confusion` - 混淆矩阵
- `heatmap` - 热力图
- `scatter` - 散点图
- `fit` - 拟合结果
- `conv` - 收敛过程

## 输出目录结构

```
项目根目录/
├── output/                 # 所有输出文件
│   ├── logs/              # 日志文件
│   │   ├── run_24112101_lr0.001.log
│   │   └── error_241121.log
│   ├── figures/           # 图表文件
│   │   ├── loss_241121_lr0.001.png
│   │   ├── acc_241121_lr0.001.png
│   │   └── confusion_241121_e100.png
│   ├── models/            # 模型文件
│   │   ├── weights_241121_lr0.001_e100.pth
│   │   └── model_241121_lr0.001.pkl
│   ├── data/              # 数据输出
│   │   ├── pred_241121_lr0.001.csv
│   │   └── results_241121.json
│   └── temp/              # 临时文件
│       └── debug_24112101.txt
```

## 图表标题规范

### 标题格式
图表标题应该与文件名保持一致，但不包含时间戳：
- 格式：`{参数组}_{描述}`
- 示例：`lr0.001_bs32_e100_loss_curve`

### 实现方式
```python
# 同时生成文件路径和标题
filepath, title_text = save_figure('loss_curve', params)

# 设置标题
ax.set_title(title_text, fontsize=13)
```

## 具体示例

### 1. 机器学习项目
```
# 训练日志
logs/train_241121_lr0.001_bs32_e100.log

# 损失曲线图（文件名）
figures/241121_lr0.001_bs32_e100_loss_curve.png
# 对应标题
lr0.001_bs32_e100_loss_curve

# 最终模型
models/241121_lr0.001_bs32_e100_weights.pth

# 预测结果
data/241121_lr0.001_bs32_e100_predictions.csv
```

### 2. 数值计算项目
```
# 计算日志
logs/solve_241121_dt0.01_tol1e-6.log

# 收敛图
figures/conv_241121_dt0.01_tol1e-6.png

# 解数据
data/solution_241121_dt0.01_tol1e-6.npy

# 误差分析
figures/error_241121_dt0.01_tol1e-6.png
```

### 3. 数据分析项目
```
# 分析日志
logs/analysis_241121_feat100.log

# 相关性热图
figures/corr_241121_feat100.png

# 处理后的数据
data/processed_241121_feat100.csv

# 统计报告
data/stats_241121_feat100.json
```

## 特殊情况处理

### 1. 参数过多时
使用简化版或分组：
```
# 完整版（参数较少）
241121_lr0.001_bs32_e100_do0.5_reg0.01_weights.pth

# 简化版（提取关键参数）
241121_lr0.001_e100_weights.pth

# 参数说明在日志中
```

### 2. 同一时间多次运行
添加序列号：
```
24112101_lr0.001_loss.png     # 第一次
24112102_lr0.001_loss.png     # 第二次（2分钟后）
24112103_lr0.001_loss.png     # 第三次（3分钟后）
```

### 3. 实验对比
保持参数一致，改变描述：
```
241121_lr0.001_loss.png       # 损失
241121_lr0.001_acc.png        # 准确率
241121_lr0.001_confusion.png  # 混淆矩阵
```

## 日志中记录输出文件

### Python 代码示例
```python
import os
from datetime import datetime

def get_output_path(subdir, name, ext):
    """生成输出文件路径"""
    # 确保目录存在
    output_dir = os.path.join('output', subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    filename = f"{name}.{ext}"
    filepath = os.path.join(output_dir, filename)

    return filepath

# 使用示例
import logging
logger = logging.getLogger(__name__)

# 记录参数
params = {
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# 生成文件名
param_str = f"lr{params['lr']}_bs{params['batch_size']}_e{params['epochs']}"
timestamp = datetime.now().strftime('%y%m%d%H%M')

# 保存模型
model_path = get_output_path('models', f"{timestamp}_{param_str}_weights", 'pth')
torch.save(model.state_dict(), model_path)
logger.info(f"模型已保存: {model_path}")

# 保存图表
plot_path = get_output_path('figures', f"{timestamp}_{param_str}_loss", 'png')
plt.savefig(plot_path)
logger.info(f"图表已保存: {plot_path}")
```

## 日志格式建议

在日志中统一格式记录文件输出：
```python
logger.info(f"[SAVE] 模型文件: {model_path}")
logger.info(f"[SAVE] 图表文件: {plot_path}")
logger.info(f"[SAVE] 数据文件: {data_path}")
logger.info(f"[SAVE] 配置文件: {config_path}")
```

## 最佳实践

1. **一致性**：整个项目使用相同的命名规范
2. **可读性**：避免过度简化，关键信息要清晰
3. **记录性**：每个输出文件都要在日志中记录
4. **清理**：定期清理 `output/temp/` 临时文件
5. **备份**：重要结果备份到其他位置

## 代码模板

```python
# 在项目开始时设置
import os
import logging
from datetime import datetime

class OutputManager:
    """输出文件管理器"""

    def __init__(self, project_name):
        self.project_name = project_name
        self.output_root = 'output'
        self.logger = logging.getLogger(project_name)

        # 创建目录
        for subdir in ['logs', 'figures', 'models', 'data', 'temp']:
            os.makedirs(os.path.join(self.output_root, subdir), exist_ok=True)

    def get_filename(self, subdir, params, description, ext):
        """生成标准文件名"""
        # 时间戳
        timestamp = datetime.now().strftime('%y%m%d%H%M')

        # 参数组
        if isinstance(params, dict):
            param_parts = []
            for key, value in sorted(params.items()):
                # 简化参数名
                key_map = {
                    'learning_rate': 'lr',
                    'batch_size': 'bs',
                    'epochs': 'e',
                    'dropout': 'do',
                    'tolerance': 'tol',
                    'max_iter': 'maxit'
                }
                short_key = key_map.get(key, key)[:4]
                param_parts.append(f"{short_key}{value}")
            param_str = '_'.join(param_parts)
        else:
            param_str = str(params)

        # 组合文件名
        filename = f"{timestamp}_{param_str}_{description}.{ext}"

        # 完整路径
        filepath = os.path.join(self.output_root, subdir, filename)

        return filepath

    def save_and_log(self, obj, subdir, params, description, ext, save_func):
        """保存对象并记录日志"""
        filepath = self.get_filename(subdir, params, description, ext)
        save_func(obj, filepath)
        self.logger.info(f"[SAVE] {description}: {filepath}")
        return filepath

# 使用示例
output = OutputManager('my_project')

# 保存模型
output.save_and_log(
    model,
    'models',
    {'lr': 0.001, 'batch_size': 32, 'epochs': 100},
    'weights',
    'pth',
    lambda obj, path: torch.save(obj.state_dict(), path)
)
```

## 注意事项

1. **时间戳精度**：根据需要选择，一般到分钟即可
2. **参数顺序**：保持参数名按字母或重要性排序
3. **扩展名**：使用标准扩展名（.png, .csv, .pth, .json）
4. **特殊字符**：文件名中避免使用空格和特殊字符
5. **长度限制**：某些系统有文件名长度限制，注意不要过长