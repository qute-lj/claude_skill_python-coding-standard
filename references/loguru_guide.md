# Loguru 日志指南

Loguru 是一个旨在让 Python 日志记录变得简单、愉快的库。它消除了传统的 logging 配置复杂性。

## 安装

```bash
# 使用 conda
conda install -c conda-forge loguru

# 或使用 pip
pip install loguru
```

## 快速开始

```python
from loguru import logger

# 直接使用，无需配置
logger.info("Hello, World!")
logger.debug("Debugging info")
logger.warning("Warning message")
logger.error("Error occurred")
logger.success("Success!")
logger.critical("Critical error")
```

## 基础配置

### 添加输出处理器

```python
import sys

# 移除默认处理器
logger.remove()

# 添加控制台输出（带颜色）
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO",
    colorize=True
)

# 添加文件输出
logger.add(
    "output/logs/app_{time:YYYY-MM-DD}.log",
    rotation="10 MB",  # 文件超过10MB自动轮转
    retention="30 days",  # 保留30天
    compression="zip",  # 压缩旧日志
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    encoding="utf-8"
)
```

### 高级配置选项

```python
# 带过滤的日志
logger.add(
    "special.log",
    filter=lambda record: "special" in record["message"],
    level="DEBUG"
)

# 异步写入（提高性能）
logger.add("async.log", enqueue=True)

# 根据级别不同保存到不同文件
logger.add("error.log", level="ERROR")
logger.add("debug.log", level="DEBUG")

# JSON 格式输出
logger.add(
    "json.json",
    format="{time} | {level} | {message} | {extra}",
    serialize=True
)
```

## 实用功能

### 1. 异常捕获（自动包含堆栈）

```python
try:
    1 / 0
except ZeroDivisionError:
    # 自动包含完整的堆栈信息
    logger.exception("Division by zero error")

# 或使用 catch 装饰器
@logger.catch
def dangerous_function(x):
    return 1 / x
```

### 2. 结构化日志

```python
# 使用 bind 添加上下文
task_logger = logger.bind(task_id=123)
task_logger.info("Task started")

# 或使用 extra
logger.info("User login", extra={"user_id": 123, "ip": "192.168.1.1"})

# 在格式中访问
logger.add(
    "app.log",
    format="{time} - {extra[user_id]} - {message}"
)
```

### 3. 性能测量

```python
# 使用 contextlib 上下文管理器
with logger.contextualize(task="processing"):
    logger.info("开始处理")
    # ... 处理代码
    logger.info("处理完成")

# 自动测量执行时间
@logger.catch
@logger.contextualize(task="important")
def process_data():
    time.sleep(1)
    logger.info("Data processed")
```

### 4. 日志轮转和压缩

```python
# 按大小轮转
logger.add("file.log", rotation="10 MB")

# 按时间轮转
logger.add("file.log", rotation="1 day")  # 每天
logger.add("file.log", rotation="1 week")  # 每周
logger.add("file.log", rotation="12:00")  # 每天12点

# 保留策略
logger.add("file.log", retention="10 days")  # 保留10天
logger.add("file.log", retention="10 files")  # 保留最近10个文件

# 压缩旧日志
logger.add("file.log", compression="gz")
logger.add("file.log", compression="zip")
```

## 科学计算专用模板

### 实验日志记录器

```python
from loguru import logger
import sys
import os
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, experiment_name: str, output_dir: str = "output/logs"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir

        # 创建目录
        os.makedirs(output_dir, exist_ok=True)

        # 配置日志
        self._setup_logger()

    def _setup_logger(self):
        """配置日志输出"""
        logger.remove()

        # 控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
            colorize=True
        )

        # 文件输出
        log_file = os.path.join(
            self.output_dir,
            f"{datetime.now().strftime('%y%m%d')}_{self.experiment_name}.log"
        )

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="20 MB",
            retention="7 days"
        )

    def log_parameters(self, params: Dict[str, Any]):
        """记录实验参数"""
        logger.info(f"=== 实验参数 [{self.experiment_name}] ===")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")

    def log_dataset_info(self, dataset_name: str, info: Dict[str, Any]):
        """记录数据集信息"""
        logger.info(f"=== 数据集: {dataset_name} ===")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """记录评估指标"""
        prefix = f"Step {step}" if step else "Metrics"
        logger.info(f"=== {prefix} ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.6f}")
            else:
                logger.info(f"  {metric}: {value}")

    def log_save(self, item_type: str, path: str):
        """记录保存的文件"""
        logger.info(f"[SAVE] {item_type}: {path}")

    def log_error(self, error_type: str, message: str, details: str = None):
        """记录错误"""
        logger.error(f"[{error_type}] {message}")
        if details:
            logger.debug(f"  详情: {details}")
```

### 使用示例

```python
# 创建实验记录器
exp_logger = ExperimentLogger("lr_experiment")

# 记录参数
params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam',
    'model': 'resnet50'
}
exp_logger.log_parameters(params)

# 记录数据集
dataset_info = {
    'train_samples': 50000,
    'test_samples': 10000,
    'features': '224x224x3',
    'classes': 1000
}
exp_logger.log_dataset_info("ImageNet", dataset_info)

# 训练循环
for epoch in range(params['epochs']):
    # ... 训练代码
    metrics = {
        'loss': 0.345,
        'accuracy': 0.892,
        'precision': 0.876,
        'recall': 0.903
    }
    exp_logger.log_metrics(metrics, epoch)

# 保存模型
model_path = "output/models/241201_lr0.001_weights.pth"
exp_logger.log_save("Model", model_path)
```

## 性能优化技巧

### 1. 异步日志

```python
# 使用 enqueue=True 避免阻塞主线程
logger.add("async.log", enqueue=True)
```

### 2. 延迟格式化

```python
# 使用 lambda 延迟格式化（只在需要时计算）
logger.debug(lambda: f"Complex computation: {expensive_function()}")
```

### 3. 日志级别过滤

```python
# 生产环境禁用 DEBUG
if os.getenv("ENV") == "production":
    logger.remove()
    logger.add("prod.log", level="INFO")
```

## 与传统 logging 比较

| 特性 | 传统 logging | Loguru |
|------|------------|--------|
| 配置复杂度 | 复杂 | 简单 |
| 异常追踪 | 需要手动 | 自动 |
| 日志轮转 | 需要额外处理器 | 内置 |
| 颜色支持 | 需要额外库 | 内置 |
| 结构化日志 | 复杂 | 简单 |
| 性能 | 好 | 很好 |
| 依赖 | 内置 | 需要安装 |

## 最佳实践

1. **始终指定输出格式**：自定义格式，包含时间和级别
2. **使用文件轮转**：避免日志文件过大
3. **合理使用级别**：DEBUG 用于调试，INFO 用于常规，ERROR 用于错误
4. **记录上下文**：使用 bind 或 extra 添加相关信息
5. **异步写入**：生产环境考虑使用 enqueue=True
6. **异常处理**：使用 exception() 记带堆栈的错误

## 常见问题

### Q: 如何同时输出到控制台和文件？

```python
logger.add(sys.stdout, level="INFO")
logger.add("file.log", level="DEBUG")
```

### Q: 如何只在特定条件下记录？

```python
logger.add("debug.log", filter=lambda record: record["extra"].get("debug"))
logger.debug("Debug info", extra={"debug": True})
```

### Q: 如何自定义日志级别？

```python
logger.add("custom.log", level=25)  # 介于 INFO(20) 和 WARNING(30) 之间
```

### Q: 如何记录 NumPy 数组？

```python
import numpy as np
arr = np.random.rand(1000)
logger.info(f"数组统计: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
```