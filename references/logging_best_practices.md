# Python Logging 最佳实践

## 重要：使用前请确认要记录的内容

在为您的代码配置 logging 之前，**请告诉我**您希望记录以下哪些内容：

1. **基本信息**
   - [ ] 程序启动/结束时间
   - [ ] 函数执行流程
   - [ ] 数据加载状态

2. **数据信息**
   - [ ] 数据集大小（样本数、特征数）
   - [ ] 数据类型和内存占用
   - [ ] 缺失值统计

3. **计算过程**
   - [ ] 关键算法的中间结果
   - [ ] 迭代过程的收敛情况
   - [ ] 数值计算的误差估计

4. **性能指标**
   - [ ] 函数执行时间
   - [ ] 内存使用峰值
   - [ ] CPU/GPU 使用率

5. **错误和警告**
   - [ ] 异常捕获和处理
   - [ ] 数值不稳定警告
   - [ ] 性能瓶颈提示

6. **结果输出**
   - [ ] 模型评估指标
   - [ ] 预测结果统计
   - [ ] 可视化图表保存路径（遵循 `output_standards.md`）

7. **数值误差相关参数**（如需关注数值稳定性）
   - [ ] 矩阵条件数
   - [ ] 数值收敛容差
   - [ ] 舍入误差估计
   - [ ] 算法稳定性指标

**📌 输出文件规范**：所有输出文件的命名和路径请参考 `output_standards.md`，确保日志中记录的文件路径遵循统一规范。

请选择您关心的内容，我会据此定制 logging 配置。

---

## 基础配置模板

```python
import logging
import sys
from datetime import datetime

def setup_logger(name, level=logging.INFO, log_file=None):
    """设置标准的 logger 配置

    Args:
        name: logger 名称
        level: 日志级别
        log_file: 可选的日志文件路径
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 文件 handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(file_handler)

    # 设置控制台格式
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# 使用示例
logger = setup_logger('my_project', log_file='run.log')
logger.info("程序开始运行")
```

## 日志级别指南

```python
# DEBUG: 详细的调试信息
logger.debug(f"变量 x 的值: {x}")
logger.debug(f"函数 {func_name} 被调用，参数: {params}")

# INFO: 一般信息，程序正常运行
logger.info("数据加载完成")
logger.info(f"处理了 {n_samples} 个样本")

# WARNING: 警告，程序可以继续运行
logger.warning("检测到缺失值，已使用均值填充")
logger.warning("内存使用率较高")

# ERROR: 错误，程序出现异常但可以恢复
logger.error(f"无法加载文件 {filename}，使用默认值")
logger.error(f"数值计算溢出: {message}")

# CRITICAL: 严重错误，程序无法继续
logger.critical("内存不足，程序终止")
logger.critical("数据库连接失败")
```

## 科学计算专用日志模板

```python
def log_computation_step(logger, step_name, data_info, time_elapsed=None):
    """记录计算步骤

    Args:
        logger: logger 实例
        step_name: 步骤名称
        data_info: 数据信息字典
        time_elapsed: 耗时（秒）
    """
    logger.info(f"开始执行: {step_name}")

    if 'shape' in data_info:
        logger.debug(f"数据形状: {data_info['shape']}")
    if 'dtype' in data_info:
        logger.debug(f"数据类型: {data_info['dtype']}")
    if 'memory_usage' in data_info:
        logger.debug(f"内存使用: {data_info['memory_usage']:.2f} MB")

    if time_elapsed:
        logger.info(f"{step_name} 完成，耗时: {time_elapsed:.3f} 秒")

# 使用示例
data_info = {
    'shape': (1000, 100),
    'dtype': 'float64',
    'memory_usage': 0.76
}
log_computation_step(logger, "数据预处理", data_info, time_elapsed=0.235)
```

## 完整的实验日志模板

```python
class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, experiment_name, log_dir='logs'):
        self.experiment_name = experiment_name
        self.log_dir = log_dir

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 设置文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

        # 初始化 logger
        self.logger = setup_logger(experiment_name, log_file=log_file)

        # 记录实验开始
        self.logger.info(f"实验 '{experiment_name}' 开始")
        self.logger.info(f"日志文件: {log_file}")

    def log_parameters(self, params):
        """记录实验参数"""
        self.logger.info("实验参数:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_dataset_info(self, dataset_name, dataset_info):
        """记录数据集信息"""
        self.logger.info(f"数据集 '{dataset_name}' 信息:")
        self.logger.info(f"  样本数: {dataset_info.get('n_samples', 'N/A')}")
        self.logger.info(f"  特征数: {dataset_info.get('n_features', 'N/A')}")
        self.logger.info(f"  缺失值: {dataset_info.get('missing_values', 'N/A')}")

    def log_model_info(self, model_name, model_info):
        """记录模型信息"""
        self.logger.info(f"模型 '{model_name}' 信息:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics):
        """记录评估指标"""
        self.logger.info("评估指标:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")

    def log_error(self, error_type, error_msg, details=None):
        """记录错误"""
        self.logger.error(f"[{error_type}] {error_msg}")
        if details:
            self.logger.debug(f"详细信息: {details}")

# 使用示例
exp_logger = ExperimentLogger("ml_classification")

# 记录参数
params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'model': 'ResNet50'
}
exp_logger.log_parameters(params)

# 记录结果
metrics = {
    'accuracy': 0.9542,
    'precision': 0.9376,
    'recall': 0.9658,
    'f1_score': 0.9515
}
exp_logger.log_metrics(metrics)
```

## 性能监控日志

```python
import time
import psutil
import tracemalloc

def log_performance_metrics(logger, func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        # 开始监控
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # 执行函数
        try:
            result = func(*args, **kwargs)
            status = "成功"
        except Exception as e:
            result = None
            status = f"失败: {str(e)}"
            logger.error(f"函数 {func.__name__} 执行{status}")
            raise
        finally:
            # 结束监控
            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 记录性能指标
            logger.info(f"函数 {func.__name__} 执行{status}")
            logger.debug(f"  执行时间: {end_time - start_time:.3f} 秒")
            logger.debug(f"  内存变化: {end_mem - start_mem:.2f} MB")
            logger.debug(f"  峰值内存: {peak / 1024 / 1024:.2f} MB")

        return result
    return wrapper

# 使用示例
@log_performance_metrics(logger)
def train_model(data, model):
    # 训练代码
    time.sleep(2)  # 模拟训练时间
    return model.fit(data)
```

## 结构化日志（JSON 格式）

```python
import json
from datetime import datetime

class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, log_file='structured.log'):
        self.log_file = log_file

    def log(self, level, message, **kwargs):
        """记录结构化日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_training_step(self, epoch, loss, accuracy, learning_rate):
        """记录训练步骤"""
        self.log('INFO', 'Training step',
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                learning_rate=learning_rate)

    def log_prediction(self, model_name, input_id, prediction, confidence):
        """记录预测结果"""
        self.log('INFO', 'Prediction',
                model=model_name,
                input_id=input_id,
                prediction=prediction,
                confidence=confidence)

# 使用示例
structured_logger = StructuredLogger()
structured_logger.log_training_step(epoch=10, loss=0.324, accuracy=0.891, learning_rate=0.001)
```

## 日志分析命令

### 查看 INFO 级别日志
```bash
grep "INFO" your_log_file.log
```

### 查看错误日志
```bash
grep -E "(ERROR|CRITICAL)" your_log_file.log
```

### 统计错误类型
```bash
grep "ERROR" your_log_file.log | awk '{print $2}' | sort | uniq -c
```

### 实时监控日志
```bash
tail -f your_log_file.log
```

### 提取特定时间段的日志
```bash
grep "2024-01-15" your_log_file.log
```

## 最佳实践建议

1. **使用合适的日志级别**
   - DEBUG: 调试信息，生产环境通常关闭
   - INFO: 重要流程步骤
   - WARNING: 潜在问题
   - ERROR: 需要处理的错误
   - CRITICAL: 严重错误

2. **日志信息要清晰**
   - 包含上下文信息
   - 使用结构化格式
   - 避免敏感信息

3. **性能考虑**
   - 避免在循环中记录过多 DEBUG 日志
   - 使用异步日志处理（对于高频日志）
   - 定期清理旧日志文件

4. **文件管理**
   - 使用日志轮转
   - 按日期或大小分割日志
   - 保留关键日志备份

## 快速开始模板

### 默认使用 Loguru（推荐）

```python
# 安装：conda install -c conda-forge loguru
# 或：pip install loguru

from loguru import logger
import sys

# 移除默认处理器
logger.remove()

# 添加控制台输出（带颜色）
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 添加文件输出
logger.add(
    "output/logs/app_{time:YYYY-MM-DD}.log",
    rotation="10 MB",  # 文件超过10MB自动轮转
    retention="30 days",  # 保留30天
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# 使用（更简洁）
logger.info("应用启动")
logger.debug("调试信息")
logger.warning("警告信息")
logger.error("错误信息")

# 支持结构化日志
logger.info("用户登录", extra={"user_id": 123, "ip": "192.168.1.1"})

# 异常捕获（自动包含堆栈信息）
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("除零错误")
```