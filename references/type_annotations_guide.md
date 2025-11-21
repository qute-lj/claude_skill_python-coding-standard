# Python Type Annotations 科学计算指南

## 📌 概述

Type Annotations（类型注解）是 Python 3.5+ 引入的重要特性，允许开发者显式指定变量、函数参数和返回值的类型。在科学计算中，合理的类型注解可以：

- 提高代码可读性和可维护性
- 利用 IDE 智能提示和静态检查
- 减少类型相关的运行时错误
- 方便团队协作和代码审查

## 🎯 为什么科学计算需要 Type Annotations

1. **数值精度控制**：明确区分 `float32`、`float64`、`int32`、`int64`
2. **数组维度**：使用 `typing.NewType` 区分标量、向量、矩阵
3. **复杂对象**：明确数据集、模型参数、结果类型
4. **函数签名**：清晰表达数学函数的输入输出关系

## 🔧 基础语法

### 1. 变量注解

```python
from typing import Union, Optional
import numpy as np

# 基本类型
learning_rate: float = 0.001
batch_size: int = 32
use_gpu: bool = True

# 可选类型
model_name: Optional[str] = None
validation_split: Union[float, None] = None

# NumPy 数组类型
weights: np.ndarray = np.random.randn(784, 10)
input_data: np.ndarray = np.zeros((100, 28, 28, 3))

# 指定 dtype
weights_float32: np.ndarray = np.random.randn(784, 10).astype(np.float32)
labels_int64: np.ndarray = np.random.randint(0, 10, size=(1000,), dtype=np.int64)
```

### 2. 函数注解

```python
from typing import Tuple, List, Dict, Any, Union, Callable

# 简单函数
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)

# 复杂返回类型
def train_test_split(
    data: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分割数据集"""
    # 实现代码...
    return X_train, X_test, y_train, y_test

# 可变参数
def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    **kwargs: Any
) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """训练模型并返回历史记录"""
    history = {}
    # 训练过程...
    return history

# 回调函数
def optimization_callback(
    epoch: int,
    metrics: Dict[str, float],
    model_params: np.ndarray
) -> bool:
    """优化回调，返回是否提前停止"""
    return metrics['loss'] < 1e-6
```

## 📊 科学计算中的高级类型定义

### 1. 使用 NewType 创建语义化类型

```python
from typing import NewType
import numpy as np

# 创建语义化类型
TimeSeries = NewType('TimeSeries', np.ndarray)
FrequencySeries = NewType('FrequencySeries', np.ndarray)
Scalar = NewType('Scalar', float)
Vector3D = NewType('Vector3D', np.ndarray)
Matrix3x3 = NewType('Matrix3x3', np.ndarray)
Probability = NewType('Probability', float)

# 使用示例
def autocorrelation(signal: TimeSeries) -> TimeSeries:
    """计算自相关函数"""
    # 实现自相关计算
    return TimeSeries(result)

def vector_magnitude(vector: Vector3D) -> Scalar:
    """计算3D向量模长"""
    return Scalar(np.linalg.norm(vector))
```

### 2. 使用 TypeVar 创建泛型

```python
from typing import TypeVar, Generic, Protocol
import numpy as np

T = TypeVar('T', np.ndarray, float, int)

class DataProcessor(Generic[T]):
    """通用数据处理器"""

    def __init__(self, data: T):
        self.data = T(data)

    def apply_filter(self, filter_func: Callable[[T], T]) -> T:
        """应用滤波器"""
        return T(filter_func(self.data))

# 协议定义
class Model(Protocol):
    """模型协议"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ...
```

### 3. 使用 TypedDict 定义结构化数据

```python
from typing import TypedDict, List, Optional, Union

class ModelConfig(TypedDict):
    """模型配置字典"""
    model_type: str
    input_dim: int
    hidden_dims: List[int]
    activation: str
    dropout_rate: Optional[float]
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int

class ExperimentResult(TypedDict):
    """实验结果字典"""
    model_name: str
    train_accuracy: float
    test_accuracy: float
    training_time: float
    final_loss: float
    best_epoch: int
    hyperparameters: ModelConfig

# 使用示例
config: ModelConfig = {
    "model_type": "MLP",
    "input_dim": 784,
    "hidden_dims": [256, 128, 64],
    "activation": "relu",
    "dropout_rate": 0.5,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}
```

## 🔬 科学计算特定类型

### 1. 物理量类型

```python
from typing import NewType, Union
import numpy as np

# 基础物理量
Time = NewType('Time', float)           # 时间 (s)
Length = NewType('Length', float)       # 长度 (m)
Mass = NewType('Mass', float)           # 质量 (kg)
Energy = NewType('Energy', float)       # 能量 (J)
Temperature = NewType('Temperature', float)  # 温度 (K)

# 导出物理量
Velocity = NewType('Velocity', float)   # 速度 (m/s)
Acceleration = NewType('Acceleration', float)  # 加速度 (m/s²)
Force = NewType('Force', float)         # 力 (N)
Power = NewType('Power', float)         # 功率 (W)

# 数组形式
TimeSeries = NewType('TimeSeries', np.ndarray)
Spectrum = NewType('Spectrum', np.ndarray)
Wavefunction = NewType('Wavefunction', np.ndarray)
Hamiltonian = NewType('Hamiltonian', np.ndarray)

# 使用示例
def kinetic_energy(mass: Mass, velocity: Velocity) -> Energy:
    """计算动能"""
    return Energy(0.5 * float(mass) * float(velocity) ** 2)

def fourier_transform(signal: TimeSeries) -> Spectrum:
    """傅里叶变换"""
    return Spectrum(np.fft.fft(signal))
```

### 2. 统计学类型

```python
from typing import NamedTuple, Union
import numpy as np

class Statistics(NamedTuple):
    """统计结果"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float

class ConfidenceInterval(NamedTuple):
    """置信区间"""
    lower: float
    upper: float
    confidence_level: float

# 使用示例
def calculate_statistics(data: np.ndarray) -> Statistics:
    """计算描述性统计"""
    return Statistics(
        mean=np.mean(data),
        std=np.std(data),
        min=np.min(data),
        max=np.max(data),
        median=np.median(data),
        q25=np.percentile(data, 25),
        q75=np.percentile(data, 75)
    )

def bootstrap_mean(
    data: np.ndarray,
    n_bootstrap: int = 1000
) -> ConfidenceInterval:
    """自助法计算均值置信区间"""
    # 实现自助法采样
    return ConfidenceInterval(
        lower=0.0,
        upper=1.0,
        confidence_level=0.95
    )
```

## 🛠️ 实用工具和技巧

### 1. 使用 dataclasses

```python
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class QuantumState:
    """量子态表示"""
    amplitudes: np.ndarray
    basis_labels: List[str]
    energy: Optional[float] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        # 验证归一化
        norm = np.linalg.norm(self.amplitudes)
        if not np.isclose(norm, 1.0):
            self.amplitudes = self.amplitudes / norm

    def probability(self, index: int) -> float:
        """计算特定基的概率"""
        return float(np.abs(self.amplitudes[index]) ** 2)

@dataclass
class SimulationParameters:
    """仿真参数"""
    time_step: float
    total_time: float
    num_steps: int

    @property
    def dt(self) -> float:
        return self.time_step

    @property
    def T(self) -> float:
        return self.total_time

    def __post_init__(self):
        # 自动计算步数
        if self.num_steps == 0:
            self.num_steps = int(self.total_time // self.time_step)
```

### 2. 使用 Protocol 定义接口

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Optimizer(Protocol):
    """优化器接口"""

    learning_rate: float

    def step(self, gradient: np.ndarray) -> np.ndarray:
        """执行一步优化"""
        ...

    def reset(self) -> None:
        """重置优化器状态"""
        ...

@runtime_checkable
class LossFunction(Protocol):
    """损失函数接口"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算损失值"""
        ...

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算梯度"""
        ...

# 使用示例
def train_model(
    model: Any,
    optimizer: Optimizer,
    loss_fn: LossFunction,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int
) -> List[float]:
    """训练模型的通用函数"""
    losses = []
    for epoch in range(epochs):
        # 前向传播
        y_pred = model(X)

        # 计算损失
        loss = loss_fn(y, y_pred)
        losses.append(loss)

        # 反向传播
        gradient = loss_fn.gradient(y, y_pred)
        params = optimizer.step(gradient)

    return losses
```

## 🔍 类型检查工具

### 1. 使用 mypy 进行静态检查

#### 安装

```bash
# 基础安装
pip install mypy

# 或使用 conda
conda install mypy -c conda-forge

# 科学计算支持（推荐）
pip install numpy-stubs pandas-stubs types-scipy types-matplotlib

# 验证安装
mypy --version
```

#### 使用

```bash
# 基本使用
mypy your_script.py

# 严格模式（推荐用于新项目）
mypy your_script.py --strict

# 检查整个包
mypy your_package/

# 显示详细的错误信息
mypy your_script.py --show-error-codes

# 配置文件
mypy your_script.py --config-file mypy.ini
```

#### 配置文件 (mypy.ini)

```ini
[mypy]
python_version = 3.9
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

# 特定模块配置
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True
```

#### 常见问题

```bash
# 如果遇到 "cannot import implementation" 错误
pip install --upgrade mypy

# 如果 NumPy 类型检查报错，确保安装了 numpy-stubs
pip install numpy-stubs

# 对于某些第三方库，可以忽略缺失的存根
# 在 mypy.ini 中添加：
[mypy-third_party_library.*]
ignore_missing_imports = True
```

### 2. 使用 VS Code 智能提示

```json
// .vscode/settings.json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.completeFunctionParens": true
}
```

## 📝 最佳实践

### 1. 选择合适的精度

```python
# 明确指定数值精度
def create_model_weights(input_dim: int, output_dim: int) -> np.ndarray:
    """创建模型权重，明确使用 float32"""
    return np.random.randn(input_dim, output_dim).astype(np.float32)

# 使用类型别名
Float32Array = np.ndarray
Float64Array = np.ndarray

def process_image(image: Float32Array) -> Float64Array:
    """处理图像，转换精度"""
    return image.astype(np.float64)
```

### 2. 避免过度复杂化

```python
# ✅ 好的实践：简洁明了
def compute_error(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    """计算预测误差"""
    return float(np.mean((true_vals - pred_vals) ** 2))

# ❌ 避免：过度复杂
def compute_error(
    true_vals: Union[np.ndarray, List[float]],
    pred_vals: Union[np.ndarray, List[float]],
    metric_type: str = "mse"
) -> Union[float, np.ndarray]:
    ...
```

### 3. 使用 Optional 处理缺失值

```python
from typing import Optional

def load_data(file_path: str, normalize: Optional[bool] = None) -> np.ndarray:
    """加载数据，可选择是否归一化"""
    data = np.loadtxt(file_path)

    if normalize is None:
        # 自动判断
        normalize = data.std() > 1.0

    if normalize:
        data = (data - data.mean()) / data.std()

    return data
```

## 🎯 科学计算示例

### 1. 数值积分函数

```python
from typing import Callable, Union
import numpy as np

# 函数类型定义
Integrand = Callable[[float], float]

def trapezoidal_rule(
    f: Integrand,
    a: float,
    b: float,
    n: int = 1000
) -> float:
    """梯形法则数值积分"""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

def adaptive_simpson(
    f: Integrand,
    a: float,
    b: float,
    tol: float = 1e-6
) -> float:
    """自适应辛普森积分"""
    # 实现自适应辛普森算法
    ...
```

### 2. 机器学习模型

```python
from typing import Tuple, Optional
import numpy as np

class LinearRegression:
    """线性回归模型"""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept: bool = fit_intercept
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'LinearRegression':
        """训练模型"""
        if self.fit_intercept:
            X_b = np.c_[np.ones(X.shape[0]), X]
            weights_b = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = float(weights_b[0])
            self.weights = weights_b[1:]
        else:
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
            self.bias = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.weights is None:
            raise ValueError("Model not fitted")
        return X @ self.weights + self.bias

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """计算 R² 分数"""
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot)
```

## 📚 参考资源

- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [PEP 585 - Built-in Generic Types](https://www.python.org/dev/peps/pep-0585/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Type Cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

---

记住：好的类型注解应该像好的文档一样清晰，既要准确又要简洁。在科学计算中，特别要注意数值精度和数据维度的正确表达。