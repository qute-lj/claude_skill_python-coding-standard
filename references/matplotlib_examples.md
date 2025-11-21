# Matplotlib 科学绘图最佳实践示例

## 📌 输出文件规范

所有图表文件请遵循 `output_standards.md` 中的命名规范：
- 路径：`output/figures/`
- 命名：`{时间戳}_{参数组}_{描述}.png`
- 示例：`241121_lr0.001_loss.png`

## 设置参数

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

# 设置 matplotlib 参数
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 辅助函数：生成标准输出路径
def save_figure(description, params=None, subdir='figures'):
    """保存图表到标准路径

    Args:
        description: 图表描述（如 'loss', 'accuracy'）
        params: 参数字典（如 {'lr': 0.001, 'batch_size': 32}）
        subdir: 子目录名（默认 'figures'）
    """
    # 时间戳
    timestamp = datetime.now().strftime('%y%m%d%H%M')

    # 参数组
    if params:
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
        param_str = ''

    # 生成文件名
    if param_str:
        filename = f"{timestamp}_{param_str}_{description}.png"
    else:
        filename = f"{timestamp}_{description}.png"

    # 生成标题（不带时间戳）
    if param_str:
        title_text = f"{param_str}_{description}"
    else:
        title_text = description

    # 创建路径
    output_dir = os.path.join('output', subdir)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # 保存
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    print(f"标题建议: {title_text}")

    return filepath, title_text
```

## 示例 1：带 LaTeX 公式的科学图表

```python
# 生成数据
t = np.linspace(0, 2*np.pi, 200)
gamma = 2 * np.pi * np.exp(-t/2) * np.sin(5*t)

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, gamma, 'b-', linewidth=2, label=r'$\gamma(t)$')

# 添加标签（使用粗体）
ax.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=12)
ax.set_ylabel(r'$\mathbf{Berry\ Phase\ (\pi\ units)}$', fontsize=12)
ax.set_title(r'$\mathbf{Landau-Zener\ Transition}$', fontsize=14)

# 添加公式文本
ax.text(0.05, 0.95,
        r'$\gamma(t) = \frac{|\langle m|\partial_t H|n\rangle|}{|E_m - E_n|^2}$',
        transform=ax.transAxes,
        fontsize=11,
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# 添加物理解释
ax.text(0.05, 0.85,
        r'$\text{Landau-Zener: } P_{LZ} = \exp\left(-\frac{2\pi V^2}{\alpha}\right)$',
        transform=ax.transAxes,
        fontsize=11,
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# 完善图表
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
plt.tight_layout()

# 使用标准保存函数
params = {'lr': 0.001, 'batch_size': 32}
filepath, title_text = save_figure('loss_curve', params)

# 设置标题（与文件名一致，去掉时间戳）
ax.set_title(title_text, fontsize=13)
plt.show()
```

## 示例 2：3D 图表与 2D 文本标注

```python
# 创建 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成 3D 表面数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)/2) * np.cos(2*np.pi*np.sqrt(X**2 + Y**2))

# 绘制表面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=20)

# 设置 3D 标签
ax.set_xlabel(r'$\mathbf{k_x}$', fontsize=11)
ax.set_ylabel(r'$\mathbf{k_y}$', fontsize=11)
ax.set_zlabel(r'$\mathbf{E(k)}$', fontsize=11)
ax.set_title(r'$\mathbf{Graphene\ Dispersion\ Relation}$', fontsize=13)

# 使用 text2D 添加 2D 文本（推荐）
ax.text2D(0.02, 0.02,
          r'$H(k) = v_F(k_x \sigma_x + k_y \sigma_y)$',
          transform=ax.transAxes,
          fontsize=11,
          ha='left', va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# 添加额外注释
ax.text2D(0.98, 0.02,
          r'$\text{Dirac\ Points: } K, K^{\prime}$',
          transform=ax.transAxes,
          fontsize=11,
          ha='right', va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.9))

# 调整视角
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig('3d_plot.png', dpi=300)
plt.show()
```

## 示例 3：矩阵可视化（避免复杂 LaTeX）

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 创建哈密顿量数据
V = 1.0
alpha = 0.5
t_values = np.linspace(-2, 2, 100)
eigenvalues_upper = []
eigenvalues_lower = []

for t in t_values:
    H = np.array([[alpha * t, V],
                 [V, -alpha * t]])
    eigvals = np.linalg.eigvalsh(H)
    eigenvalues_upper.append(eigvals[1])
    eigenvalues_lower.append(eigvals[0])

# 绘制本征值
ax1.plot(t_values, eigenvalues_upper, 'b-', linewidth=2, label=r'$\text{Upper\ Level}$')
ax1.plot(t_values, eigenvalues_lower, 'r-', linewidth=2, label=r'$\text{Lower\ Level}$')
ax1.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=11)
ax1.set_ylabel(r'$\mathbf{Energy\ (E)}$', fontsize=11)
ax1.set_title(r'$\mathbf{Energy\ Levels\ Evolution}$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 使用 Unicode 字符显示矩阵（避免复杂 LaTeX）
matrix_text = 'H(t) = ⎡ αt   V ⎤\n' + \
              '       ⎣ V  -αt ⎦'

ax1.text(0.5, 0.5, matrix_text,
         transform=ax1.transAxes,
         fontsize=12,
         fontfamily='monospace',
         ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))

# 绘制能隙
ax2.plot(t_values, np.array(eigenvalues_upper) - np.array(eigenvalues_lower),
         'g-', linewidth=2)
ax2.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=11)
ax2.set_ylabel(r'$\mathbf{Energy\ Gap}$', fontsize=11)
ax2.set_title(r'$\mathbf{Avoided\ Crossing}$', fontsize=12)
ax2.grid(True, alpha=0.3)

# 添加 gap 公式
ax2.text(0.5, 0.8,
         r'$\Delta E = 2\sqrt{V^2 + \alpha^2 t^2}$',
         transform=ax2.transAxes,
         fontsize=11,
         ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('matrix_plot.png', dpi=300)
plt.show()
```

## 示例 4：多子图布局

```python
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# 子图 1: 主图
ax1 = fig.add_subplot(gs[0, :2])
x = np.linspace(0, 10, 500)
y1 = np.sin(x) * np.exp(-x/5)
y2 = np.cos(x) * np.exp(-x/5)

ax1.plot(x, y1, 'b-', label=r'$\sin(x)e^{-x/5}$', linewidth=2)
ax1.plot(x, y2, 'r-', label=r'$\cos(x)e^{-x/5}$', linewidth=2)
ax1.set_xlabel(r'$\mathbf{x}$', fontsize=12)
ax1.set_ylabel(r'$\mathbf{f(x)}$', fontsize=12)
ax1.set_title(r'$\mathbf{Damped\ Oscillations}$', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 子图 2: 相空间图
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(y1, y2, 'g-', linewidth=1.5)
ax2.set_xlabel(r'$\mathbf{Re}$', fontsize=11)
ax2.set_ylabel(r'$\mathbf{Im}$', fontsize=11)
ax2.set_title(r'$\mathbf{Phase\ Space}$', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# 子图 3: 频谱
ax3 = fig.add_subplot(gs[1, 1])
freqs = np.fft.fftfreq(len(x), x[1]-x[0])
fft_vals = np.abs(np.fft.fft(y1 + 1j*y2))

ax3.semilogy(freqs[:len(freqs)//2], fft_vals[:len(freqs)//2], 'b-')
ax3.set_xlabel(r'$\mathbf{Frequency}$', fontsize=11)
ax3.set_ylabel(r'$\mathbf{Amplitude}$', fontsize=11)
ax3.set_title(r'$\mathbf{Fourier\ Spectrum}$', fontsize=12)
ax3.grid(True, alpha=0.3)

# 子图 4: 信息文本
ax4 = fig.add_subplot(gs[:, 2])
ax4.axis('off')

info_text = r'$\textbf{System Properties:}$' + '\n\n' + \
            r'$\text{Damping: } \gamma = 0.2$' + '\n' + \
            r'$\text{Frequency: } \omega_0 = 1$' + '\n' + \
            r'$\text{Quality: } Q = \omega_0/\gamma = 5$' + '\n\n' + \
            r'$\textbf{Equation:}$' + '\n' + \
            r'$\ddot{x} + \gamma\dot{x} + \omega_0^2 x = 0$'

ax4.text(0.1, 0.9, info_text,
         transform=ax4.transAxes,
         fontsize=11,
         ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

plt.tight_layout()
plt.savefig('multi_subplot.png', dpi=300)
plt.show()
```

## 示例 5：误差棒和拟合

```python
fig, ax = plt.subplots(figsize=(8, 6))

# 生成带误差的实验数据
x_exp = np.linspace(0, 10, 20)
y_true = 2 * x_exp + 1 + 0.5 * x_exp**2 / 10
y_exp = y_true + np.random.normal(0, 0.5, len(x_exp))
y_err = np.random.uniform(0.3, 0.7, len(x_exp))

# 绘制实验数据点
ax.errorbar(x_exp, y_exp, yerr=y_err, fmt='ro',
            markersize=6, capsize=3, label='实验数据')

# 多项式拟合
coeffs = np.polyfit(x_exp, y_exp, 2)
x_fit = np.linspace(0, 10, 100)
y_fit = np.polyval(coeffs, x_fit)

# 绘制拟合曲线
ax.plot(x_fit, y_fit, 'b-', linewidth=2, label='拟合曲线')

# 添加拟合公式
fit_text = r'$y = %.2fx^2 + %.2fx + %.2f$' % (coeffs[0], coeffs[1], coeffs[2])
ax.text(0.05, 0.95, fit_text,
        transform=ax.transAxes,
        fontsize=11,
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# 计算 R²
y_pred = np.polyval(coeffs, x_exp)
ss_res = np.sum((y_exp - y_pred) ** 2)
ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# 添加 R² 值
ax.text(0.05, 0.88, r'$R^2 = %.4f$' % r_squared,
        transform=ax.transAxes,
        fontsize=11,
        ha='left', va='top')

# 完善图表
ax.set_xlabel(r'$\mathbf{x}$', fontsize=12)
ax.set_ylabel(r'$\mathbf{y}$', fontsize=12)
ax.set_title(r'$\mathbf{实验数据拟合}$', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('fit_plot.png', dpi=300)
plt.show()
```

## 最佳实践要点总结

### 1. LaTeX 公式使用

- **简单公式**：使用 `r'$\formula$'` 格式
- **一般文本**：使用 `\text{}`
- **重要标签**：使用 `\mathbf{}`
- **避免复杂环境**：使用 Unicode 字符替代

### 2. 文本定位

- 3D 图中使用 `text2D` 而非 `text`
- 使用 `transform=ax.transAxes` 进行相对定位
- 通过 `ha` 和 `va` 控制对齐方式

### 3. 避免遮挡

- 使用透明度 `alpha`
- 添加背景框 `bbox`
- 选择空白区域放置文本

### 4. 代码执行

使用 conda 运行代码，避免环境问题：
```bash
conda run -n your_env python your_script.py
```