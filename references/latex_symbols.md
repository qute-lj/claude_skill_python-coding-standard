# LaTeX 数学符号速查表

## 重要：矩阵书写规范

在 Python 中书写矩阵时，**强烈建议使用多行字符串并手动对齐**，而不是使用复杂的 LaTeX 环境：

### ✅ 推荐方式：使用多行字符串 + Unicode 字符

```python
# 对齐每一行，使用空格调整位置
matrix_text = 'H(t) = ⎡ αt   V ⎤\n' + \
              '       ⎣ V  -αt ⎦'

# 更大的矩阵
hamiltonian = '⎡ 0   -i   1  ⎤\n' + \
              '⎢ i    0  -i  ⎥\n' + \
              '⎣ 1    i   0  ⎦'

# 使用等宽字体显示
plt.text(0.5, 0.5, matrix_text,
         fontfamily='monospace',
         ha='center', va='center')
```

### ❌ 避免：复杂的 LaTeX 矩阵环境

```python
# 这种方式在 matplotlib 中容易出错
ax.text(0.5, 0.5, r'$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$')
```

### 对齐技巧

1. **计算每列宽度**：
```python
def format_matrix(matrix):
    """格式化矩阵，自动对齐"""
    # 将元素转为字符串
    str_matrix = [[str(cell) for cell in row] for row in matrix]

    # 计算每列最大宽度
    col_widths = [max(len(str_matrix[i][j]) for i in range(len(matrix)))
                  for j in range(len(matrix[0]))]

    # 格式化每行
    formatted_rows = []
    for row in str_matrix:
        formatted_row = '  '.join(
            cell.center(col_widths[j]) for j, cell in enumerate(row)
        )
        formatted_rows.append(formatted_row)

    return '⎡ ' + formatted_rows[0] + ' ⎤\n' + \
           '⎢ ' + formatted_rows[1] + ' ⎥\n' + \
           '⎣ ' + formatted_rows[2] + ' ⎦'
```

2. **使用 tab 字符对齐**（在支持的编辑器中）：
```python
matrix = '⎡ a\t\tb ⎤\n' + \
         '⎢ c\t\td ⎥\n' + \
         '⎣ e\t\tf ⎦'
```

## 常用希腊字母

| 大写 | LaTeX | 小写 | LaTeX | 名称 |
|------|-------|------|-------|------|
| Α | `$A$` | α | `$\alpha$` | Alpha |
| Β | `$B$` | β | `$\beta$` | Beta |
| Γ | `$\Gamma$` | γ | `$\gamma$` | Gamma |
| Δ | `$\Delta$` | δ | `$\delta$` | Delta |
| Ε | `$E$` | ε | `$\epsilon$` | Epsilon |
| Ζ | `$Z$` | ζ | `$\zeta$` | Zeta |
| Η | `$H$` | η | `$\eta$` | Eta |
| Θ | `$\Theta$` | θ | `$\theta$` | Theta |
| Ι | `$I$` | ι | `$\iota$` | Iota |
| Κ | `$K$` | κ | `$\kappa$` | Kappa |
| Λ | `$\Lambda$` | λ | `$\lambda$` | Lambda |
| Μ | `$M$` | μ | `$\mu$` | Mu |
| Ν | `$N$` | ν | `$\nu$` | Nu |
| Ξ | `$\Xi$` | ξ | `$\xi$` | Xi |
| Ο | `$O$` | ο | `$\omicron$` | Omicron |
| Π | `$\Pi$` | π | `$\pi$` | Pi |
| Ρ | `$P$` | ρ | `$\rho$` | Rho |
| Σ | `$\Sigma$` | σ | `$\sigma$` | Sigma |
| Τ | `$T$` | τ | `$\tau$` | Tau |
| Υ | `$\Upsilon$` | υ | `$\upsilon$` | Upsilon |
| Φ | `$\Phi$` | φ | `$\phi$` | Phi |
| Χ | `$X$` | χ | `$\chi$` | Chi |
| Ψ | `$\Psi$` | ψ | `$\psi$` | Psi |
| Ω | `$\Omega$` | ω | `$\omega$` | Omega |

## 常用数学符号

### 运算符
- 加减乘除：`$+$`, `$-$`, `$\times$`, `$\div$`
- 正负：`$\pm$`, `$\mp$`
- 分数：`$\frac{a}{b}$`
- 根号：`$\sqrt{x}$`, `$\sqrt[n]{x}$`
- 求和：`$\sum_{i=1}^{n}$`
- 求积：`$\prod_{i=1}^{n}$`
- 积分：`$\int_a^b$`, `$\oint$`
- 极限：`$\lim_{x \to \infty}$`

### 关系符号
- 等于：`$=$`, `$\equiv$`, `$\approx$`
- 不等：`$\neq$`, `$\not\equiv$`
- 大于小于：`$<$`, `$>$`, `$\leq$`, `$\geq$`
- 远小于：`$\ll$`, `$\gg$`

### 集合符号
- 属于：`$\in$`, `$\notin$`
- 包含：`$\subset$`, `$\subseteq$`, `$\supset$`, `$\supseteq$`
- 空集：`$\emptyset$`
- 并交：`$\cup$`, `$\cap$`
- 全集：`$\mathbb{R}$`, `$\mathbb{C}$`, `$\mathbb{Z}$`, `$\mathbb{N}$`

### 箭头
- 右箭头：`$\rightarrow$`, `$\Rightarrow$`
- 左箭头：`$\leftarrow$`, `$\Leftarrow$`
- 双箭头：`$\leftrightarrow$`, `$\Leftrightarrow$`
- 映射：`$\mapsto$`

## 常用数学函数

- 三角函数：`$\sin$`, `$\cos$`, `$\tan$`, `$\arcsin$`, `$\arccos$`, `$\arctan$`
- 对数指数：`$\exp$`, `$\ln$`, `$\log$`, `$\log_{10}$`
- 极值：`$\max$`, `$\min$`, `$\sup$`, `$\inf$`
- 微分：`$\frac{d}{dx}$`, `$\partial$`
- 偏导：`$\frac{\partial f}{\partial x}$`

## 格式命令

- 粗体：`$\mathbf{x}$`, `$\mathbf{A}$`
- 黑板粗体：`$\mathbb{R}$`, `$\mathbb{N}$`
- 花体：`$\mathcal{L}$`, `$\mathcal{H}$`
- 倾斜：`$\mathit{variable}$`
- 无衬线：`$\mathsf{text}$`
- 打字机：`$\mathtt{code}$`
- 文本模式：`$\text{normal text}$`

## 矩阵和向量

- 向量：`$\vec{v}$`, `$\mathbf{v}$`
- 矩阵：`$\mathbf{A}$`
- 转置：`$\mathbf{A}^T$`, `$\mathbf{A}^\top$`
- 伴随：`$\mathbf{A}^\dagger$`, `$\mathbf{A}^*$`
- 伪逆：`$\mathbf{A}^+$`
- 点积：`$\mathbf{a} \cdot \mathbf{b}$`
- 叉积：`$\mathbf{a} \times \mathbf{b}$`

## 物理量常用表示

- 时间：`$t$`, `$\tau$`
- 频率：`$f$`, `$\omega$`, `$\nu$`
- 波长：`$\lambda$`
- 速度：`$v$`, `$\mathbf{v}$`
- 加速度：`$a$`, `$\mathbf{a}$`
- 质量：`$m$`, `$M$`
- 力：`$F$`, `$\mathbf{F}$`
- 能量：`$E$`, `$\mathcal{E}$`
- 动量：`$p$`, `$\mathbf{p}$`
- 角动量：`$L$`, `$\mathbf{L}$`
- 温度：`$T$`
- 电荷：`$q$`, `$Q$`
- 电场：`$\mathbf{E}$`
- 磁场：`$\mathbf{B}$`
- 电流：`$I$`
- 电压：`$V$`
- 电阻：`$R$`
- 电容：`$C$`
- 电感：`$L$`

## Unicode 矩阵字符

```python
# 方括号矩阵
⎡ a  b ⎤
⎢ c  d ⎥
⎣ e  f ⎦

# 圆括号矩阵
⎛ a  b ⎞
⎜ c  d ⎟
⎝ e  f ⎠

# 花括号矩阵
⎧ a  b ⎫
⎨ c  d ⎬
⎩ e  f ⎭

# 单竖线
⎪ a  b ⎪
⎪ c  d ⎪

# 双竖线
‖ a  b ‖
‖ c  d ‖
```

## 在 matplotlib 中的使用示例

```python
import matplotlib.pyplot as plt

# 设置轴标签
plt.xlabel(r'$\mathbf{Time\ (t)}$ [s]')
plt.ylabel(r'$\mathbf{Amplitude\ A(\omega)}$ [m]')

# 添加简单公式
plt.text(0.5, 0.5, r'$f(t) = A\sin(\omega t + \phi)$')

# 添加矩阵（推荐使用 Unicode）
matrix_text = '⎡ 1  2  3 ⎤\n' + \
              '⎢ 4  5  6 ⎥\n' + \
              '⎣ 7  8  9 ⎦'
plt.text(0.5, 0.3, matrix_text,
         fontfamily='monospace',
         ha='center', va='center',
         fontsize=12)
```

## 注意事项

1. 在 Python 中使用原始字符串：`r'$\alpha$'`
2. 空格需要显式添加：`$\mathbf{Time\ (t)}$`
3. **矩阵优先使用 Unicode 字符对齐，避免 LaTeX 环境**
4. 上下标使用 `^` 和 `_`：`$x^2$`, `$x_i$`
5. 多字符上下标需要花括号：`$x_{max}$`
6. 矩阵对齐时使用等宽字体 `fontfamily='monospace'`