# ε-RMP: Epsilon-Robust Multi-scale Predictability
## 基于容差的鲁棒多尺度可预测性指标

## 论文核心思想提炼

从Jamal Mohammed的博士论文中提取的关键创新：

### 1. 容差匹配（Tolerance-based Matching）
**原问题**：离散化导致临近值被错误分类  
**解决方案**：使用容差ε，|x - y| ≤ ε 时认为匹配

### 2. Pmax公式
论文中的核心公式（公式3.19）：
```
H(X) ≤ Pmax·log₂Pmax + (1-Pmax)·[log₂(1-Pmax) + log₂(N-2)]
```

其中：
- H(X): 时间序列的熵率
- Pmax: 最大可预测性
- N: 基于容差的有效字母表大小 N = (xmax - xmin)/ε

### 3. NLZ1/NLZ2算法
- NLZ1: 基于字典的数值Lempel-Ziv解析
- NLZ2: 基于历史匹配的数值熵估计

### 4. 多步预测扩展
使用Lyapunov指数：Pmax^h = Pmax^0 · exp(λ⁺_min · h)

---

## ε-RMP 指标设计

### 核心创新点

**创新1: 自适应容差选择**
不同于固定容差，根据数据分布自适应选择ε：
```
ε = α · σ(data)  # α默认为0.1，σ为标准差
```

**创新2: 容差感知的转移熵**
不是精确匹配位置，而是匹配容差范围内的位置：
```python
def tolerance_match(loc1, loc2, tolerance):
    return abs(loc1 - loc2) <= tolerance
```

**创新3: 多尺度容差金字塔**
不同时间尺度使用不同容差：
- 短期: 小容差 (精确匹配)
- 中期: 中等容差
- 长期: 大容差 (粗粒度匹配)

**创新4: 鲁棒熵估计**
结合论文的NLZ2思想和MRP的Miller-Madow校正

---

## 数学定义

### 1. 自适应容差
```
ε_short = 0.05 · σ(data)
ε_medium = 0.15 · σ(data)  
ε_long = 0.30 · σ(data)
```

### 2. 容差感知的条件熵
对于给定的容差ε：
```
p_ε(y|x) = #{j: d_j ≈_ε y and d_{j-1} ≈_ε x} / #{j: d_{j-1} ≈_ε x}
H_ε(Y|X) = -Σ p_ε(y|x) log₂ p_ε(y|x)
```

### 3. 多尺度融合
```
H_εRMP = (2·H_short + H_medium + 0.5·H_long) / 3.5
```

### 4. 有效字母表大小
```
N_ε = (max(data) - min(data)) / ε + 1
```

### 5. 最终ε-RMP
使用论文的Pmax公式求解：
```
H_εRMP = Pmax·log₂Pmax + (1-Pmax)·[log₂(1-Pmax) + log₂(N_ε-2)]
```

---

## 与已有指标的对比

| 特性 | RMP | ATDP | MRP | ε-RMP |
|------|-----|------|-----|-------|
| 容差匹配 | ❌ | ❌ | ❌ | ✅ |
| 自适应容差 | - | - | - | ✅ |
| 多尺度 | 6维度硬选 | 3尺度 | 3尺度加权 | 3尺度+容差金字塔 |
| 熵校正 | ❌ | ❌ | Miller-Madow | Miller-Madow |
| 时间衰减 | ❌ | 过度 | 半衰期 | 半衰期+容差 |
| 理论基础 | Fano不等式 | Fano | Fano | 论文Pmax公式 |

---

## 预期优势

1. **鲁棒性**: 容差匹配对噪声更鲁棒
2. **自适应性**: 根据数据分布自动调整
3. **多粒度**: 不同时间尺度不同精度
4. **理论支撑**: 基于论文的Pmax公式

---

*设计时间: 2026-03-20*
*参考: Jamal Mohammed, "A Model-Agnostic Upper Bound for Univariate Time Series Prediction", PhD Thesis, University of Zurich, 2025*
