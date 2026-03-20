# MRP: Multi-scale Recency-aware Predictability
## 多尺度近期感知可预测性指标

### 论文发现与启发

根据搜索的相关论文，发现以下关键洞察：

1. **多horizon估计** [Cognitive Biases in Agentic AI-Driven 6G]
   - 使用不同衰减率的并行估计器
   - 加权融合：θ_fused = Σ ω_h * θ^(h)

2. **时间衰减公式** [DAO-GP, LinkedIn Feed SR]
   - 标准指数衰减：w_t = exp(-γ·Δt)
   - 半衰期概念：60天半衰期意味着2个月前的样本权重为50%

3. **位置加权** [LinkedIn Feed SR]
   - 序列中早期位置降权
   - 最后位置（最近）获得全权重
   - 第一个位置获得50%权重

4. **多尺度分析** [Song et al. 原始论文]
   - 不同时间分辨率（1小时、2小时、4小时）影响可预测性
   - 需要综合考虑多个时间尺度

---

### MRP 指标设计

#### 核心改进点

**改进1: 使用半衰期而非固定衰减率**
```
w_t = exp(-ln(2) * Δt / t_half)
```
- t_half: 数据驱动的半衰期
- 避免ATDP中过度衰减的问题

**改进2: 数据驱动的半衰期计算**
```
t_half = median(time_gap_between_revisits) * α
```
- 基于用户实际行为模式计算
- α 为调节参数（默认0.5）

**改进3: 多尺度平滑融合**
- 短期尺度 (最近10个): 关注即时模式
- 中期尺度 (最近50个): 关注周期性
- 长期尺度 (全部): 关注总体分布
- 使用平滑权重过渡，避免硬切换

**改进4: 熵的偏差校正**
- 使用Miller-Madow校正：S_mm = S_ml + (m-1)/(2n)
- 解决小样本下的熵估计偏差

---

### 数学定义

#### 1. 半衰期计算
```
gaps = [time_i - time_{i-1} for all consecutive same-location visits]
t_half = median(gaps) * 0.5 if len(gaps) > 0 else default_half_life
```

#### 2. 时间权重
```
weight_i = exp(-ln(2) * (T_now - T_i) / t_half)
```

#### 3. Miller-Madow熵校正
```
S_corrected = S_raw + (m - 1) / (2 * n)
```
- m: 不同符号数
- n: 总样本数

#### 4. 多尺度融合
```
S_multiscale = (2*S_short + S_medium + 0.5*S_long) / 3.5
```

#### 5. 最终MRP
基于Song et al.公式求解最大可预测性

---

### 与ATDP的关键区别

| 特性 | ATDP (失败) | MRP (改进) |
|------|-------------|-----------|
| 衰减公式 | exp(-λ·√t) | exp(-ln(2)·Δt/t_half) |
| 半衰期 | 自适应λ计算 | 基于实际重访间隔 |
| 多尺度 | 3尺度等权 | 加权融合(短期优先) |
| 熵估计 | 原始熵 | Miller-Madow校正 |
| 预测策略 | 硬分类3类 | 平滑过渡 |

---

### 预期改进

1. **更合理的衰减**: 基于实际行为而非经验公式
2. **更稳定的熵估计**: 偏差校正减少小样本误差
3. **更平滑的多尺度融合**: 避免某尺度主导
4. **更好的预测性能**: 预期接近或超越RMP

---

*设计时间: 2026-03-20*
