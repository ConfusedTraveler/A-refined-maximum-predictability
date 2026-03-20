# 实验复现总结

## 项目信息

- **项目名称**: A-refined-maximum-predictability-for-next-location-prediction
- **论文标题**: A refined maximum predictability for next location prediction with fusion knowledge
- **复现时间**: 2026-03-20
- **复现工具**: Kimi Code CLI + uv

---

## 复现步骤

### 1. 环境配置
```bash
# 创建虚拟环境 (Python 3.8)
uv venv --python 3.8

# 安装依赖
uv pip install pandas numpy
```

### 2. 数据预处理
```bash
cd data_preprocessing
..\.venv\Scripts\python.exe preprocesing_p1.py
```
- 处理 1,080 个用户的 Foursquare NYC 数据
- 生成时间特征和位置序列

### 3. RMP计算
```bash
cd "refined maximum predictability"
..\.venv\Scripts\python.exe "refined maximum predictability train.py"
```
- 计算 6 种不同的可预测性指标
- 为每个用户生成 RMP 配置文件

### 4. 模型训练与评估
```bash
cd "improved markov"

# 基础马尔可夫模型
..\.venv\Scripts\python.exe "base markov model_v2.py"

# 改进马尔可夫模型
..\.venv\Scripts\python.exe "improved_markov model_v2.py"
```

---

## 复现结果

| 指标 | 基础马尔可夫 | 改进马尔可夫 (RMP) | 提升幅度 |
|------|------------|-------------------|---------|
| Top-1 | 23.18% | 25.05% | **+8.1%** |
| Top-5 | 48.50% | 56.11% | **+15.7%** |
| Top-10 | 57.58% | 67.30% | **+16.8%** |

**测试样本总数**: 20,884

---

## 文件结构

```
A-refined-maximum-predictability/
├── docs/
│   └── project_analysis.md       # 项目详细分析报告
├── results/
│   ├── experiment_results.md     # 实验结果详细报告
│   ├── base_markov_output.txt    # 基础模型输出
│   ├── improved_markov_output.txt # 改进模型输出
│   └── rmp_train_sample.txt      # RMP计算结果样本
├── paper/
│   └── Huang_2026_Refined_Maximum_Predictability.pdf  # 原论文
├── data_preprocessing/
│   ├── data/foursquare_nyc.txt   # 原始数据
│   └── base_foursquare_nyc_v1.txt # 预处理后的数据
├── refined maximum predictability/
│   ├── refined maximum predictability train.py
│   ├── refined maximum predictability_all.py
│   └── foursquare_nyc_rmp_train.txt # RMP训练结果
├── improved markov/
│   ├── base markov model_v2.py
│   ├── improved_markov model_v2.py
│   ├── foursquare_nyc_base_markov.txt
│   └── foursquare_nyc_improved_markov.txt
├── .venv/                         # uv管理的虚拟环境
├── README.md
└── REPRODUCTION_SUMMARY.md        # 本文件
```

---

## 复现成功确认

- ✅ 数据预处理成功
- ✅ RMP计算成功
- ✅ 基础马尔可夫模型运行成功
- ✅ 改进马尔可夫模型运行成功
- ✅ 结果与论文报告一致

---

## 依赖信息

```
Python: 3.8.20
numpy: 1.24.4
pandas: 2.0.3
python-dateutil: 2.9.0.post0
pytz: 2026.1.post1
six: 1.17.0
tzdata: 2025.3
```

---

*复现完成于 2026-03-20*
