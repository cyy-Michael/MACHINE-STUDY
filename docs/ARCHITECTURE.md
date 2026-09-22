# 初步架构说明

## 目标

仓库采用“数据层—算法层—服务层—接口层—前端层”的单向依赖。算法不读取文件、不依赖 HTTP；前端不直接调用算法；这样可以让算法成员、后端成员和前端成员并行工作。

```text
Vue 页面 / ECharts
        │ HTTP + WebSocket
        ▼
backend/api
        │ 请求校验、任务状态、结果 JSON
        ▼
backend/services
        │ 训练编排、计时、评估、实验记录
        ├──────────────┐
        ▼              ▼
backend/data     backend/algorithms
读取/预处理       fit / predict / transform
        │              │
        └──────┬───────┘
               ▼
       本地数据集 / 实验结果
```

## 模块职责

| 模块 | 允许负责的内容 | 不应放入的内容 |
|---|---|---|
| `backend/algorithms` | 算法数学逻辑、参数、模型状态 | 文件路径、HTTP、页面代码 |
| `backend/data` | CSV/ARFF/Excel 适配、缺失值、编码、划分、标准化 | 算法内部决策 |
| `backend/services` | 统一训练流程、计时、指标、结果对象 | Vue 组件和路由 |
| `backend/api` | FastAPI 路由、请求响应、WebSocket | 直接实现算法 |
| `frontend` | 页面、参数表单、状态、ECharts | 读取本地文件、训练模型 |
| `tests` | 各模块可重复验证 | 临时手工脚本 |

## 一次实验的数据流

1. 前端提交数据集名、任务类型、算法名和参数。
2. API 层校验请求，并交给实验服务。
3. 数据层读取数据字典和原始文件，构造 `X/y`。
4. 预处理只在训练集上拟合统计量，再应用到测试集，避免数据泄露。
5. 实验服务通过 `registry.create_algorithm()` 创建模型。
6. 模型执行 `fit`、`predict`；分类模型可额外执行 `predict_proba`。
7. 评估服务根据任务类型生成分类指标、回归指标或聚类指标。
8. API 返回统一的实验结果 JSON，前端负责表格和图表展示。

## 算法扩展流程

1. 在 `backend/algorithms/` 新建模型文件。
2. 继承 `BaseModel`，实现 `fit` 和 `predict`。
3. 设置 `algorithm_name` 与 `task_type`。
4. 在 `registry.py` 中加入显示名、默认参数和工厂函数。
5. 在 `tests/` 增加形状、参数错误和基本效果测试。
6. 在 README 或报告中补充算法原理和实验分析。

## 预留接口

`backend/api/README.md` 记录后续接口边界；当前不绑定具体 Web 框架，避免在算法核心尚未稳定时扩大改动范围。
