# 架构说明

## 目标

仓库采用“数据层—算法层—服务层—接口层—前端层”的单向依赖。算法不读取文件、不依赖 HTTP；前端不直接调用算法；这样可以让算法成员、后端成员和前端成员并行工作。

```text
Vue 页面 / ECharts
        │ HTTP + WebSocket
        ▼
backend/api               路由、请求校验、统一错误响应、请求日志
        │ 任务状态、结果 JSON
        ▼
backend/services          训练编排、线程池、计时、评估、实验记录
        │
        ├──────────────┐
        ▼              ▼
backend/data     backend/algorithms
读取/预处理       fit / predict / transform
        │              │
        └──────┬───────┘
               ▼
       本地数据集 / 实验结果 / 日志
```

横切关注点：`backend/logging_config.py` 提供统一日志，被上面各层共用。

## 模块职责

| 模块 | 允许负责的内容 | 不应放入的内容 |
|---|---|---|
| `backend/algorithms` | 算法数学逻辑、参数、模型状态 | 文件路径、HTTP、页面代码 |
| `backend/data` | CSV/ARFF/Excel 适配、缺失值、重复行、编码、划分、标准化 | 算法内部决策 |
| `backend/services` | 统一训练流程、线程池、计时、指标、结果对象、实验落盘 | Vue 组件和路由 |
| `backend/api` | FastAPI 路由、请求响应、WebSocket、错误响应、请求日志 | 直接实现算法 |
| `backend/logging_config.py` | 日志级别、输出目标（控制台 + 轮转文件） | 业务逻辑 |
| `frontend` | 页面、参数表单、状态、ECharts | 读取本地文件、训练模型 |
| `tests` | 各模块可重复验证 | 临时手工脚本 |

## 一次实验的数据流

1. 前端提交数据集名、任务类型、算法名和参数。
2. API 层校验请求（先用元数据校验，不读文件），交给实验运行器。
3. 运行器把任务丢进线程池，**立即**返回 `experiment_id`；进度推给 WebSocket。
4. 数据层读取原始文件，修同行宽错位、剔除目标缺失行与完全重复行，构造 `X/y`。
5. 划分训练/测试集。
6. **只用训练集**拟合缺失值填充与标准化统计量，再套用到测试集（防泄露，评分点）。
7. 通过 `registry.create_algorithm()` 创建模型，执行 `fit`（计时）与 `predict`（计时）。
8. 评估服务按任务类型生成分类、回归或聚类指标。
9. 结果做严格 JSON 安全化（剔除模型对象、把 `inf`/`nan` 转成 `null`）后落盘并返回。
10. 前端用 `result.metrics` 画指标、`result.visualization` 画图。

## 日志

- 统一走 `backend/logging_config.get_logger(__name__)`，不用裸 `print`。
- 输出：控制台 + `logs/backend.log`（UTF-8，单文件 2 MB、保留 3 个备份）。
- 级别由 `ML_LOG_LEVEL` 控制，默认 `INFO`。
- 请求日志由 `backend/api/main.py` 的中间件产生，含方法、路径、状态码、耗时。
- 训练异常的堆栈会写进实验记录，可在接口五的 `traceback` 字段取到。

## 算法扩展流程

1. 在 `backend/algorithms/` 新建模型文件。
2. 继承 `BaseModel`，实现 `fit` 和 `predict`。
3. 设置 `algorithm_name` 与 `task_type`。
4. 在 `registry.py` 中加入显示名、默认参数和工厂函数。
5. 在 `tests/` 增加形状、参数错误和基本效果测试。
6. 在 README 或报告中补充算法原理和实验分析。

## 数据集扩展流程

1. 在 `backend/data/datasets/` 新建 `<name>.json`。
2. 填列名、类型、目标列、缺失标记、要剔除的列。
3. 数据文件放到 `数据集/` 下（或设 `ML_DATA_ROOT`），路径写进元数据。
4. 跑 `python -m unittest discover` —— 元数据与文件列数不一致会直接报错。

## 接口扩展流程

1. 在 `backend/api/main.py` 的 `api_router` 上加路由（自动带 `/api` 前缀）。
2. WebSocket 路由直接挂 `app`，**不要**放进 `api_router`（规范里它不带 `/api`）。
3. 业务逻辑放 `backend/services/`，不要写在路由函数里。
4. 抛异常即可，统一错误响应由 `@app.exception_handler` 转换。
5. 在 `tests/test_api.py` 补用例，并更新 `docs/api.md`。

详见 [BACKEND_GUIDE.md](BACKEND_GUIDE.md)（面向队友的开发与排错指南）。

