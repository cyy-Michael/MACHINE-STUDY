# 机器学习实验与可视化平台

面向课程大作业的机器学习实验平台后端。目标形态是「网页上选数据集 → 选算法 →
调参数 → 点训练 → 看指标和图表」，本仓库负责**从数据集到结果 JSON 的整条链路**，
并对外提供 HTTP / WebSocket 接口。

前端由组内其他成员负责，见 [docs/TEAM_WORK.md](docs/TEAM_WORK.md)。

## 快速开始

```powershell
# 1) 建虚拟环境并装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # 运行必需
pip install -r requirements-dev.txt      # 跑测试才需要（含 httpx2）

# 2) 放好数据集（本体不进仓库，见下一节）
#    默认读取仓库根的 数据集\ 目录

# 3) 跑测试
python -m unittest discover              # 期望 157 tests OK

# 4) 启动接口服务
python -m uvicorn backend.api.main:app --port 8000 --reload
#    接口文档 http://127.0.0.1:8000/docs
#    健康检查 http://127.0.0.1:8000/health

# 5) 命令行演示（会跑真实数据集，不需要启动服务）
python -m backend.main algorithms
python -m backend.main demo
```

> **本机注意**：PATH 里的 `python` 可能是 Microsoft Store 的占位程序，会报
> 「找不到指定的路径」。请使用真实解释器或项目内 `.venv\Scripts\python.exe`。

## 数据集

**数据集本体不提交到 Git**（`.gitignore` 已排除 `数据集/` 以及 `*.csv`/`*.arff`/
`*.xlsx`/`*.xls`）。换一台电脑时把数据放到仓库根的 `数据集/` 目录即可；
也可以设置环境变量指向别处：

```powershell
$env:ML_DATA_ROOT = 'D:\somewhere\数据集'
```

已登记 **8 个数据集**，元数据在 `backend/data/datasets/*.json`：

| 数据集 | 任务 | 规模 | 格式 | 特点 |
|---|---|---|---|---|
| `iris` | 分类 | 147×4 | CSV 无表头 | 最干净的小规模基准；文件 150 行含 3 行重复 |
| `car` | 分类 | 1728×6 | CSV 无表头 | 全类别特征，类别不平衡 |
| `phishing` | 分类 | 5849×30 | ARFF | 文件 11055 行含 **5206 行重复**（47%） |
| `credit` | 分类 | 690×15 | CSV 无表头 | 含 `?` 缺失值 |
| `ckd` | 分类 | 400×24 | ARFF | 10% 缺失，另有 3 行格式缺陷 |
| `bike_day` | 回归 | 731×11 | CSV 有表头 | 已剔除泄露列 |
| `airquality` | 回归 | 8991×12 | CSV 分号 | 脏数据：逗号小数点、`-200` 缺失 |
| `concrete` | 回归 | 1005×8 | Excel `.xls` | 需要 `xlrd`；文件 1030 行含 25 行重复 |

> 表中的规模是**清洗后的可用样本数**，不是文件行数。加载时会依次做三件事：
> 修同行宽错位、剔除目标缺失的样本、删除完全重复的行（重复行会让同一行同时
> 落进训练集和测试集，指标虚高）。接口二会分别报出 `n_rows_raw` 与这三个计数。

新增数据集：在 `backend/data/datasets/` 放一份 JSON（列名、类型、目标列、
缺失标记、要剔除的列），不需要写解析代码。元数据与文件列数不一致时会直接报错，
不会静默跑错。详见 [docs/DATASET_CATALOG.md](docs/DATASET_CATALOG.md)。

## 接口一览

| 接口 | 说明 |
|---|---|
| `GET /api/datasets` | 数据集列表，可按 `task_type` 筛选 |
| `GET /api/datasets/{name}` | 字段、类型、样例、缺失统计、标签分布 |
| `GET /api/algorithms` | 算法列表，可按 `task_type` 筛选 |
| `POST /api/experiments` | 创建实验，**立即返回 202**，训练在线程池执行 |
| `GET /api/experiments/{id}` | 查询实验状态与完整结果 |
| `GET /api/experiments` | 历史实验列表 |
| `WS /ws/experiments/{id}` | 训练进度实时推送 |
| `GET /health` | 健康检查 |

注意 WebSocket 按规范**不带** `/api` 前缀。每个接口的请求、响应、错误码
和示例见 [docs/api.md](docs/api.md)。

```powershell
# 手最快的一次验证
curl.exe -X POST http://127.0.0.1:8000/api/experiments `
  -H "Content-Type: application/json" `
  -d '{\"dataset\":\"iris\",\"algorithm\":\"knn\",\"parameters\":{\"k\":5}}'
```

## 目录职责

```text
backend/
├── logging_config.py 统一日志：控制台 + logs/backend.log（轮转，UTF-8）
├── algorithms/       10 个算法实现；统一 fit/predict，走 registry 注册
├── api/main.py       HTTP/WebSocket 接口层、CORS、统一错误响应、请求日志
├── data/
│   ├── formats.py    脏 CSV / ARFF / Excel 读取、行宽修复
│   ├── encoding.py   缺失值、类型推断、one-hot、重复行、防泄露填充
│   ├── registry.py   数据集登记表（读 metadata）
│   ├── datasets/     8 份数据集元数据 JSON
│   └── preprocessing.py  划分、分层 K 折、标准化
├── services/
│   ├── experiment.py 端到端编排：数据集 → X/y → 训练 → 结果 JSON
│   ├── evaluation.py 分类 / 回归 / 聚类指标
│   ├── runner.py     线程池执行 + 进度订阅
│   ├── store.py      实验记录落盘
│   └── train_service.py  单次实验与交叉验证
└── main.py           本地 CLI 入口
frontend/             前端（由其他成员负责）
tests/                157 个用例
experiments/results/  实验结果（不提交）
logs/                 运行日志（不提交）
数据集/                本地数据集（不提交）
```

详细分层见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 文档导航

| 我想…… | 看这里 |
|---|---|
| 把项目跑起来 | 本文件上面的「快速开始」 |
| 对接接口、看请求响应示例 | [docs/api.md](docs/api.md) |
| **改代码、加数据集/算法/接口、排错** | **[docs/BACKEND_GUIDE.md](docs/BACKEND_GUIDE.md)** |
| 理解分层设计 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 看数据集字段说明 | [docs/DATASET_CATALOG.md](docs/DATASET_CATALOG.md) |
| 看组内分工 | [docs/TEAM_WORK.md](docs/TEAM_WORK.md) |

## 必须遵守的约定

1. **统一模型接口**：继承 `BaseModel`，实现 `fit`/`predict`；分类提供
   `predict_proba`；聚类与降维提供 `transform`。参数非法抛 `ValueError`，
   未 fit 先 predict 抛 `RuntimeError`。
2. **建模只走注册表**：`registry.create_algorithm(name, **params)`，
   不要在 API 里写算法分支。
3. **预处理防泄露**（评分点）：划分和所有统计量（缺失值填充、标准化）
   **只能来自训练集**。编码层故意保留 `np.nan`，把填充推迟到
   `train_test_split` 之后，就是为了这件事。
4. **算法层不读文件、不依赖 HTTP**；API 层不实现算法。
5. **结果 JSON 里不放模型对象**——它不可序列化，且 `inf`/`nan` 也不是合法 JSON。
6. **库代码用 `logging`，不用 `print`**。`get_logger(__name__)` 是唯一入口；测试用断言
   而不是打印结果（曾经因为 print emoji 在 GBK 控制台抛异常，把整个测试套件搞挂）。
   唯一例外是 `backend/main.py` 的 CLI —— 它的 `print` 是给用户看的命令行输出，那是它的职责。

## 测试

```powershell
python -m unittest discover              # 全部
python -m unittest discover -s tests -v  # 详细输出
```

数据集本体不在仓库里，所以依赖真实数据的 50 个用例在缺少数据时会**自动跳过**
而不是失败：

```
有数据  -> Ran 157 tests, OK
无数据  -> Ran 157 tests, OK (skipped=59)
```

## 日志

日志同时输出到控制台和 `logs/backend.log`（UTF-8，单文件 2 MB、保留 3 个备份）：

```powershell
$env:ML_LOG_LEVEL = 'DEBUG'    # 默认 INFO
$env:ML_LOG_DIR   = 'D:\mylogs'
$env:ML_LOG_TO_FILE = '0'      # 只输出到控制台
```

每个响应都带 `X-Process-Time-Ms` 响应头。训练失败的完整堆栈可在
`GET /api/experiments/{id}` 的 `traceback` 字段取到。详见
[docs/BACKEND_GUIDE.md](docs/BACKEND_GUIDE.md#七日志)。

## Git 协作

- `main`：稳定版本（受保护，改动走 PR）。
- 提交信息使用 `feat:`、`fix:`、`test:`、`docs:`、`refactor:`、`chore:` 前缀。
- 不要提交 `.venv/`、`.vs/`、`__pycache__/`、`数据集/`、`教案/`、`experiments/results/`。
