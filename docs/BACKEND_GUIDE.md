# 后端开发与排错指南

> **这份文档给谁看**：接手后端的队友、要对接接口的前端同学、以及做汇报时需要说清楚
> 「后端做了什么」的人。
>
> 想跑起来看 [README](../README.md)；想看接口细节看 [api.md](api.md)；
> 想看分层设计看 [ARCHITECTURE.md](ARCHITECTURE.md)。本文解决的是
> **「我要改点什么，该动哪个文件」**和**「报错了怎么办」**。

---

## 一、后端现在能做什么

组长分配的 9 项任务与代码的对应关系，以及**当前完成度**：

| # | 任务 | 状态 | 主要文件 |
|---|---|---|---|
| 1 | 数据集加载模块 | ✅ | `backend/data/formats.py`、`registry.py`、`datasets/*.json` |
| 2 | 数据预处理调用流程 | ✅ | `backend/data/encoding.py`、`services/experiment.py` |
| 3 | 算法注册模块 | ⚠️ **团队原有**，后端只补了测试 | `backend/algorithms/registry.py` |
| 4 | 训练和预测接口 | ✅ | `backend/api/main.py`、`services/experiment.py` |
| 5 | WebSocket 通信 | ✅ | `services/runner.py`、`api/main.py` |
| 6 | 实验结果保存和查询 | ✅ | `services/store.py`、`api/main.py` |
| 7 | 错误处理和日志记录 | ✅ | `api/main.py`（错误）、`backend/logging_config.py`（日志） |
| 8 | 后端接口文档 | ✅ | `docs/api.md`、本文 |
| 9 | 后端测试 | ✅ | `tests/`，157 个用例 |

第 3 项要特别说明：`backend/algorithms/registry.py` 是团队原有代码（10 个算法、47 行），
**后端没有改动它**，只是补了 `tests/test_registry.py`。如果组长认为该项应重做，
需要先确认口径。

**当前能力边界**：8 个数据集（分类 5 + 回归 3）、10 个算法、6 个接口 + WebSocket、
统一错误响应、请求日志、实验记录落盘。前端不在范围内。

---

## 二、5 分钟上手

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt      # 跑测试才需要

# 把数据集放到仓库根的 数据集\ 目录（本体不进 Git）
python -m unittest discover              # 期望 157 tests OK
python -m backend.main demo              # 不启服务也能看到完整链路跑通
python -m uvicorn backend.api.main:app --port 8000 --reload
```

打开 <http://127.0.0.1:8000/docs> 可以直接点着测。

> **本机 `python` 可能是 Microsoft Store 的假壳**，会报「找不到指定的路径」。
> 用 `.venv\Scripts\python.exe`。

---

## 三、我要新增一个数据集

**不需要写任何解析代码**，加一份 JSON 就行。

1. 把数据文件放进 `数据集/` 下的任意子目录。
2. 在 `backend/data/datasets/` 新建 `<名字>.json`：

```json
{
  "name": "mydata",
  "display_name": "我的数据集",
  "task_type": "classification",
  "target": "label",
  "source": {
    "path": "分类/mydata.csv",
    "format": "csv",
    "delimiter": ",",
    "has_header": true
  },
  "columns": ["f1", "f2", "label"],
  "dtypes": { "f1": "numeric", "f2": "numeric", "label": "nominal" },
  "drop_columns": [],
  "missing_markers": ["", "?"],
  "decimal_comma": false,
  "recommended_algorithms": ["knn", "decision_tree"],
  "shape": { "n_samples": 200, "n_features": 2 },
  "notes": "备注，会出现在接口二的返回里"
}
```

3. 跑 `python -m unittest discover`。

**会自动帮你处理的**：

| 情况 | 行为 |
|---|---|
| 无表头文件 | 用 `columns` 命名 |
| 列数对不上 | **直接报错**，不静默跑错 |
| 尾部空行 / 恒空尾列 | 自动丢弃 |
| 行宽错位（多一个逗号） | 删掉多余空字段并计数上报 |
| 逗号做小数点 | 设 `decimal_comma: true` |
| `-200` 这类缺失标记 | 写进 `missing_markers`，按数值等价匹配 |
| 目标列有缺失 | 整行剔除并计数，不抛异常 |
| 完全重复的行 | 默认剔除，可设 `drop_duplicates: false` 关闭 |

**支持的格式**：`csv`（任意分隔符）、`arff`、`excel`（`.xls` 需 xlrd、`.xlsx` 需 openpyxl）。

**⚠️ 别照抄 `.names` 文件**。实测 `car.names` 写的是 `v-high`/`5-more`/`v-good`，
而 `car.data` 里实际是 `vhigh`/`5more`/`vgood`。**以数据为准**，照文档写必错。

---

## 四、我要新增一个算法

1. 在 `backend/algorithms/` 新建文件，继承 `BaseModel`：

```python
from .base_model import BaseModel, as_float_matrix, require_y

class MyModel(BaseModel):
    algorithm_name = "my_model"
    task_type = "classification"        # classification/regression/clustering/dimensionality_reduction

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")   # 参数非法必须抛 ValueError
        self.k = k

    def fit(self, X, y=None):
        self._X = as_float_matrix(X)
        self._y = require_y(y, len(self._X))
        return self

    def predict(self, X):
        if not hasattr(self, "_X"):
            raise RuntimeError("call fit before predict")  # 未 fit 必须抛 RuntimeError
        return ...
```

2. 在 `backend/algorithms/registry.py` 的 `ALGORITHM_REGISTRY` 里加一条（显示名、
   `task_type`、`implemented_in_house`、`factory`、`default_params`）。
3. 在 `tests/test_algorithms.py` 补用例。

**双任务算法**（既能分类又能回归）把 `task_type` 写成 `"classification / regression"`，
构造参数用 `task` 切换 —— 参考 `random_forest.py` 和 `gradient_boosting.py`。

**注意 `task_type` 是子串匹配**：`list_algorithms(task_type="classification")` 会同时
命中 `"classification / regression"`，这是有意设计，不是 bug。

**不要碰算法核心**：老师只要求 4~6 个算法、3~4 个自实现，现在已经 10 个了。

---

## 五、我要新增一个接口

```python
# backend/api/main.py
@api_router.get("/my/endpoint", summary="我的接口")     # 自动带 /api 前缀
def my_endpoint(name: str) -> dict[str, Any]:
    return some_service_function(name)                  # 逻辑放 services，别写在路由里
```

三条规则：

1. **业务逻辑放 `backend/services/`**，路由函数只做参数接收和调用。
2. **WebSocket 不要放进 `api_router`**。规范里 WS 是 `/ws/experiments/{id}`，
   **不带 `/api`**；塞进 `APIRouter(prefix="/api")` 会变成 `/api/ws/...`，与规范不符。
   直接挂 `@app.websocket(...)`。
3. **异常直接抛**，统一错误响应由 `@app.exception_handler` 转换，不用手写 try/except。

加完记得在 `tests/test_api.py` 补用例，并更新 `docs/api.md`。

---

## 六、目录速查

```
backend/
├── logging_config.py       日志：级别、控制台 + 轮转文件
├── main.py                 CLI：demo / algorithms
├── algorithms/
│   ├── base_model.py       统一接口：fit / predict / get_params
│   └── registry.py         ★ 算法注册表（加算法改这里）
├── data/
│   ├── formats.py          脏 CSV / ARFF / Excel 读取，行宽修复
│   ├── encoding.py         缺失值、类型推断、one-hot、去重、防泄露填充
│   ├── registry.py         ★ 数据集登记表（加数据集改这里）
│   ├── datasets/*.json     ★ 8 份数据集元数据
│   └── preprocessing.py    划分、分层 K 折、标准化
├── services/
│   ├── experiment.py       ★ 端到端编排（整条链路在这里串起来）
│   ├── evaluation.py       分类 / 回归 / 聚类指标
│   ├── runner.py           线程池 + 进度订阅
│   ├── store.py            实验记录落盘
│   └── train_service.py    单次实验与交叉验证
└── api/main.py             ★ 6 个接口 + CORS + 错误处理 + 请求日志
```

`★` = 最常需要改的文件。

---

## 七、日志

日志写到控制台和 `logs/backend.log`（UTF-8，单文件 2 MB，保留 3 个备份）。

```powershell
# 临时提高详细程度
$env:ML_LOG_LEVEL = 'DEBUG'
# 换日志目录
$env:ML_LOG_DIR = 'D:\mylogs'
# 只输出到控制台
$env:ML_LOG_TO_FILE = '0'
```

在自己的模块里加日志：

```python
from backend.logging_config import get_logger
logger = get_logger(__name__)
logger.info("加载数据集 %s：%d 行", name, rows)
```

**不要用裸 `print`**（例外：`backend/main.py` 是 CLI，它打印给用户看的内容本来就该用
`print`）。原因见第九节的坑 #1 —— print 会因控制台编码把整个测试套件搞挂。

日志里能看到什么：

```
2026-10-13 20:26:04 INFO     backend.api.main    POST /api/experiments -> 202（2.9 ms）
2026-10-13 20:26:04 INFO     backend.services.runner  实验开始 id=20261013-202604-knn-iris-85e6e4
2026-10-13 20:26:04 WARNING  backend.services.runner  实验失败 id=...：ValueError: k must be at least 1
2026-10-13 20:26:04 WARNING  backend.api.main    404 实验不存在 /api/experiments/nope："找不到实验 'nope'"
```

每个响应还带 `X-Process-Time-Ms` 响应头，方便前端或 `curl -i` 看耗时。

**训练失败的堆栈去哪了**：`GET /api/experiments/{id}` 的 `traceback` 字段（仅失败时返回），
同时写进 `experiments/results/<id>.json`。`logs` 字段是完整的进度日志，以失败原因收尾。

---

## 八、常见错误速查

| 报错 | 原因 | 怎么办 |
|---|---|---|
| `DatasetUnavailable: 数据集 'x' 的文件不存在：...` | 数据没放好 | 放到 `数据集/` 下，或设 `ML_DATA_ROOT` |
| `DatasetNotFound: 未知数据集 'x'` | 名字拼错 | 看 `GET /api/datasets` 的返回 |
| `metadata 声明 N 列，实际文件 M 列` | 元数据与数据漂移 | 改 `datasets/*.json` 的 `columns` |
| `ExperimentError: 算法 'x' 用于 regression，与数据集的 classification 任务不匹配` | 算法与数据集任务不符 | 换算法；聚类/降维不受此限制 |
| `读取 .xls 需要 xlrd` | 缺依赖 | `pip install -r requirements.txt` |
| `RuntimeError: ... requires the httpx2 package` | 缺测试依赖 | `pip install -r requirements-dev.txt` |
| `ValidationError` (400) | 请求体字段缺失或越界 | 看 `error.details` |
| `ARFF 缺少 @attribute 声明` | ARFF 文件损坏 | 检查文件头 |
| `ValueError: k must be at least 1` | 算法参数非法 | 这是**正常报错**，实验会标记为 failed |
| `SetNamedSecurityInfoW failed (WinError 5)` | 工作区文件夹权限问题 | 环境问题，不是代码问题；换有写权限的目录或调整会话权限 |

---

## 九、踩过的坑（别再踩）

这些都是**真踩过的**，不是假想。

### 1. emoji + GBK 控制台 = 整个测试套件挂掉

根目录原先有个 `test_registry.py`，用 `print("✅ ...")` 报结果。`unittest discover`
会把它当测试模块导入，✅ 在 GBK 控制台抛 `UnicodeEncodeError`，于是整个套件变成
`Ran 9 tests, FAILED (errors=1)` —— 而真正的问题跟测试毫无关系。

**教训**：测试用断言，不用 print；日志用 `logging`，不用 print。

### 2. 不要在 Git 里提交数据集、`.pyc`、日志

`.gitignore` 已经排除了 `数据集/`、`教案/`、`.vs/`、`logs/`、`experiments/results/`、
`__pycache__/`。曾经有 18 个 `.pyc` 被误提交上去，后来靠 `git rm --cached` 清理。

### 3. `casual + registered == cnt`：标签泄露

`bike_day` 的这两列加起来**恰好等于目标列** `cnt`。当特征用的话，任何模型都能拿到
R²=1.0，指标完全失真。已放进元数据的 `drop_columns`。

**教训**：加数据集前先想想有没有列能直接推出目标。

### 4. `-200` 藏在逗号小数点的列里

`airquality` 用逗号做小数点，缺失值写成 `-200,0` 而不是 `-200`。只比原始文本会漏判，
结果**回归目标里混进了 -200**，取值范围变成 `-200..63.7`。

**教训**：缺失标记比较要做数值等价匹配（现在的 `missing_marker_set` 会同时匹配
`-200`、`-200.0`、`-200,0`）。

### 5. 重复行会让指标虚高

`phishing` 有 **5206/11055 行完全重复（47%）**，iris 3 行，concrete 25 行。
不剔除的话同一行会同时落进训练集和测试集，准确率自然好看。

### 6. 预处理统计量必须只来自训练集

这是**评分点**。编码层故意保留 `np.nan` 不做填充，把填充推迟到 `train_test_split`
之后，就是为了这件事。`tests/test_encoding.py::NoLeakageTest` 有专项用例守着。

### 7. `FastAPI(prefix="/api")` 不报错，但也不生效

`prefix` 不是 `FastAPI` 的合法参数，会被**静默吞掉**（不抛异常），路由实际挂在
`/algorithms`。必须用 `APIRouter(prefix="/api")`。这种「不报错但不对」的问题最难发现。

### 8. Windows 上文件被占用就删不掉

写日志/写结果的 handler 不关，临时目录清理会报 `WinError 32`。
`services/store.py` 用「先写临时文件再 `os.replace`」的原子写法，避免读到写一半的 JSON。

---

## 十、还没做的 / 已知限制

| 项 | 说明 |
|---|---|
| sklearn 对照实验 | 老师「推荐完成目标」要求自实现与 sklearn 对照，`registry` 已预留 `implemented_in_house` 字段，但对照算法未实现 |
| 缺失值删除策略 | 只实现了填充（训练集均值），没有「删列/删行」的选项 |
| 多算法对比接口 | `BACKEND_PLAN` 里的 P2 项，未实现 |
| 实验记录存储 | 用 JSON 文件；数据量大时按 `BACKEND_PLAN` 的建议换 SQLite |
| 时序特征 | `airquality` 的 `Date` 被直接剔除，没派生年/月/星期特征 |
| 前端 | 不在后端范围 |

**待组内确认**（来自 `HANDOFF.md` §10）：最终数据集选择、前端归属、是否补 sklearn 对照、
实验记录存储选型、分支策略、push 由谁执行。

---

## 十一、改完之后的自检

```powershell
python -m unittest discover              # 必须全绿
python -m backend.main demo              # 真实数据集链路能跑通
```

提交信息用 `feat:` / `fix:` / `test:` / `docs:` / `refactor:` / `chore:` 前缀。
