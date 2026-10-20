# ML 平台后端接口文档

- 服务基础地址：`http://127.0.0.1:8000`
- 调试页面：`http://127.0.0.1:8000/docs`（Swagger 自动生成，可在线测试）
- 健康检查：`http://127.0.0.1:8000/health`

> **WebSocket 不带 `/api` 前缀**：`/ws/experiments/{id}`。其余 HTTP 接口都在
> `/api` 下。这是老师方案的约定。
>
> 历史坑：早期版本写成 `FastAPI(title=..., prefix="/api")`，而 `prefix` 不是
> `FastAPI` 的合法参数，会被**静默忽略**（不会报错），导致路由实际挂在
> `/algorithms`。现改为 `APIRouter(prefix="/api")`。所以 WebSocket 必须单独
> 挂在 `app` 上，不能塞进那个 router。

## 跨域（CORS）

允许的前端开发地址：

```
http://localhost:5173      http://127.0.0.1:5173
http://localhost:4173      http://127.0.0.1:4173
```

## 统一错误格式

**所有错误都返回 JSON**，不会出现 HTML 错误页：

```json
{
  "error": {
    "type": "DatasetNotFound",
    "message": "未知数据集 'no_such'；可选：['airquality', 'bike_day', ...]"
  }
}
```

| HTTP | type | 触发场景 |
|---|---|---|
| 400 | `ExperimentError` | 算法与数据集任务不匹配、参数非法 |
| 400 | `ValidationError` | 请求体字段缺失或越界（额外带 `details`） |
| 400 | `ValueError` | 其他参数错误 |
| 404 | `DatasetNotFound` | 数据集登记名不存在 |
| 404 | `ExperimentNotFound` | 实验 id 不存在 |
| 500 | `DatasetFormatError` | 数据文件存在但解析失败 |
| 500 | `InternalError` | 未预期的异常 |
| 503 | `DatasetUnavailable` | 数据文件不在本机（属于环境问题，不是请求问题） |

---

## 接口一：数据集列表

- 请求路径：`/api/datasets`
- 请求方法：`GET`

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| task_type | string | 否 | `classification` 或 `regression`；不传返回全部 |

请求示例：

```
GET http://127.0.0.1:8000/api/datasets
GET http://127.0.0.1:8000/api/datasets?task_type=regression
```

返回示例（200 OK）：

```json
[
  {
    "name": "iris",
    "display_name": "鸢尾花 Iris",
    "task_type": "classification",
    "n_samples": 147,
    "n_features": 4,
    "target": "class",
    "description": "Fisher 鸢尾花数据集：4 个连续形态特征，3 个品种各 50 条。",
    "source_format": "csv"
  },
  {
    "name": "bike_day",
    "display_name": "共享单车日租量 Bike Sharing (day)",
    "task_type": "regression",
    "n_samples": 731,
    "n_features": 11,
    "target": "cnt",
    "source_format": "csv"
  }
]
```

> 该接口只读元数据、不打开数据文件，所以是常数时间。`n_samples` 是**清洗后**
> 的可用样本数（例如 `airquality` 是 8991，不是文件里的 9472 行），
> 与 `GET /api/datasets/{name}` 返回的 `n_samples` 完全一致。

---

## 接口二：数据集详情

- 请求路径：`/api/datasets/{name}`
- 请求方法：`GET`

请求示例：

```
GET http://127.0.0.1:8000/api/datasets/ckd
```

返回示例（200 OK，节选）：

```json
{
  "name": "ckd",
  "display_name": "慢性肾病 Chronic Kidney Disease",
  "task_type": "classification",
  "target": "class",
  "n_samples": 400,
  "n_features": 24,
  "source_format": "arff",
  "fields": [
    { "name": "age", "dtype": "numeric", "missing": 9, "unique_sample": 62,
      "samples": ["48", "7", "62", "51", "60"] },
    { "name": "rbc", "dtype": "nominal", "missing": 152, "unique_sample": 2,
      "samples": ["normal", "abnormal"] }
  ],
  "missing_total": 1012,
  "repaired_rows": 3,
  "n_rows_raw": 400,
  "dropped_target_missing_rows": 0,
  "duplicate_rows": 0,
  "drop_duplicates": true,
  "recommended_algorithms": ["decision_tree", "random_forest", "..."],
  "notes": "缺失标记是 ?，共 1012 个缺失单元格……",
  "label_distribution": { "ckd": 250, "notckd": 150 }
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| fields[].name | string | 列名 |
| fields[].dtype | string | `numeric` / `nominal` / `target` / `inferred` |
| fields[].missing | int | 该列缺失单元格数 |
| fields[].unique_sample | int | 前 200 行里的不同取值个数 |
| fields[].samples | array | 前 5 个非缺失取值样例 |
| missing_total | int | 全表缺失单元格总数 |
| repaired_rows | int | 被自动修复行宽的行数（见下） |
| n_rows_raw | int | 文件原始行数 |
| dropped_target_missing_rows | int | 目标列缺失而被整行剔除的样本数 |
| duplicate_rows | int | 完全重复而被剔除的样本数 |
| drop_duplicates | bool | 该数据集是否启用去重 |
| label_distribution | object | **仅分类任务**返回，各标签样本数 |

### 样本数是怎么从文件行数变成 n_samples 的

以 `phishing` 为例：

```
文件行数                    11055
  − 目标缺失                 0
  − 完全重复的行             5206     ← 47% 的行是重复的
  = n_samples               5849
```

重复行必须剔除：否则同一行会同时落进训练集和测试集，准确率会明显虚高。
各数据集的实际数字：

| 数据集 | 文件行数 | 目标缺失 | 重复行 | n_samples |
| --- | --- | --- | --- | --- |
| iris | 150 | 0 | 3 | 147 |
| car | 1728 | 0 | 0 | 1728 |
| phishing | 11055 | 0 | 5206 | 5849 |
| credit | 690 | 0 | 0 | 690 |
| ckd | 400 | 0 | 0 | 400 |
| bike_day | 731 | 0 | 0 | 731 |
| airquality | 9357 | 366 | 0 | 8991 |
| concrete | 1030 | 0 | 25 | 1005 |

> 去重只删「整行完全相同」的记录。特征相同但标签不同的**冲突样本**不会被删
> （那会丢掉真实信息），但它们会限制模型的准确率上限。实测 `phishing` 有 64 组
> 这样的矛盾记录，`concrete` 有 9 组「配合比相同但强度不同」的实验误差。

> `repaired_rows` 说明：少数原始文件存在「多一个逗号」的行（ckd 有 3 行字段数
> 是 26 而非 25）。读取层会删掉多余的空字段并计数上报，而不是丢弃样本。
> 解析后发现所有取值都落回合法域，所以能确定是格式缺陷。

---

## 接口三：算法列表

- 请求路径：`/api/algorithms`
- 请求方法：`GET`

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| task_type | string | 否 | `classification` / `regression`；不传返回全部 |

请求示例：

```
GET http://127.0.0.1:8000/api/algorithms
GET http://127.0.0.1:8000/api/algorithms?task_type=classification
```

返回示例（200 OK，节选）：

```json
[
  {
    "name": "linear_regression",
    "display_name": "线性回归",
    "task_type": "regression",
    "implemented_in_house": true,
    "default_params": {}
  },
  {
    "name": "knn",
    "display_name": "K-近邻",
    "task_type": "classification",
    "implemented_in_house": true,
    "default_params": { "k": 5 }
  }
]
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| name | string | 算法内部标识名，创建实验时使用 |
| display_name | string | 前端展示用中文名 |
| task_type | string | 任务类型。`random_forest` 与 `gradient_boosting` 是 `"classification / regression"`，靠参数 `task` 切换 |
| implemented_in_house | bool | 是否为本项目手写实现 |
| default_params | object | 算法默认超参，前端参数表单据此渲染 |

共 10 个算法。**`task_type` 是子串匹配**，所以 `classification` 会同时命中
`random_forest` 这类双任务算法。

异常说明：传入非法 `task_type` 返回空数组 `[]`，服务不会崩。

---

## 接口四：创建训练实验

- 请求路径：`/api/experiments`
- 请求方法：`POST`
- 成功状态码：**202 Accepted**（任务已接受，尚未完成）

请求体：

| 字段 | 类型 | 必须 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| dataset | string | 是 | — | 数据集登记名，如 `iris` |
| algorithm | string | 是 | — | 算法注册名，如 `knn` |
| parameters | object | 否 | `{}` | 算法超参 |
| test_size | float | 否 | `0.2` | 测试集比例，需在 (0,1) 开区间 |
| random_state | int | 否 | `42` | 随机种子，保证可复现 |
| standardize | bool | 否 | `true` | 是否标准化（统计量只用训练集） |

请求示例：

```
POST http://127.0.0.1:8000/api/experiments
Content-Type: application/json

{
  "dataset": "iris",
  "algorithm": "knn",
  "parameters": { "k": 5 },
  "test_size": 0.2,
  "random_state": 42
}
```

返回示例（202 Accepted）：

```json
{
  "experiment_id": "20261013-201448-knn-iris-292cbd",
  "status": "pending",
  "poll_url": "/api/experiments/20261013-201448-knn-iris-292cbd",
  "websocket_url": "/ws/experiments/20261013-201448-knn-iris-292cbd"
}
```

> **为什么立即返回**：随机森林和梯度增强是纯 Python 循环，`airquality` 9000 行
> 跑一次梯度增强要几秒。若在事件循环里同步训练，这几秒内整个服务都会卡住，
> 连 `/docs` 都打不开。所以训练放进线程池，用接口五轮询或接口六订阅进度。

错误示例：

```json
// 404 —— 算法与数据集任务不匹配时是 400
{ "error": { "type": "DatasetNotFound", "message": "未知数据集 'nope'；可选：[...]" } }
{ "error": { "type": "ExperimentError", "message": "算法 'linear_regression' 用于 regression，与数据集的 classification 任务不匹配" } }
{ "error": { "type": "ValidationError", "message": "请求参数校验失败", "details": [...] } }
```

---

## 接口五：查询实验结果

- 请求路径：`/api/experiments/{experiment_id}`
- 请求方法：`GET`

请求示例：

```
GET http://127.0.0.1:8000/api/experiments/20261013-201448-knn-iris-292cbd
```

返回示例（200 OK，训练完成后）：

```json
{
  "experiment_id": "20261013-201448-knn-iris-292cbd",
  "status": "finished",
  "progress": 1.0,
  "message": "实验完成",
  "created_at": "2026-10-13T20:14:48",
  "request": { "dataset": "iris", "algorithm": "knn", "parameters": { "k": 5 },
               "test_size": 0.2, "random_state": 42, "standardize": true },
  "logs": [
    { "progress": 0.0, "message": "排队中" },
    { "progress": 0.25, "message": "划分训练集与测试集" },
    { "progress": 0.55, "message": "训练 knn" },
    { "progress": 1.0, "message": "实验完成" }
  ],
  "result": {
    "experiment_id": "20261013-201448-knn-iris-292cbd",
    "status": "finished",
    "dataset": "iris",
    "algorithm": "knn",
    "task_type": "classification",
    "parameters": { "k": 5 },
    "split": { "test_size": 0.2, "random_state": 42,
               "n_train": 120, "n_test": 30, "n_features": 4 },
    "preprocessing": {
      "imputed": true,
      "impute_strategy": "train_mean",
      "standardized": true,
      "encoded_missing": 0,
      "dropped_rows": 0,
      "scaler": { "mean": [5.8225, 3.0542, 3.7325, 1.2075], "scale": [0.8233, 0.4408, 1.7478, 0.7612] }
    },
    "metrics": {
      "accuracy": 0.9,
      "precision_macro": 0.9,
      "recall_macro": 0.9024,
      "f1_macro": 0.8997,
      "confusion_matrix": [[10, 0, 0], [0, 8, 1], [0, 2, 9]],
      "labels": [0, 1, 2],
      "label_names": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    },
    "train_time_seconds": 0.0117,
    "predict_time_seconds": 0.0005,
    "predictions": [2, 1, 1, "..."],
    "predictions_total": 30,
    "predictions_truncated": false,
    "visualization": {
      "scatter_2d": [[-1.2, 0.3, 0], "..."],
      "projection": "pca"
    },
    "error": null,
    "label_names": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    "y_test_sample": [2, 1, 1],
    "created_at": "2026-10-13T20:14:48",
    "saved": true
  },
  "error": null,
  "traceback": null
}
```

顶层字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| status | string | `pending` / `running` / `finished` / `failed` |
| progress | float | 0~1，训练进度 |
| message | string | 当前阶段的中文描述 |
| logs | array | 完整进度日志，每项 `{progress, message}`，**以结果收尾** |
| result | object | 训练成功时的完整结果；失败时为 `null` |
| error | string | 失败原因（成功时为 `null`） |
| traceback | string | **仅失败时返回**的 Python 堆栈 |

> **失败时怎么排错**：`GET /api/experiments/{id}` 的 `traceback` 字段带完整堆栈，
> `logs` 的最后一条会写明失败原因。同一个实验的失败记录也会写进
> `experiments/results/<id>.json`，事后不用翻控制台。
>
> 失败的典型响应：
>
> ```json
> {
>   "experiment_id": "20261013-202729-knn-iris-47e411",
>   "status": "failed",
>   "message": "ValueError: k must be at least 1",
>   "logs": [ "...", { "progress": 0.55, "message": "失败：ValueError: k must be at least 1" } ],
>   "result": null,
>   "error": "ValueError: k must be at least 1",
>   "traceback": "Traceback (most recent call last):\n  File \".../experiment.py\", ...\nValueError: k must be at least 1"
> }
> ```

`status` 取值：`pending`（排队中）、`running`（训练中）、`finished`、`failed`。

### 各任务类型的 metrics 字段

**分类**（`task_type = "classification"`）

`accuracy`、`precision_macro`、`recall_macro`、`f1_macro`、`confusion_matrix`、
`labels`（整数下标）、`label_names`（可读标签名）

**回归**（`task_type = "regression"`）

`mae`、`mse`、`rmse`、`r2`

**聚类**（`task_type = "clustering"`）

`n_clusters`、`cluster_sizes`、`silhouette`（轮廓系数）、
`calinski_harabasz`（CH 指数）、`silhouette_sampled`、`silhouette_sample_size`

> 轮廓系数是 O(n²)。样本超过 5000 时会**按簇分层抽样**计算，并在
> `silhouette_sampled` 里标注。CH 指数在完美分离时数学上是无穷大，
> 但 `inf` 不是合法 JSON，所以此时返回 `null`。

### 计时字段

`train_time_seconds` 与 `predict_time_seconds` **严格分开**：

- `train_time_seconds`：只包含 `fit` 的耗时
- `predict_time_seconds`：只包含 `predict` 的耗时

### 可视化字段

- 分类 / 聚类：`visualization.scatter_2d`，每项是 `[x, y, label]`。
  特征多于 2 维时用 PCA 投影到 2 维（`projection: "pca"`），直接取前两个原始
  特征会掩盖真实结构。最多返回 500 个点。
- 回归：`visualization.true_vs_pred`，每项是 `[真实值, 预测值]`。

---

## 附加：历史实验列表

- 请求路径：`/api/experiments`
- 请求方法：`GET`

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| limit | int | 否 | 默认 50，范围 1~500 |

按时间倒序返回摘要（不含预测值与散点，避免响应过大）。实验记录落盘在
`experiments/results/`（该目录不提交 Git）。

---

## 接口六：训练进度（WebSocket）

- 请求路径：`/ws/experiments/{experiment_id}`（**不带 `/api`**）
- 协议：WebSocket

连接后依次收到的消息类型：

| type | 说明 |
| --- | --- |
| `snapshot` | 连接建立时的当前状态 |
| `progress` | 进度更新，带 `progress`(0~1) 与 `message` |
| `heartbeat` | 训练较久时的保活包（每 20 秒） |
| `result` | 训练成功，带完整 `result` |
| `error` | 训练失败或实验 id 未知 |

消息示例：

```json
{"type": "snapshot", "experiment_id": "20261013-...", "status": "running", "progress": 0.25, "message": "划分训练集与测试集"}
{"type": "progress", "experiment_id": "20261013-...", "status": "running", "progress": 0.55, "message": "训练 knn"}
{"type": "progress", "experiment_id": "20261013-...", "status": "running", "progress": 1.0,  "message": "实验完成"}
{"type": "result",   "experiment_id": "20261013-...", "status": "finished", "result": { "...同接口五的 result..." }}
{"type": "error",    "experiment_id": "20261013-...", "status": "failed", "error": "ValueError: k must be positive"}
```

前端用法（JavaScript）：

```js
const ws = new WebSocket(`ws://127.0.0.1:8000/ws/experiments/${experimentId}`);
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'progress') progressBar.value = msg.progress * 100;
  if (msg.type === 'result') renderCharts(msg.result);
  if (msg.type === 'error') showError(msg.error);
};
```

> 连接时若任务**已经结束**，服务会先发 `snapshot` 再直接发 `result` / `error`
> 然后关闭，所以前端不需要区分「先连还是后连」。

---

## 日志与排查

后端日志同时输出到控制台和 `logs/backend.log`（UTF-8，单文件 2 MB，保留 3 个备份）。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `ML_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `ML_LOG_DIR` | 仓库根的 `logs/` | 日志目录 |
| `ML_LOG_TO_FILE` | `1` | 设为 `0` 只输出到控制台 |

日志长这样（含请求方法、路径、状态码、耗时）：

```
2026-10-13 20:26:04 INFO     backend.api.main    POST /api/experiments -> 202（2.9 ms）
2026-10-13 20:26:04 INFO     backend.services.runner  实验开始 id=20261013-202604-knn-iris-85e6e4
2026-10-13 20:26:04 WARNING  backend.services.runner  实验失败 id=20261013-202604-knn-iris-85e6e4：ValueError: k must be at least 1
2026-10-13 20:26:04 WARNING  backend.api.main    404 实验不存在 /api/experiments/nope："找不到实验 'nope'"
```

排查建议：

1. 先看 `logs/backend.log` 的最后几行，请求级问题那里都有。
2. 训练失败看接口五的 `traceback`，比日志更完整。
3. 想看每次训练的内部步骤，把 `ML_LOG_LEVEL` 设成 `DEBUG`。
4. 每个响应都带 `X-Process-Time-Ms` 响应头，`curl -i` 就能看到后端耗时。

---

## 健康检查

```
GET /health
```

```json
{ "status": "ok", "datasets": 8 }
```

`datasets` 是已登记的数据集数量（只读元数据，不检查文件是否存在）。

---

## 完整链路时序（前端视角）

```
1. GET  /api/datasets                     选数据集
2. GET  /api/datasets/{name}              看字段、标签分布、缺失情况
3. GET  /api/algorithms?task_type=...     按任务类型选算法
4. POST /api/experiments         -> 202   { experiment_id }
5. WS   /ws/experiments/{id}              实时进度 → result
   （或轮询 GET /api/experiments/{id}）
6. 用 result.metrics 画指标、result.visualization 画图
```
