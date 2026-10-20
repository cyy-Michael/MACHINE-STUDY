# 后端开发说明书（BACKEND_PLAN）

> 面向本项目的后端负责人。内容包括：团队已完成什么、必须遵守的接口约定、当前已知缺陷、后端任务清单与验收对照、环境与常用命令、待确认事项。
>
> 最后更新：后端开发启动阶段

---

## 一、项目定位与当前进度

平台目标（见 `项目实施方案_机器学习可视化平台.txt`）：用户在网页上**选数据集 → 选算法 → 调参数 → 点训练 → 看指标和图表**。后端负责"从数据集到结果 JSON"的整条链路，并对外提供 HTTP / WebSocket 接口。

老师方案里对第一阶段的要求是**先打通一条完整链路**：

```
数据集加载 → 数据预处理 → 模型训练 → 模型预测 → 指标计算 → 前端展示
```

当前状态：

| 层 | 目录 | 状态 |
|---|---|---|
| 算法层 | `backend/algorithms/` | ✅ 10 个算法全部自实现完成，测试通过 |
| 数据层 | `backend/data/` | ⚠️ 只有读 CSV、划分、标准化；**缺缺失值/重复值/类别编码** |
| 服务层 | `backend/services/` | ⚠️ 能跑单次实验和分层 K 折；**缺"按数据集名编排"、缺聚类指标、无实验记录** |
| 接口层 | `backend/api/` | ❌ 只有 `GET /api/algorithms`（且路径前缀有 bug），其余 5 个接口未做 |
| 前端 | `frontend/` | ❌ 无代码，仅有页面职责说明 |

**链路目前断在第一环**：没有任何代码能从"选择一个数据集"走到"得到 X 和 y"。这是后端的第一优先级。

---

## 二、团队已完成的内容（后端依赖它们）

### 2.1 算法清单（`backend/algorithms/`）

| 注册名 | 显示名 | 任务类型 | 实现要点 |
|---|---|---|---|
| `linear_regression` | 线性回归 | regression | 正规方程 + 伪逆，偏置折进列向量 |
| `logistic_regression` | 逻辑回归 | classification | 交叉熵梯度下降，OvR 多分类，内部标准化 |
| `knn` | K-近邻 | classification | 欧氏距离 + 多数投票，`predict_proba` = 邻居票数比例 |
| `decision_tree` | 决策树（CART） | classification | 基尼指数，扫描全部特征与候选阈值 |
| `svm` | 支持向量机（线性核） | classification | 合页损失次梯度下降，OvR；概率是决策值 softmax（未校准） |
| `kmeans` | K-平均聚类 | clustering | k-means++ 初始化，inertia/中心位移双收敛判断 |
| `random_forest` | 随机森林 | classification / regression | 自助采样 + 特征子集（sqrt/log2），**用参数 `task` 切换任务** |
| `gaussian_naive_bayes` | 高斯朴素贝叶斯 | classification | 对数概率，`var_smoothing` 防止方差为 0 |
| `pca` | 主成分分析 | dimensionality_reduction | 协方差特征分解，含 `inverse_transform` |
| `gradient_boosting` | 梯度增强 | classification / regression | 平方误差回归树做弱学习器，支持回归/二分类/多分类 softmax，**`task` 切换** |

### 2.2 数据与服务层

- `backend/data/loader.py`：`load_csv()`（返回字符串字典记录）、`list_local_datasets(root="data/local")`。
- `backend/data/preprocessing.py`：`train_test_split()`（固定随机种子）、`k_fold_indices()`（**分层** K 折）、`standardize()`（**只在训练集拟合均值方差**，防数据泄露）。
- `backend/data/sample_datasets.py`：内置西瓜数据集（17 条 2 特征，随代码分发）；`iris` / `diabetes` 走 scikit-learn（可选依赖）。
- `backend/services/evaluation.py`：`classification_metrics()`（accuracy、precision/recall/f1 macro、混淆矩阵、labels）、`regression_metrics()`（mae、mse、rmse、r2）。
- `backend/services/train_service.py`：`run_experiment()`（单次 fit/predict + 指标）、`run_cross_validation()`（每折指标 + mean/std）。
- `backend/main.py`：CLI 演示，`python -m backend.main demo` / `algorithms`。
- `tests/test_algorithms.py`：8 个用例，覆盖 10 个算法、可分离数据精度、交叉验证服务。

### 2.3 组内分工（见 `docs/TEAM_WORK.md`）

| 角色 | 负责目录 |
|---|---|
| 1 负责人/集成 | README、docs、`algorithms/registry.py` |
| 2 聚类/降维 | `kmeans.py`、`pca.py` |
| 3 分类 | `naive_bayes.py`、`random_forest.py` |
| 4 回归/增强 | `gradient_boosting.py`、`services/evaluation.py` |
| 5 数据/后端/前端 | `backend/data/`、`backend/api/`、`frontend/` |

> 注意：老师原方案的分工是「2=分类、3=回归/聚类、4=后端、5=前端」，本项目组做了调整。**前端页面归属需要最终确认**。

---

## 三、必须遵守的接口约定（不要破坏）

1. **统一模型接口**：所有模型继承 `BaseModel`，实现 `fit(X, y)`、`predict(X)`；分类模型提供 `predict_proba`；聚类/降维提供 `transform`。参数非法抛 `ValueError`，未 fit 先 predict 抛 `RuntimeError`。
2. **`task_type`**：类属性，取值 `classification` / `regression` / `clustering` / `dimensionality_reduction`。随机森林和梯度增强是实例属性，通过构造参数 `task` 切换。
3. **`get_params()`** 自动返回构造参数（过滤 `_` 开头），前端参数表单依赖它。
4. **建模只走注册表**：`registry.create_algorithm(name, **params)`、`registry.list_algorithms(task_type)`。不要在 API 里写算法分支。
5. **预处理防泄露**：划分和标准化的统计量只能来自训练集，测试集信息不得参与训练（评分点）。
6. **算法层不读文件、不依赖 HTTP**；API 层不实现算法。

---

## 四、已知缺陷清单（P0：开工先修）

| # | 问题 | 位置 | 说明 / 修法 |
|---|---|---|---|
| 1 | `python -m unittest discover` 失败 | 根目录 `test_registry.py` | 该脚本被 unittest 当测试模块导入，print 的 ✅/❌ 在 GBK 控制台抛 `UnicodeEncodeError`，导致 `Ran 9 tests, FAILED (errors=1)`。改为 `tests/test_registry.py` 并且只保留断言、去掉 emoji；或用 `discover -s tests` |
| 2 | `/api` 前缀未生效，文档路径错误 | `backend/api/main.py` | `FastAPI(prefix="/api")` 中 `prefix` 不是 FastAPI 参数，会被静默忽略。实测 `GET /algorithms` → 200，`GET /api/algorithms` → 404。改用 `APIRouter(prefix="/api")` + `app.include_router(router)` |
| 3 | `requirements.txt` 不完整 | `requirements.txt` | 只有 `numpy`，但服务实际需要 `fastapi`、`uvicorn`；`scikit-learn` 是 iris/diabetes 的可选依赖，建议单独放 `requirements-dev.txt` |
| 4 | 接口文档编码不是 UTF-8 | `docs/api.md` | 当前是 GB2312，VS Code / GitHub 会乱码，转为 UTF-8 |
| 5 | 提交未推送 | git | 本地仅 `master` 一个提交，无 `origin/master`；老师要求仓库可访问 |
| 6 | `.vs/` 未忽略 | `.gitignore` | 补 `.vs/`，别把 IDE 缓存提交上去 |
| 7 | 训练/预测计时混在一起 | `services/train_service.py` | 计时从 `fit` 前开始、`predict` 后结束，字段却叫 `prediction_time_seconds`。需拆成 `train_time_seconds` / `predict_time_seconds` |
| 8 | 结果含不可序列化对象 | `services/train_service.py` | `result["model"] = model`，接口返回前必须剔除 |

---

## 五、后端任务清单

### P1 数据层（最高优先级，其他接口的地基）

- [ ] 数据集登记：落实 `docs/DATASET_CATALOG.md` 的字段，为每个数据集写 `metadata.json`（字段名、类型、标签列、任务类型、缺失值、推荐算法）。
- [ ] 格式适配：ARFF（Phishing Websites）、Excel、以及"脏 CSV"——AirQualityUCI 常见问题为**分号分隔、`-200` 表示缺失、逗号作小数点、末尾空列**。
- [ ] 预处理补齐：缺失值填充/删除、重复行删除、字符串类别编码、无意义编号列剔除、数值化。
- [ ] 构造 `X / y`：按数据集名 + 标签列返回特征矩阵和标签，并给出字段类型、样例、缺失统计、标签分布。
- [ ] **接口一**：`GET /api/datasets` → 名称、任务类型、样本数、特征数、标签列。
- [ ] **接口二**：`GET /api/datasets/{name}` → 字段列表、字段类型、数据样例、缺失值统计、标签分布。

### P2 训练编排层

- [ ] 编排函数：`(dataset, algorithm, params, test_size, seed)` → 加载 → 划分 → 标准化/编码 → `fit`（计时）→ `predict`（计时）→ 指标 → 结果 JSON。
- [ ] 拆分训练时间与预测时间，写入结果。
- [ ] 聚类指标：轮廓系数（silhouette）、Calinski-Harabasz 指数；聚类标签与二维散点数据（如需 PCA 降维到 2 维）。
- [ ] 结果 JSON 剔除 `model` 对象，保证可序列化。
- [ ] 实验记录：参数、指标、时间、时间戳落盘（`experiments/` 下的 JSON，或数据量大时用 SQLite）。
- [ ] 历史查询：列出历史实验、按 id 取回。
- [ ] 多算法对比：同数据集同划分下比较多个算法的指标与耗时。

### P3 接口层

- [ ] 修正 `/api` 前缀（P0-#2），统一响应结构。
- [ ] **接口三**：`GET /api/algorithms?task_type=` → 算法名、类型、是否自实现、默认参数（已基本可用，注意 `task_type` 目前是子串匹配）。
- [ ] **接口四**：`POST /api/experiments` → 创建实验，返回 experiment_id。
- [ ] **接口五**：`GET /api/experiments/{id}` → 状态、指标、预测结果、训练时间、可视化数据。
- [ ] **接口六**：`WS /ws/experiments/{id}` → 训练开始/进度/日志/异常/完成。
- [ ] 训练放后台任务或线程池，**不要阻塞事件循环**（随机森林/梯度增强是纯 Python 循环，训练慢）。
- [ ] CORS：允许前端开发地址（Vue 默认 5173）。
- [ ] 统一错误响应（400/404/500 都是 JSON）、请求体校验、日志。

### P4 测试与文档

- [ ] 数据层测试：划分形状、分层比例、标准化统计量只来自训练集、缺失值/编码、格式适配。
- [ ] 指标测试：混淆矩阵、precision/recall/f1、mae/rmse/r2、聚类指标。
- [ ] 服务测试：`run_experiment` 的指标与计时字段、实验记录读写。
- [ ] 接口测试：FastAPI `TestClient` 覆盖全部 HTTP 接口；WebSocket 用 `TestClient` 的 websocket 连接测试。
- [ ] 补全 `docs/api.md`（UTF-8）：每个接口的请求、响应、错误码、示例。
- [ ] 更新 README 的运行说明与依赖安装。

---

## 六、环境与常用命令

```powershell
# 依赖
pip install -r requirements.txt          # numpy（当前）
pip install fastapi uvicorn              # 接口层需要，务必补进 requirements.txt
pip install scikit-learn                 # 可选：iris / diabetes 与 sklearn 对照

# 测试
python -m unittest discover -s tests -v  # 推荐（只跑 tests/，不受根目录脚本干扰）
python -m unittest discover -v           # 修好 test_registry.py 之后才可用

# CLI 演示
python -m backend.main demo
python -m backend.main algorithms

# 启动接口服务
python -m uvicorn backend.api.main:app --port 8000 --reload
# 文档页 http://127.0.0.1:8000/docs
```

> 注意：本机 `python` 若指向 Microsoft Store 的占位程序会报"找不到指定的路径"。请使用真实解释器或项目内 `.venv\Scripts\python.exe`。

---

## 七、实验结果 JSON 契约（草案，需与前端确认后固定）

```json
{
  "experiment_id": "20260101-120000-knn-iris",
  "status": "finished",
  "dataset": "iris",
  "algorithm": "knn",
  "task_type": "classification",
  "parameters": { "k": 5 },
  "split": { "test_size": 0.2, "random_state": 42 },
  "metrics": {
    "accuracy": 0.96,
    "precision_macro": 0.96,
    "recall_macro": 0.96,
    "f1_macro": 0.96,
    "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]],
    "labels": ["setosa", "versicolor", "virginica"]
  },
  "train_time_seconds": 0.0021,
  "predict_time_seconds": 0.0004,
  "predictions": [0, 1, 2],
  "visualization": {
    "scatter_2d": [[x, y, label]],
    "true_vs_pred": [[true, pred]]
  },
  "error": null
}
```

要点：
- 分类必须有 accuracy / precision / recall / f1 / 混淆矩阵；回归必须有 mae / mse / rmse / r2；聚类建议给轮廓系数、CH 指数、二维散点。
- `model` 对象绝不返回；`predictions` 过长时考虑只返回前 N 条或落盘。
- 字段一旦与前端约定就不要随意改名。

---

## 八、协作规范

- 分支：`main`（稳定）、`dev`（集成）、`feature/backend-*`、`feature/algorithm-*`、`feature/frontend-*`。
- 提交前缀：`feat:`、`fix:`、`test:`、`docs:`、`refactor:`。
- 提交前自检：`python -m unittest discover -s tests` 全绿；不提交 `.venv/`、`.vs/`、`数据集/`、`*.csv`、`*.arff`、实验结果。
- 数据集本体不进仓库，只提交数据字典与获取方式。

---

## 九、待确认事项

1. **数据集文件位置**：`docs/DATASET_CATALOG.md` 中登记的 Iris / Phishing Websites / AirQualityUCI 三个文件当前不在仓库中（`.gitignore` 已忽略 `数据集/`），需要向组长获取。
2. **前端页面归属**：`TEAM_WORK.md` 把前端划给角色 5，需确认实际由谁实现。
3. **自实现 vs sklearn 对照**：老师"推荐完成目标"要求提供对照实验与算法调参/改进（报告第六章），当前 10 个算法全部自实现、无 sklearn 对照。是否补 sklearn 包装算法（`implemented_in_house: false`）需组内决定。
4. **实验记录存储**：JSON 文件足够，还是需要 SQLite。
5. **结果 JSON 契约**：与前端逐字段确认后冻结。
