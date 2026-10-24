# 交接摘要（HANDOFF）

> **给接手的新会话 AI 助手**：你不是从零开始。这是一个已经跑起来的课程项目，本文件把"你必须知道的前置信息"压缩成一份文档。**先读完这份，再动代码。**
>
> 来源：上一个会话（工作区 `D:\MACHINE-STUDY-main (1)`，会话号 `session-889d474a-dc71-4da1-bef6-28a89c9413f5`）整理。
> 配套文档：`docs/BACKEND_PLAN.md`（后端开发说明书，含算法逐条说明、8 项缺陷的详细定位、任务清单、环境命令）。

---

## 0. 用户是谁、要做什么

- **用户身份**：机器学习课程大作业小组的**后端负责人**（组内角色 5：数据层 + 后端 + 前端接口）。对项目细节了解不多，需要边讲解边推进；中文交流。
- **项目**：机器学习实验与可视化平台。目标形态是"网页上选数据集 → 选算法 → 调参数 → 点训练 → 看指标和图表"，技术栈定为 Vue 3 + Element Plus + ECharts（前端）+ FastAPI（后端）。
- **老师方案的原文**在仓库根目录：`项目实施方案_机器学习可视化平台.txt`（1045 行），关键约束：
  1. **第一阶段最重要的是完成一条完整链路**：数据集加载 → 数据预处理 → 模型训练 → 模型预测 → 指标计算 → 前端展示。只要这条链路跑通，其余都是扩展。
  2. 后端要提供 **6 个接口**（见 §6）+ WebSocket 训练进度。
  3. 指标要求：分类必须有 accuracy / precision / recall / f1 / 混淆矩阵；回归必须有 mae / mse / rmse / r²；聚类建议轮廓系数、Calinski-Harabasz 指数。
  4. **数据集本体不进仓库**；代码要能在另一台电脑按 README 跑起来。
  5. 老师建议的项目目录、分工、14 天进度、报告目录，都在那份 txt 里（第九节接口、第十节目录、第十一节分工、第十五节报告目录、第十八节检查清单）。

---

## 1. 当前工作区（`D:\ML-Backend`）里有什么

**这个目录本身就是仓库根**（不是仓库的子目录），git 已 clone 好，当前分支 `houduanxiugai`。

```
D:\ML-Backend\
├── .git\                     ← 完整仓库，含三个远程分支引用；已配好代理
├── backend\                  ← 算法层/数据层/服务层/接口层
│   └── __pycache__\          ← ⚠️ 从远程分支带下来的垃圾（远程提交过 .pyc），要清
├── docs\                     ← ARCHITECTURE.md / DATASET_CATALOG.md / TEAM_WORK.md / api.md
│   └── api.md                ← ⚠️ GB2312 编码，需要转 UTF-8
├── experiments\              ← 只有 .gitkeep，实验记录功能未实现
├── frontend\README.md        ← 只有边界说明，无代码
├── tests\test_algorithms.py  ← 8 个用例，全部通过
├── 数据集\                    ← 真实数据集（见 §7），已被 .gitignore 排除
│   ├── 分类\  回归\  read me .txt
├── 教案\机器学习理论与实践教案.pdf   ← 老师的教案原件（363 KB）
├── .gitignore  README.md  requirements.txt  test_registry.py
└── 项目实施方案_机器学习可视化平台.txt
```

---

## 2. 仓库与 git 现状

| 项目 | 内容 |
|---|---|
| 远程 | `https://github.com/cyy-Michael/MACHINE-STUDY` —— **public**，描述"机器学习实践课程用" |
| 默认分支 | `main`，且 **protected = true**，不能直接 push，必须走 PR（团队已合并过 PR #1） |
| 分支清单 | `main` → `4ae5f57`（cyy-Michael）；`houduanxiugai`（后端修改）→ `9303a6e`（**fylx06 = 用户本人**，2858812466@qq.com）；`feature/algorithm-supervised-five` → `e018c47`（wucan1） |
| 提交历史 | `4608616` Initial commit → `a2000cc` feat: add architecture and algorithms → `b6d1b92` chore: merge remote history → `4ae5f57` Merge PR #1（算法合并进 main）→ `9303a6e` Add files via upload（用户通过网页上传，把 `backend/api/main.py`、`docs/api.md`、`test_registry.py` 放到自己的分支） |
| 组员账号 | cyy-Michael（组长/仓库所有者）、wucan1、fylx06（用户）；另有一个 fork `123sjbj123/MACHINE-STUDY` |

**git 联网配置（本机唯一可用方式，必须保留）**：

```powershell
git config --global http.proxy http://127.0.0.1:10090
git config --global http.sslBackend openssl
```

- 这台机器**直连 `github.com:443` 会被挡**（21 秒超时）。
- 本机有 DSH（DeepSeek Harness）自带的本地代理 `127.0.0.1:10090`，Windows 系统代理已指向它；它是明文隧道（`CONNECT` 返回 200，不做 TLS 中间人）。
- **必须同时设 `http.sslBackend=openssl`**：Windows 默认的 schannel 后端走这个代理会报 `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`，换 openssl 立刻通。（实测）

**历史教训**：旧工作区里有一个用户自己 `git init` 出来的**孤立提交**（`1ec25f5`，无父提交），与远程历史无关，`git push` 会被拒。**不要在新工作区重复这种操作**——永远 clone，不要 init。

---

## 3. 环境（本机有三个 python，只有一个假货）

| 解释器 | 版本 | 用途 |
|---|---|---|
| `C:\Users\fuyunliuxv\WindowsApps\python.exe`（PATH 里的 `python`） | — | ❌ **假壳**，报"找不到指定的路径"，永远别用 |
| `C:\Users\fuyunliuxv\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe` | 3.12.14 | ✅ 自带 **numpy 2.3.5**、pandas。**跑算法测试最方便**（无 fastapi） |
| `C:\Users\fuyunliuxv\AppData\Local\Python\pythoncore-3.14-64\python.exe` | 3.14.3 | ✅ 干净解释器，用它建项目自己的 `.venv`；3.14 标准库内置 `compression.zstd` |

当前项目里**还没有 `.venv`**。装依赖要走代理：

```powershell
& 'C:\Users\fuyunliuxv\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --proxy http://127.0.0.1:10090 numpy fastapi "uvicorn[standard]"
.\.venv\Scripts\python.exe -m pip install --proxy http://127.0.0.1:10090 scikit-learn   # 可选：iris/diabetes + sklearn 对照
```

常用命令：

```powershell
# 测试（注意 -s tests：不加会因为根目录的 test_registry.py 崩掉，见 §5 P0-1）
& 'C:\Users\fuyunliuxv\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v

# CLI 演示
& <python> -m backend.main algorithms
& <python> -m backend.main demo

# 起接口服务（需要 fastapi/uvicorn）
& <python> -m uvicorn backend.api.main:app --port 8000
# 文档页 http://127.0.0.1:8000/docs
```

---

## 4. 团队已经做完了什么（后端依赖它们，不要重写）

### 4.1 算法层：10 个算法全部自实现完成

| 注册名 | 显示名 | 任务类型 | 实现要点 |
|---|---|---|---|
| `linear_regression` | 线性回归 | regression | 正规方程 + 伪逆，偏置折进列向量 |
| `logistic_regression` | 逻辑回归 | classification | 交叉熵梯度下降，OvR 多分类，内部标准化 |
| `knn` | K-近邻 | classification | 欧氏距离 + 多数投票，`predict_proba` = 邻居票数比例 |
| `decision_tree` | 决策树（CART） | classification | 基尼指数，扫描全部特征与候选阈值中点 |
| `svm` | 支持向量机（线性核） | classification | 合页损失次梯度下降，OvR；概率是决策值 softmax（未校准） |
| `kmeans` | K-平均聚类 | clustering | k-means++ 初始化，inertia + 中心位移双收敛判断 |
| `random_forest` | 随机森林 | classification / regression | 自助采样 + 特征子集（sqrt/log2），用构造参数 `task` 切换任务 |
| `gaussian_naive_bayes` | 高斯朴素贝叶斯 | classification | 对数概率，`var_smoothing` 防方差为 0 |
| `pca` | 主成分分析 | dimensionality_reduction | 协方差特征分解，含 `inverse_transform` |
| `gradient_boosting` | 梯度增强 | classification / regression | 平方误差回归树做弱学习器，支持回归/二分类/多分类 softmax，`task` 切换 |

> 老师只要求 4~6 个算法、其中 3~4 个自实现 —— **已超额完成**，不要动算法核心。

### 4.2 数据层 / 服务层 / 测试

- `backend/data/loader.py`：`load_csv()`（返回字符串字典记录）、`list_local_datasets(root="data/local")`（**默认目录不存在，返回空**）
- `backend/data/preprocessing.py`：`train_test_split()`（固定种子）、`k_fold_indices()`（**分层** K 折）、`standardize()`（**只在训练集拟合均值方差**，防数据泄露 —— 这是评分点）
- `backend/data/sample_datasets.py`：西瓜数据集内置在代码里（17 条 2 特征）；`iris`/`diabetes` 走 scikit-learn（可选依赖）
- `backend/services/evaluation.py`：`classification_metrics()`（accuracy、precision/recall/f1 macro、混淆矩阵、labels）、`regression_metrics()`（mae、mse、rmse、r2）。**没有聚类指标**
- `backend/services/train_service.py`：`run_experiment()`、`run_cross_validation()`
- `backend/main.py`：CLI 演示（`demo` / `algorithms`）
- `tests/test_algorithms.py`：8 个用例，覆盖 10 个算法 + 可分离数据精度 + 交叉验证服务 → **全部通过（0.3 秒）**

### 4.3 组内分工（`docs/TEAM_WORK.md`）

角色 1=负责人/集成（registry、docs）、角色 2=聚类/降维（kmeans、pca）、角色 3=分类（naive_bayes、random_forest）、角色 4=回归/增强（gradient_boosting、evaluation）、**角色 5=数据/后端/前端（就是用户）**。
注意：老师原方案的分工与你们组不同；**前端页面归属需要最终确认**。

---

## 5. 后端的现状：8 项已知缺陷（P0，开工先修）

| # | 问题 | 位置 | 说明 / 修法 |
|---|---|---|---|
| 1 | `python -m unittest discover`（不加 `-s tests`）**失败** | 根目录 `test_registry.py` | 该脚本被 unittest 当测试模块导入，print 的 ✅/❌ 在 GBK 控制台抛 `UnicodeEncodeError` → `Ran 9 tests, FAILED (errors=1)`。应挪到 `tests/` 并只保留断言、去掉 emoji |
| 2 | **`/api` 前缀没生效，接口文档路径是错的** | `backend/api/main.py` | `FastAPI(title=..., prefix="/api")` 中 `prefix` 不是 FastAPI 参数，被静默忽略。实测 `GET /algorithms` → **200**，`GET /api/algorithms` → **404**。必须改成 `APIRouter(prefix="/api")` + `app.include_router(...)` |
| 3 | `requirements.txt` 不完整 | 根目录 | 目前只有 `numpy`；缺 `fastapi`、`uvicorn`；`scikit-learn` 建议放 `requirements-dev.txt` |
| 4 | `docs/api.md` 编码不是 UTF-8 | `docs/api.md` | 现状是 **GB2312**，VS Code/GitHub 显示乱码，需转 UTF-8 |
| 5 | 远程分支里混入 `__pycache__/*.pyc` | `backend/**/__pycache__` | 17 个编译缓存文件被提交（网页上传时带上去的）。需要 `git rm -r --cached` 清理 + 提交 |
| 6 | `.vs/` 未被忽略 | `.gitignore` | 补 `.vs/` |
| 7 | **训练时间与预测时间混在一起** | `services/train_service.py` | 计时从 `fit` 前开始、`predict` 后结束，字段却叫 `prediction_time_seconds`。老师要求两者分开 → 拆成 `train_time_seconds` / `predict_time_seconds` |
| 8 | 返回结果含不可序列化对象 | `services/train_service.py` | `result["model"] = model`，接口返回前必须剔除 |

---

## 6. 后端待办清单（对齐老师 6 个接口）

### 接口对照

| 接口 | 现状 | 依赖 |
|---|---|---|
| 一 `GET /api/datasets` | ❌ | 数据登记表 + 文件适配 |
| 二 `GET /api/datasets/{name}` | ❌ | 字段类型 / 样例 / 缺失统计 / 标签分布 |
| 三 `GET /api/algorithms` | ⚠️ 有，但路径前缀错 | 已完成，只需修前缀 |
| 四 `POST /api/experiments` | ❌ | 编排函数 + 后台任务 |
| 五 `GET /api/experiments/{id}` | ❌ | 结果 JSON + 实验记录落盘 |
| 六 `WS /ws/experiments/{id}` | ❌ | 进度回调 |
| 附 | ❌ | CORS（前端 5173）、统一错误响应、请求校验、日志 |

### P1 数据层（最高优先级，其他接口的地基）

- [ ] 为最终选定的数据集写 `metadata.json`（字段名、类型、标签列、任务类型、缺失值、推荐算法）
- [ ] 格式适配：**ARFF**（Phishing、Chronic Kidney）、**Excel**（.xls/.xlsx）、**脏 CSV**（见 §7 陷阱）
- [ ] 预处理补齐：**缺失值**（填充/删除）、**重复行**、**字符串类别编码**、剔除无意义编号列、数值化 —— 老师的"第三步：统一预处理"要求，现在**完全没有**
- [ ] 构造 `X / y`：按数据集名 + 标签列返回特征矩阵和标签
- [ ] 接口一、接口二

### P2 训练编排层

- [ ] 编排函数：`(dataset, algorithm, params, test_size, seed)` → 加载 → 划分 → 编码/标准化 → `fit`（计时）→ `predict`（计时）→ 指标 → 结果 JSON
- [ ] 聚类指标：轮廓系数、Calinski-Harabasz 指数、二维散点数据（可用 PCA 降到 2 维）
- [ ] 结果 JSON 剔除 `model`
- [ ] 实验记录落盘（`experiments/` 下 JSON，或 SQLite）+ 历史查询 + 多算法对比

### P3 接口层

- [ ] 修 `/api` 前缀；接口四、五、六
- [ ] 训练放**后台任务/线程池**（随机森林、梯度增强是纯 Python 循环，会阻塞事件循环）
- [ ] CORS、统一错误响应（400/404/500 都是 JSON）、请求体校验、日志

### P4 测试与文档

- [ ] 测试：数据层（形状、分层比例、标准化只用训练集、缺失值/编码、格式适配）、指标、`run_experiment`、API（`TestClient`）、WebSocket
- [ ] 补全 `docs/api.md`（UTF-8）
- [ ] 更新 README 运行说明与依赖

---

## 7. 数据集：真实清单与陷阱

`D:\ML-Backend\数据集\` 下分 `分类\` 和 `回归\` 两个子目录，既有解压好的数据也有原始 zip。`read me .txt` 说明：每个子文件夹命名为 `数据集1` / `数据集2 - 副本X`，格式有 XLSX / DOC / TXT 等。

### 分类

| 数据集 | 路径 | 规模 | 注意 |
|---|---|---|---|
| **Iris** | `分类\数据集2 - 副本 (4)\iris.data` | 150×4，4.4 KB | 无表头，标签在第 5 列（字符串）；同目录有 `iris.names` |
| **Phishing Websites** | `分类\数据集2 - 副本 (6)\Training Dataset.arff` | 782 KB | **ARFF 格式**，需要解析器；特征说明在同目录 `Phishing Websites Features.docx` |
| Adult | `分类\数据集2\adult.data` + `adult.test` | 3.8 MB / 1.9 MB | 有 `adult.names` 数据字典；含字符串列与空格分隔；训练/测试已分文件 |
| Car Evaluation | `分类\数据集2 - 副本\car.data` | 50 KB | **全部是类别特征** → 必须编码；有 `car.names` |
| Chronic Kidney Disease | `分类\数据集2 - 副本 (2)\Chronic_Kidney_Disease\*.arff` | 43 KB | **ARFF**，含缺失值 |
| Credit Approval | `分类\数据集2 - 副本 (3)\crx.data` | 31 KB | **`?` 表示缺失值**、含字符串列；有 `crx.names` |
| Mushroom | `分类\数据集2 - 副本 (5)\agaricus-lepiota.data` | 365 KB | 全类别特征，含缺失（`?`） |

### 回归

| 数据集 | 路径 | 规模 | 注意 |
|---|---|---|---|
| **Air Quality (UCI)** | `回归\数据集1\AirQualityUCI.csv`（另有 .xlsx） | 767 KB | **头号陷阱**：分号分隔、`-200` 表示缺失、逗号作小数点、末尾有空列 |
| Bike Sharing | `回归\数据集1 - 副本\day.csv` / `hour.csv` | 56 KB / 1.1 MB | 有 `Readme.txt`；day/hour 两个粒度 |
| Communities & Crime | `回归\数据集1 - 副本 (2)\communities.data` | 1 MB | 有 `communities.names`；缺失值多 |
| Concrete Strength | `回归\数据集1 - 副本 (3)\Concrete_Data.xls` | 122 KB | **Excel**，需要读取器 |
| Liver Disorders (bupa) | `回归\数据集1 - 副本 (4)\bupa.data` | 7 KB | 有 `bupa.names` |

> `docs/DATASET_CATALOG.md` 里只登记了 Iris / Phishing / Air Quality 三个，实际可用数据远多于此。**必须先和用户确认最终选用哪 2~3 个**，再写适配代码。

`.gitignore` 已排除：`数据集/`、`data/raw/`、`data/local/`、`experiments/results/`、`*.csv`、`*.xlsx`、`*.xls`、`*.arff`、`*.zip` —— 所以数据不会被误提交，但**也别在测试里依赖仓库内的 csv**（换成代码内置样例）。

---

## 8. 实验结果 JSON 契约（草案，需与前端确认后冻结）

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
    "confusion_matrix": [[10,0,0],[0,9,1],[0,0,10]],
    "labels": ["setosa","versicolor","virginica"]
  },
  "train_time_seconds": 0.0021,
  "predict_time_seconds": 0.0004,
  "predictions": [0, 1, 2],
  "visualization": { "scatter_2d": [[0.2, 0.1, 0]], "true_vs_pred": [[1.0, 0.98]] },
  "error": null
}
```

要点：`model` 对象绝不返回；`predictions` 过长时只返回前 N 条或落盘；字段一旦与前端约定好就不要改名。

---

## 9. 上一会话踩过的坑（别再踩）

1. **中途给工作区文件夹改名** → 会话工具链彻底断掉（命令报 `ENOENT ... realpath`、写文件被拒）。上次靠建目录链接（junction）才救回来。**定好名字就别动。**
2. **工作区子目录可能写不进去**：沙箱只给工作区根目录打「低完整性」标记，事先解压出来的子目录没有 → 沙箱命令写不进（DSH 的文件工具能写）。**新会话第一件事就是验证写权限**（建临时文件再删）。
3. **PATH 里的 `python` 是 Store 假壳**，必须用 §3 里的真实解释器。
4. **git 直连 GitHub 不通**，必须 `http.proxy` + **`http.sslBackend=openssl`**（少了后者必失败）。
5. **不要提交** `__pycache__/`、`.venv/`、`.vs/`、`数据集/`、`*.csv/*.arff/*.xlsx`、实验结果。

---

## 10. 待确认事项（要主动问用户或组员）

1. **最终选用哪 2~3 个数据集**？（老师要求 2~3 个；现在有 11 个可选）
2. **前端页面谁做**？`TEAM_WORK.md` 把前端划给角色 5（用户），需确认，否则工作量要重分。
3. **要不要加 scikit-learn 对照算法**（`implemented_in_house: false`）？老师"推荐完成目标"要求自实现与 sklearn 对照，报告第六章也要"算法改进和调参"。registry 里已预留该字段。
4. **实验记录用 JSON 还是 SQLite**？
5. **分支策略**：继续在 `houduanxiugai` 上开发，还是从 `main` 新开 `feature/backend-api`（推荐后者，便于走 PR）。
6. **push 由谁做**？`git push` 需要账号凭据；上一会话只做过 `fetch`。新工作区的 git 已配好代理，具备 push 的网络条件。

---

## 11. 新会话第一步（建议按此顺序）

1. 读 `docs/BACKEND_PLAN.md`（同目录，含算法逐条说明与缺陷定位）。
2. **验证沙箱写权限**：在 `backend\`、`tests\` 各建一个临时文件再删掉。
3. **跑基线**：用 §3 的 3.12 解释器执行 `-m unittest discover -s tests`，期望 **8/8 通过**。
4. 确认 git：`git branch -a`（应有三个远程分支）、`git status`（应干净，除未跟踪项）。
5. **动代码前先问用户**：从哪一项开始（建议 P0 → P1 数据层）；改动放哪个分支；push 由谁执行。

---

## 12. 需要细节时：完整对话记录在哪

上一会话的完整事件日志（**纯本机，DSH 不上传**）：

```
C:\Users\fuyunliuxv\.dsh\sessions\--D-MACHINE-STUDY-main~0020~00281~0029--\
  session-889d474a-dc71-4da1-bef6-28a89c9413f5\session.v4.jsonl.zstd
```

- zstd 压缩的 JSONL，解压后 **3.02 MB / 715 条事件**（含 `user/message` 12、`assistant/message` 92、`tool/call` 138、`tool/result` 137）。
- 解压方式：Python 3.14 自带的 `compression.zstd`（`zstd.decompress(open(p,'rb').read())`）。
- 目录命名规则是路径转义：空格→`~0020~`、`(`→`~0028~`、`)`→`~0029~`；新工作区对应 `--D-ML-Backend--`。
