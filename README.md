# MACHINE-STUDY

# 机器学习实验与可视化平台（初版）
# 机器学习实验与可视化平台（初版）

这是一个面向课程大作业的机器学习实验平台骨架。当前版本先完成可独立测试的算法核心和后端分层，为后续接入 FastAPI 后端、Vue 前端和 ECharts 可视化保留明确边界。

## 当前已完成

- 自实现 K-平均聚类：`backend/algorithms/kmeans.py`
- 自实现随机森林：`backend/algorithms/random_forest.py`
- 自实现高斯朴素贝叶斯：`backend/algorithms/naive_bayes.py`
- 自实现 PCA 降维：`backend/algorithms/pca.py`
- 自实现梯度增强：`backend/algorithms/gradient_boosting.py`
- 统一算法注册表：`backend/algorithms/registry.py`
- 数据读取、训练/测试划分、标准化和评价指标基础模块
- CLI 演示和单元测试

## 目录职责

```text
backend/
├── algorithms/       算法实现；每个模型遵循 fit/predict
├── api/               预留 HTTP/WebSocket 接口层
├── data/              数据集读取、数据字典和预处理
├── services/          训练编排、指标计算和实验记录
└── main.py            本地 CLI 演示入口
frontend/              预留 Vue 页面与可视化代码
tests/                 算法和服务测试
docs/                  架构、分工和协作说明
experiments/           实验配置与结果（结果不提交）
数据集/                 本地数据集，仅用于本机开发，不提交 Git
```

详细分层说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，具体认领方式见 [docs/TEAM_WORK.md](docs/TEAM_WORK.md)。

## 环境与运行

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -v
python -m backend.main demo
python -m backend.main algorithms
```

当前算法核心只依赖 NumPy。后续增加 FastAPI、Vue 或 ECharts 时，应分别更新后端/前端依赖文件，不要把前端依赖混入 Python 环境。

## 数据集规则

方案文档要求数据集本体不要上传到 GitHub。本地现有数据集位于 `数据集/`，已经由 `.gitignore` 忽略。接入数据层时，建议按以下方式整理：

```text
data/local/<dataset-name>/raw/       原始文件
data/local/<dataset-name>/metadata.json  字段、标签和任务说明
```

提交仓库时只提交数据字典、下载地址、文件名和必要的样例，不提交数据集压缩包或原始数据。

## 统一模型约定

监督学习模型：

```python
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

聚类/降维模型：

```python
model.fit(X_train)
labels_or_components = model.predict(X_test)
```

分类模型提供 `predict_proba`；PCA 和 K-Means 提供 `transform`。新增算法时，先在 `registry.py` 注册，再补充 `tests/` 测试，后端无需为每个算法写一套分支。

## 当前边界

本次初版不实现具体的 Vue 页面、FastAPI 路由、WebSocket 和数据库。它们分别在 `frontend/`、`backend/api/`、`backend/services/` 和 `experiments/` 约定了接入位置，后续成员可按分工并行开发。

## Git 协作

- `main`：稳定版本。
- `dev`：集成分支。
- `feature/algorithm-*`：算法分支。
- `feature/backend-*`：后端分支。
- `feature/frontend-*`：前端分支。

提交信息使用 `feat:`、`fix:`、`test:`、`docs:`、`refactor:` 等前缀。每个成员提交自己的模块测试和说明。
