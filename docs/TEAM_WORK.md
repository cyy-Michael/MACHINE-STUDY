# 五人协作与认领边界

以下按“目录 + 交付物”划分责任。成员可以在自己的 feature 分支工作，合并前必须补测试和说明。

| 角色 | 主要目录 | 初版交付物 |
|---|---|---|
| 1. 负责人/集成 | `README.md`、`docs/`、`backend/algorithms/registry.py` | 架构、接口约定、分支管理、最终联调 |
| 2. 聚类/降维算法 | `backend/algorithms/kmeans.py`、`pca.py`、对应测试 | K-Means、PCA、聚类/降维可视化数据约定 |
| 3. 分类算法 | `backend/algorithms/naive_bayes.py`、`random_forest.py`、对应测试 | 朴素贝叶斯、随机森林分类、分类指标分析 |
| 4. 回归/增强算法 | `backend/algorithms/gradient_boosting.py`、`services/evaluation.py`、对应测试 | 梯度增强、回归流程、回归指标 |
| 5. 数据/后端/前端 | `backend/data/`、`backend/api/`、`frontend/` | 数据集清单、接口、页面和 ECharts；当前骨架阶段先完成接口设计 |

## 当前五个算法的认领方式

- K-平均：角色 2
- 降维 PCA：角色 2
- 朴素贝叶斯：角色 3
- 随机森林：角色 3
- 梯度增强：角色 4
- 算法统一注册、跨模块集成：角色 1
- 数据集格式适配和训练接口：角色 5

如果组员数量或名字不同，只需要修改本表，不需要修改算法代码中的模块边界。

## 每个模块的合并检查

- 能否在全新环境中按 README 运行？
- 是否保留统一的 `fit/predict` 调用方式？
- 是否有至少一个自动化测试？
- 是否把参数写入注册表或数据字典？
- 是否误提交了数据集、个人路径、密钥或生成结果？
