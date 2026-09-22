# API 接入边界（预留）

这一层后续由后端成员接入 FastAPI。建议接口保持以下职责：

- `GET /api/algorithms`：调用 `registry.list_algorithms()` 返回算法名、任务类型和默认参数。
- `GET /api/datasets`：调用 `data.loader.list_local_datasets()` 返回数据集清单。
- `POST /api/experiments`：接收数据集、算法和参数，调用 `services.train_service.run_experiment()`。
- `GET /api/experiments/{id}`：读取实验状态或保存的结果。
- `WS /ws/experiments/{id}`：推送训练状态；算法核心不感知 WebSocket。

建议 API 响应只序列化结果中的 `model` 以外字段，模型对象本身不直接返回给前端。
