# -*- coding: utf-8 -*-
"""HTTP / WebSocket 接口层。

老师要求的 6 个接口（见 ``docs/BACKEND_PLAN.md`` 第五节）：

===================================  ==========================================
``GET  /api/datasets``               数据集列表
``GET  /api/datasets/{name}``        单个数据集的字段、类型、缺失统计、标签分布
``GET  /api/algorithms``             算法列表（支持按 task_type 筛选）
``POST /api/experiments``            创建实验（**立即返回**，训练在线程池里跑）
``GET  /api/experiments/{id}``       查询实验状态与结果
``WS   /ws/experiments/{id}``        训练进度实时推送
===================================  ==========================================

注意 WebSocket 按规范**不带** ``/api`` 前缀，所以它直接挂在 ``app`` 上，
而不是挂在 ``api_router`` 上——机械地把所有路由塞进 ``APIRouter(prefix="/api")``
会把它变成 ``/api/ws/experiments/{id}``，与规范不符。
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.algorithms.registry import list_algorithms
from backend.data.formats import DatasetFormatError
from backend.data.registry import (
    DatasetNotFound,
    DatasetUnavailable,
    dataset_names,
    describe_dataset,
    list_datasets,
)
from backend.services import store
from backend.services.experiment import ExperimentError, algorithm_task_type
from backend.services.runner import TERMINAL_STATUSES, get_runner

# 前端开发服务器默认端口
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

# WebSocket 空闲多久发一次心跳，避免中间代理掐掉长连接
WS_HEARTBEAT_SECONDS = 20.0


def error_payload(error_type: str, message: str, **extra: Any) -> dict[str, Any]:
    """统一错误结构：客户端只需读 error.message 就能拿到人话。"""
    return {"error": {"type": error_type, "message": message, **extra}}


app = FastAPI(
    title="ML Platform API",
    description="机器学习实验与可视化平台后端",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


# ==========================================================================
# 统一错误响应：所有异常都返回 JSON，绝不吐 HTML 错误页
# ==========================================================================

@app.exception_handler(DatasetNotFound)
async def _dataset_not_found(_request, exc: DatasetNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content=error_payload("DatasetNotFound", str(exc)))


@app.exception_handler(DatasetUnavailable)
async def _dataset_unavailable(_request, exc: DatasetUnavailable) -> JSONResponse:
    # 数据文件不在本机属于环境问题，不是请求问题，所以是 503 而不是 400
    return JSONResponse(status_code=503, content=error_payload("DatasetUnavailable", str(exc)))


@app.exception_handler(store.ExperimentNotFound)
async def _experiment_not_found(_request, exc: store.ExperimentNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content=error_payload("ExperimentNotFound", str(exc)))


@app.exception_handler(ExperimentError)
async def _experiment_error(_request, exc: ExperimentError) -> JSONResponse:
    return JSONResponse(status_code=400, content=error_payload("ExperimentError", str(exc)))


@app.exception_handler(DatasetFormatError)
async def _format_error(_request, exc: DatasetFormatError) -> JSONResponse:
    return JSONResponse(status_code=500, content=error_payload("DatasetFormatError", str(exc)))


@app.exception_handler(RequestValidationError)
async def _validation_error(_request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=error_payload(
            "ValidationError", "请求参数校验失败", details=exc.errors()
        ),
    )


@app.exception_handler(ValueError)
async def _value_error(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content=error_payload("ValueError", str(exc)))


@app.exception_handler(Exception)
async def _unhandled(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_payload("InternalError", f"{type(exc).__name__}: {exc}"),
    )


# ==========================================================================
# 接口一、二：数据集
# ==========================================================================

@api_router.get("/datasets", summary="数据集列表")
def get_datasets(
    task_type: Optional[str] = Query(
        default=None, description="按任务类型筛选：classification / regression"
    )
) -> list[dict[str, Any]]:
    return list_datasets(task_type=task_type)


@api_router.get("/datasets/{name}", summary="单个数据集详情")
def get_dataset(name: str) -> dict[str, Any]:
    return describe_dataset(name)


# ==========================================================================
# 接口三：算法
# ==========================================================================

@api_router.get("/algorithms", summary="算法列表")
def get_algorithms(
    task_type: Optional[str] = Query(
        default=None, description="按任务类型筛选：classification / regression"
    )
) -> list[dict[str, Any]]:
    return list_algorithms(task_type=task_type)


# ==========================================================================
# 接口四、五：实验
# ==========================================================================

class ExperimentRequest(BaseModel):
    dataset: str = Field(..., description="数据集登记名，如 iris")
    algorithm: str = Field(..., description="算法注册名，如 knn")
    parameters: dict[str, Any] = Field(default_factory=dict, description="算法超参")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="测试集比例")
    random_state: int = Field(default=42, description="随机种子，保证可复现")
    standardize: bool = Field(default=True, description="是否标准化（统计量只用训练集）")


@api_router.post("/experiments", status_code=202, summary="创建训练实验")
def create_experiment(request: ExperimentRequest) -> dict[str, Any]:
    """创建实验。**立即返回**，训练在线程池里跑。

    返回 202 而不是 200：任务已接受但尚未完成，客户端应轮询接口五或订阅
    WebSocket 获取进度。
    """
    # 先做同步校验，参数错误立刻返回，不浪费一次后台任务
    available = dataset_names()
    if request.dataset not in available:
        raise DatasetNotFound(
            f"未知数据集 {request.dataset!r}；可选：{available}"
        )

    declared = algorithm_task_type(request.algorithm)  # 未知算法会抛 ExperimentError
    dataset_task = next(
        item["task_type"] for item in list_datasets() if item["name"] == request.dataset
    )
    supported = ("classification", "regression")
    if declared not in ("clustering", "dimensionality_reduction"):
        if declared == "classification / regression":
            if dataset_task not in supported:
                raise ExperimentError(
                    f"算法 {request.algorithm!r} 只支持分类或回归，"
                    f"数据集 {request.dataset!r} 是 {dataset_task}"
                )
        elif dataset_task in supported and declared != dataset_task:
            raise ExperimentError(
                f"算法 {request.algorithm!r} 用于 {declared}，"
                f"与数据集 {request.dataset!r} 的 {dataset_task} 任务不匹配"
            )

    runner = get_runner()
    experiment_id = runner.submit(request.model_dump())
    return {
        "experiment_id": experiment_id,
        "status": "pending",
        "poll_url": f"/api/experiments/{experiment_id}",
        "websocket_url": f"/ws/experiments/{experiment_id}",
    }


@api_router.get("/experiments", summary="历史实验列表")
def get_experiments(
    limit: int = Query(default=50, ge=1, le=500, description="最多返回多少条")
) -> dict[str, Any]:
    return {"experiments": store.list_experiments(limit=limit)}


@api_router.get("/experiments/{experiment_id}", summary="查询实验结果")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    snapshot = get_runner().snapshot(experiment_id)
    if snapshot is None:
        raise store.ExperimentNotFound(f"找不到实验 {experiment_id!r}")
    return {
        "experiment_id": snapshot["experiment_id"],
        "status": snapshot["status"],
        "progress": snapshot.get("progress"),
        "message": snapshot.get("message"),
        "created_at": snapshot.get("created_at"),
        "request": snapshot.get("request"),
        "result": snapshot.get("result"),
        "error": snapshot.get("error"),
    }


# ==========================================================================
# 接口六：WebSocket 进度推送（按规范不带 /api 前缀）
# ==========================================================================

@app.websocket("/ws/experiments/{experiment_id}")
async def experiment_progress(websocket: WebSocket, experiment_id: str) -> None:
    await websocket.accept()
    runner = get_runner()

    snapshot = runner.snapshot(experiment_id)
    if snapshot is None:
        await websocket.send_json(
            {"type": "error", "status": "unknown", "error": f"未知实验 {experiment_id!r}"}
        )
        await websocket.close(code=1008)
        return

    await websocket.send_json(
        {
            "type": "snapshot",
            "experiment_id": experiment_id,
            "status": snapshot["status"],
            "progress": snapshot.get("progress"),
            "message": snapshot.get("message"),
        }
    )

    # 任务已经结束：直接把结果推完就关闭，不让客户端白等
    if snapshot["status"] in TERMINAL_STATUSES:
        await _send_terminal(websocket, snapshot)
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue = runner.subscribe(experiment_id, loop)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=WS_HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                # 训练可能很久（梯度增强是纯 Python 循环），定期发心跳保活
                await websocket.send_json({"type": "heartbeat", "status": "running"})
                continue
            await websocket.send_json(event)
            if event.get("status") in TERMINAL_STATUSES:
                break
    except WebSocketDisconnect:
        return
    finally:
        runner.unsubscribe(experiment_id, queue)
        try:
            await websocket.close()
        except RuntimeError:
            # 客户端已经断开时 close 会抛 RuntimeError，忽略
            pass


async def _send_terminal(websocket: WebSocket, snapshot: dict[str, Any]) -> None:
    if snapshot["status"] == "finished":
        await websocket.send_json(
            {"type": "result", "status": "finished", "result": snapshot.get("result")}
        )
    else:
        await websocket.send_json(
            {"type": "error", "status": "failed", "error": snapshot.get("error")}
        )


# ==========================================================================
# 装配
# ==========================================================================

app.include_router(api_router)


@app.get("/health", summary="健康检查")
def health() -> dict[str, Any]:
    return {"status": "ok", "datasets": len(dataset_names())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=True)
