"""FastAPI application for the ML experiment platform.

Endpoints:
  GET  /api/algorithms
  GET  /api/datasets
  GET  /api/datasets/{name}
  POST /api/experiments
  GET  /api/experiments/{id}
  WS   /ws/experiments/{id}
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.algorithms.registry import list_algorithms
from backend.data.dataset_loader import DATA_LOCAL, dataset_preview, load_dataset
from backend.data.loader import list_local_datasets
from backend.data.preprocessing import train_test_split
from backend.services.evaluation import classification_metrics, regression_metrics
from backend.services.train_service import run_experiment

app = FastAPI(title="ML Experiment Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory experiment store
# ---------------------------------------------------------------------------

class ExperimentRequest(BaseModel):
    dataset: str
    algorithm: str
    task_type: str
    params: dict[str, Any] = {}
    test_size: float = 0.2
    random_state: int = 42


class ExperimentRecord:
    def __init__(self, exp_id: str, request: ExperimentRequest):
        self.id = exp_id
        self.request = request
        self.status = "pending"  # pending | running | done | error
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.logs: list[str] = []
        self._progress = 0
        self._subscribers: set[asyncio.Queue] = set()

    @property
    def progress(self) -> int:
        return self._progress

    def set_progress(self, value: int) -> None:
        self._progress = max(0, min(100, int(value)))
        self._broadcast({"type": "progress", "percent": self._progress})

    def add_log(self, message: str) -> None:
        self.logs.append(message)
        self._broadcast({"type": "log", "message": message})

    def subscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self, payload: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


EXPERIMENTS: dict[str, ExperimentRecord] = {}


def run_training(record: ExperimentRecord) -> None:
    """Run the training in a background thread and update the record."""
    try:
        record.status = "running"
        record.add_log(f"加载数据集: {record.request.dataset}")
        record.set_progress(10)

        X, y, _ = load_dataset(record.request.dataset)
        record.add_log(f"数据集形状: X={X.shape}")
        record.set_progress(25)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=record.request.test_size, random_state=record.request.random_state
        )
        record.add_log(f"训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")
        record.set_progress(40)

        record.add_log(f"训练算法: {record.request.algorithm}")
        record.set_progress(60)

        result = run_experiment(
            record.request.algorithm,
            X_train, X_test, y_train, y_test,
            **record.request.params,
        )
        record.set_progress(90)

        # drop the non-serializable model object before storing
        result.pop("model", None)
        record.result = result
        record.status = "done"
        record.set_progress(100)
        record.add_log("训练完成")
        record._broadcast({"type": "done"})
    except Exception as exc:  # noqa: BLE001
        record.status = "error"
        record.error = str(exc)
        record.add_log(f"错误: {exc}")
        record._broadcast({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/algorithms")
def get_algorithms(task_type: str | None = None):
    return list_algorithms(task_type)


@app.get("/api/datasets")
def get_datasets():
    # prefer local files, fall back to nothing
    datasets = list_local_datasets(DATA_LOCAL)
    # attach task_type from the loader metadata when possible
    enriched = []
    for ds in datasets:
        try:
            _, _, meta = load_dataset(ds["name"])
            enriched.append({**ds, "task_type": meta["task_type"]})
        except Exception:
            enriched.append({**ds, "task_type": "unknown"})
    return enriched


@app.get("/api/datasets/{name}")
def get_dataset_detail(name: str):
    try:
        return dataset_preview(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"dataset '{name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/experiments")
def create_experiment(req: ExperimentRequest):
    exp_id = str(uuid.uuid4())[:8]
    record = ExperimentRecord(exp_id, req)
    EXPERIMENTS[exp_id] = record
    thread = threading.Thread(target=run_training, args=(record,), daemon=True)
    thread.start()
    return {"experiment_id": exp_id}


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str):
    record = EXPERIMENTS.get(exp_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"experiment '{exp_id}' not found")
    return {
        "id": record.id,
        "status": record.status,
        "result": record.result,
        "error": record.error,
        "logs": record.logs,
        "progress": record.progress,
    }


@app.websocket("/ws/experiments/{exp_id}")
async def ws_experiment(websocket: WebSocket, exp_id: str):
    record = EXPERIMENTS.get(exp_id)
    if not record:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    record.subscribe(queue)
    try:
        # send current state
        await websocket.send_json({"type": "progress", "percent": record.progress})
        for log in record.logs:
            await websocket.send_json({"type": "log", "message": log})
        if record.status == "done":
            await websocket.send_json({"type": "done"})
        elif record.status == "error":
            await websocket.send_json({"type": "error", "message": record.error or ""})
        # forward future messages
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        record.unsubscribe(queue)
