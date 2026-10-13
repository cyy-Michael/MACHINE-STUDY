# -*- coding: utf-8 -*-
"""异步实验运行器。

**为什么一定要放线程池**：随机森林和梯度增强都是纯 Python 循环，
``airquality`` 9000 行跑一次梯度增强要好几秒。直接在 FastAPI 的事件循环里
执行会把整个服务卡死——那几秒里连 ``/docs`` 都打不开。所以 POST 立即返回
``experiment_id``，真正训练在后台线程里跑。

进度通过订阅队列推给 WebSocket，跨线程投递用
``loop.call_soon_threadsafe``，这是唯一安全的做法。
"""
from __future__ import annotations

import asyncio
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from . import store
from .experiment import ExperimentError, run_dataset_experiment

# 内存里最多保留多少条任务记录，防止长跑的服务无限增长
MAX_RETAINED_JOBS = 200

TERMINAL_STATUSES = ("finished", "failed")


class ExperimentRunner:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="experiment"
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, list[tuple[Any, asyncio.Queue]]] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 提交
    def submit(self, payload: dict[str, Any]) -> str:
        experiment_id = payload.get("experiment_id") or self._new_id(payload)

        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": "pending",
            "progress": 0.0,
            "message": "排队中",
            "logs": [{"progress": 0.0, "message": "排队中"}],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "request": {
                key: value
                for key, value in payload.items()
                if key != "experiment_id"
            },
            "result": None,
            "error": None,
        }

        with self._lock:
            self._jobs[experiment_id] = record
            self._trim_locked()

        self._executor.submit(self._run, experiment_id, payload)
        return experiment_id

    def _new_id(self, payload: dict[str, Any]) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset = str(payload.get("dataset", "dataset"))
        algorithm = str(payload.get("algorithm", "algorithm"))
        return f"{stamp}-{algorithm}-{dataset}-{uuid.uuid4().hex[:6]}"

    def _trim_locked(self) -> None:
        if len(self._jobs) <= MAX_RETAINED_JOBS:
            return
        for key in list(self._jobs):
            if len(self._jobs) <= MAX_RETAINED_JOBS:
                break
            if self._jobs[key]["status"] in TERMINAL_STATUSES:
                del self._jobs[key]

    # ---------------------------------------------------------------- 执行
    def _run(self, experiment_id: str, payload: dict[str, Any]) -> None:
        self._update(experiment_id, status="running", progress=0.01, message="开始")

        def on_progress(message: str, fraction: float) -> None:
            self._update(
                experiment_id, status="running", progress=float(fraction), message=message
            )

        try:
            result = run_dataset_experiment(
                payload["dataset"],
                payload["algorithm"],
                payload.get("parameters") or {},
                test_size=float(payload.get("test_size", 0.2)),
                random_state=int(payload.get("random_state", 42)),
                standardize_features=bool(payload.get("standardize", True)),
                max_predictions=int(payload.get("max_predictions", 200)),
                experiment_id=experiment_id,
                progress=on_progress,
            )
            result["created_at"] = self._jobs[experiment_id]["created_at"]
            try:
                store.save_experiment(result)
                result["saved"] = True
            except OSError as exc:
                # 落盘失败不该让训练成果丢失，标记出来即可
                result["saved"] = False
                result["save_error"] = str(exc)

            with self._lock:
                self._jobs[experiment_id].update(
                    status="finished", progress=1.0, message="完成", result=result
                )
            self._publish(
                experiment_id,
                {"type": "result", "status": "finished", "result": result},
            )
        except (ExperimentError, ValueError, KeyError) as exc:
            self._fail(experiment_id, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 后台线程必须兜住一切
            self._fail(
                experiment_id,
                f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc(),
            )

    def _fail(self, experiment_id: str, message: str, detail: str | None = None) -> None:
        with self._lock:
            record = self._jobs.get(experiment_id)
            if record is not None:
                record.update(status="failed", message=message, error=message)
                if detail:
                    record["traceback"] = detail
                record["result"] = {
                    "experiment_id": experiment_id,
                    "status": "failed",
                    "error": message,
                    "dataset": record["request"].get("dataset"),
                    "algorithm": record["request"].get("algorithm"),
                    "created_at": record["created_at"],
                }
            saved = record["result"] if record else None
        if saved is not None:
            try:
                store.save_experiment(saved)
            except OSError:
                pass
        self._publish(
            experiment_id,
            {"type": "error", "status": "failed", "error": message},
        )

    # ---------------------------------------------------------------- 查询
    def _update(self, experiment_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs.get(experiment_id)
            if record is None:
                return
            record.update(changes)
            progress = record.get("progress")
            message = record.get("message")
            if message:
                record["logs"].append({"progress": progress, "message": message})
        if "message" in changes:
            self._publish(
                experiment_id,
                {
                    "type": "progress",
                    "status": changes.get("status", "running"),
                    "progress": changes.get("progress"),
                    "message": changes.get("message"),
                },
            )

    def snapshot(self, experiment_id: str) -> dict[str, Any] | None:
        """返回任务快照；内存里没有就去磁盘找已完成的记录。"""
        with self._lock:
            record = self._jobs.get(experiment_id)
            if record is not None:
                return {
                    key: value for key, value in record.items() if key != "traceback"
                }
        if store.experiment_exists(experiment_id):
            try:
                payload = store.load_experiment(experiment_id)
            except (store.ExperimentNotFound, ValueError):
                return None
            return {
                "experiment_id": experiment_id,
                "status": payload.get("status", "finished"),
                "progress": 1.0,
                "message": "完成",
                "created_at": payload.get("created_at"),
                "request": {},
                "result": payload,
                "error": payload.get("error"),
                "logs": [],
            }
        return None

    # ------------------------------------------------------------ 订阅推送
    def subscribe(self, experiment_id: str, loop: Any) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(experiment_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, experiment_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            entries = self._subscribers.get(experiment_id)
            if not entries:
                return
            self._subscribers[experiment_id] = [
                entry for entry in entries if entry[1] is not queue
            ]
            if not self._subscribers[experiment_id]:
                del self._subscribers[experiment_id]

    def _publish(self, experiment_id: str, event: dict[str, Any]) -> None:
        event = {"experiment_id": experiment_id, **event}
        with self._lock:
            entries = list(self._subscribers.get(experiment_id, []))
        for loop, queue in entries:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # 事件循环已经关闭（客户端断开），忽略即可
                continue

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


_runner: ExperimentRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> ExperimentRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = ExperimentRunner()
    return _runner
