# -*- coding: utf-8 -*-
from fastapi import APIRouter, FastAPI
from typing import Optional
from backend.algorithms.registry import list_algorithms

app = FastAPI(title="ML Platform API")

# 老师要求的接口统一挂在 /api 下（见 docs/api.md）。
# 注意：FastAPI(prefix=...) 不是合法参数，会被静默忽略，必须用 APIRouter。
# WebSocket 接口 /ws/experiments/{id} 不在 /api 下，另行直接挂在 app 上。
api_router = APIRouter(prefix="/api")


@api_router.get("/algorithms")
def get_algorithms(task_type: Optional[str] = None):
    return list_algorithms(task_type=task_type)


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=True)
