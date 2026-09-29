# -*- coding: utf-8 -*-
from fastapi import FastAPI
from typing import Optional
from backend.algorithms.registry import list_algorithms

app = FastAPI(title="ML Platform API", prefix="/api")

@app.get("/algorithms")
def get_algorithms(task_type: Optional[str] = None):
    return list_algorithms(task_type=task_type)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=True)
