import request from './request'
import { mock } from './mock'

/*
 * 实验接口
 * ─────────────────────────────────────────────
 * 接口位置 1：POST /api/experiments
 * 后端来源：backend/api/main.py（训练在线程池异步执行，返回 202）
 *
 * 后端请求体（ExperimentRequest）：
 * {
 *   "dataset": "iris",
 *   "algorithm": "knn",
 *   "parameters": { "k": 5 },
 *   "test_size": 0.2,
 *   "random_state": 42,
 *   "standardize": true
 * }
 * 注意超参字段名是 parameters，表单里叫 params，这里负责映射；
 * task_type 由后端按算法自动判定，不需要传。
 *
 * 返回：
 * { "experiment_id": "...", "status": "pending",
 *   "poll_url": "/api/experiments/{id}", "websocket_url": "/ws/experiments/{id}" }
 *
 * ─────────────────────────────────────────────
 * 接口位置 2：GET /api/experiments/{id}
 * 用途：查询实验状态与结果
 *
 * 返回（status 为 pending/running/finished/failed）：
 * {
 *   "experiment_id": "...",
 *   "status": "finished",
 *   "progress": 1.0,           // 0~1 的小数
 *   "message": "完成",
 *   "logs": [{ "progress": 0.5, "message": "训练 knn" }, ...],
 *   "result": {                // 仅 finished 时有值
 *     "algorithm": "knn",
 *     "task_type": "classification",
 *     "parameters": { "k": 5 },
 *     "train_time_seconds": 0.01,
 *     "predict_time_seconds": 0.01,
 *     "predictions": [...],
 *     "metrics": { "accuracy", "precision_macro", "recall_macro", "f1_macro",
 *                  "confusion_matrix", "labels", "label_names" }
 *     // 回归任务 metrics 为：{ "mae", "mse", "rmse", "r2" }
 *   },
 *   "error": null,
 *   "traceback": null          // 仅 failed 时有值
 * }
 *
 * ─────────────────────────────────────────────
 * 接口位置 3：WS /ws/experiments/{id}  （WebSocket，非 axios）
 * 后端推送格式：
 *   { "type": "snapshot"|"progress", "status", "progress": 0~1, "message" }
 *   { "type": "result", "status": "finished", "result": {...} }
 *   { "type": "error", "status": "failed", "error": "..." }
 *   { "type": "heartbeat", "status": "running" }   // 保活帧，忽略即可
 * 前端在 Training.vue 中用原生 WebSocket 连接，见该文件注释
 */

// 接口位置：POST /api/experiments
export async function createExperiment(payload) {
  // 表单字段映射为后端 ExperimentRequest：params → parameters
  const body = {
    dataset: payload.dataset,
    algorithm: payload.algorithm,
    parameters: payload.params || {},
    test_size: payload.test_size,
    random_state: payload.random_state,
  }
  try {
    return await request.post('/experiments', body)
  } catch {
    console.warn('[mock] POST /api/experiments 后端未就绪，使用 mock 数据')
    // mock：返回一个随机 experiment_id，实际结果由 getExperiment 的 mock 返回
    return { experiment_id: 'mock-' + Date.now() }
  }
}

// 接口位置：GET /api/experiments/{id}
export async function getExperiment(id) {
  try {
    return await request.get(`/experiments/${id}`)
  } catch {
    console.warn(`[mock] GET /api/experiments/${id} 后端未就绪，使用 mock 数据`)
    // mock：与新后端结构对齐（status + result 包装），分类任务展示混淆矩阵
    return {
      experiment_id: id,
      status: 'finished',
      progress: 1.0,
      message: '完成',
      logs: [],
      result: {
        algorithm: 'knn',
        task_type: 'classification',
        parameters: { k: 5 },
        train_time_seconds: 0.01,
        predict_time_seconds: 0.012,
        predictions: [0, 1, 1, 0, 2, 1, 0, 2],
        metrics: {
          accuracy: 0.96,
          precision_macro: 0.96,
          recall_macro: 0.96,
          f1_macro: 0.96,
          confusion_matrix: [[50, 0, 0], [0, 48, 2], [1, 1, 48]],
          labels: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
        },
      },
      error: null,
    }
  }
}
