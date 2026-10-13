import request from './request'
import { mock } from './mock'

/*
 * 实验接口
 * ─────────────────────────────────────────────
 * 接口位置 1：POST /api/experiments
 * 后端来源：backend/services/train_service.py 的 run_experiment()
 * 用途：提交一次实验（数据集 + 算法 + 参数），返回 experiment_id
 *
 * 请求体：
 * {
 *   "dataset": "iris",
 *   "algorithm": "knn",
 *   "task_type": "classification",
 *   "params": { "k": 5 },
 *   "test_size": 0.2,
 *   "random_state": 42
 * }
 *
 * 期望返回：
 * { "experiment_id": "..." }
 *
 * ─────────────────────────────────────────────
 * 接口位置 2：GET /api/experiments/{id}
 * 用途：查询实验结果
 *
 * 期望返回（与 train_service.run_experiment 返回对齐）：
 * {
 *   "algorithm": "knn",
 *   "task_type": "classification",
 *   "parameters": { "k": 5 },
 *   "prediction_time_seconds": 0.01,
 *   "predictions": [...],
 *   "metrics": {
 *     "accuracy": 0.96,
 *     "precision_macro": 0.96,
 *     "recall_macro": 0.96,
 *     "f1_macro": 0.96,
 *     "confusion_matrix": [[...]],
 *     "labels": [...]
 *   }
 * }
 * 回归任务 metrics 为：{ "mae", "mse", "rmse", "r2" }
 *
 * ─────────────────────────────────────────────
 * 接口位置 3：WS /ws/experiments/{id}  （WebSocket，非 axios）
 * 用途：训练过程实时推送进度/日志
 * 前端在 Training.vue 中用原生 WebSocket 连接，见该文件注释
 */

// 接口位置：POST /api/experiments
export async function createExperiment(payload) {
  try {
    return await request.post('/experiments', payload)
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
    // mock：返回一份示例结果，分类任务展示混淆矩阵
    return {
      algorithm: 'knn',
      task_type: 'classification',
      parameters: { k: 5 },
      prediction_time_seconds: 0.012,
      predictions: [0, 1, 1, 0, 2, 1, 0, 2],
      metrics: {
        accuracy: 0.96,
        precision_macro: 0.96,
        recall_macro: 0.96,
        f1_macro: 0.96,
        confusion_matrix: [[50, 0, 0], [0, 48, 2], [1, 1, 48]],
        labels: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
      },
    }
  }
}
