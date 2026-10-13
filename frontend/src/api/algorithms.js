import request from './request'
import { mock } from './mock'

/*
 * 算法接口
 * ─────────────────────────────────────────────
 * 接口位置：GET /api/algorithms
 * 后端来源：backend/algorithms/registry.py 的 list_algorithms()
 * 用途：返回所有已注册算法及其任务类型、默认参数
 *
 * 期望返回结构（与 registry.list_algorithms 对齐）：
 * [
 *   {
 *     "name": "knn",
 *     "display_name": "K-近邻",
 *     "task_type": "classification",
 *     "implemented_in_house": true,
 *     "default_params": { "k": 5 }
 *   },
 *   ...
 * ]
 *
 * 支持的算法（当前 registry 中）：
 *   linear_regression, logistic_regression, knn, decision_tree,
 *   svm, kmeans, random_forest, gaussian_naive_bayes, pca, gradient_boosting
 */

// 接口位置：GET /api/algorithms?task_type=classification
// 后端未就绪时自动回退到 mock 数据，后端就绪后自动使用真实接口
export async function listAlgorithms(taskType) {
  try {
    return await request.get('/algorithms', { params: taskType ? { task_type: taskType } : {} })
  } catch {
    console.warn('[mock] GET /api/algorithms 后端未就绪，使用 mock 数据')
    return mock.algorithms.filter((a) => !taskType || a.task_type.includes(taskType))
  }
}
