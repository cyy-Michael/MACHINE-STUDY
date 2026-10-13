import request from './request'
import { mock } from './mock'

/*
 * 数据集接口
 * ─────────────────────────────────────────────
 * 接口位置 1：GET /api/datasets
 * 后端来源：backend/data/loader.py 的 list_local_datasets()
 * 用途：返回本地数据集清单
 *
 * 期望返回结构：
 * [
 *   { "name": "iris", "path": "...", "format": "csv" },
 *   ...
 * ]
 *
 * 接口位置 2：GET /api/datasets/{name}
 * 用途：返回数据集详情（字段、样例、缺失值、标签分布）
 *
 * 期望返回结构（建议后端提供，字段名需与后端确认）：
 * {
 *   "name": "iris",
 *   "n_samples": 150,
 *   "n_features": 4,
 *   "task_type": "classification",
 *   "label_column": "species",
 *   "columns": ["sepal_length", "sepal_width", ...],
 *   "head": [ { ... }, ... ],
 *   "missing": { "col1": 0, "col2": 0 },
 *   "label_distribution": { "setosa": 50, ... }
 * }
 */

// 接口位置：GET /api/datasets
export async function listDatasets() {
  try {
    return await request.get('/datasets')
  } catch {
    console.warn('[mock] GET /api/datasets 后端未就绪，使用 mock 数据')
    return mock.datasetList
  }
}

// 接口位置：GET /api/datasets/{name}
export async function getDataset(name) {
  try {
    return await request.get(`/datasets/${name}`)
  } catch {
    console.warn(`[mock] GET /api/datasets/${name} 后端未就绪，使用 mock 数据`)
    return mock.datasetDetails[name] || null
  }
}
