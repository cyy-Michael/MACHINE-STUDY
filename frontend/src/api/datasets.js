import request from './request'
import { mock } from './mock'

/*
 * 数据集接口
 * ─────────────────────────────────────────────
 * 接口位置 1：GET /api/datasets
 * 后端来源：backend/data/registry.py 的 list_datasets()
 * 返回结构：
 * [
 *   { "name": "iris", "display_name": "鸢尾花", "task_type": "classification",
 *     "n_samples": 150, "n_features": 4, "target": "species",
 *     "description": "...", "source_format": "csv" },
 *   ...
 * ]
 *
 * 接口位置 2：GET /api/datasets/{name}
 * 后端来源：backend/data/registry.py 的 describe_dataset()
 * 返回结构：
 * {
 *   "name": "iris",
 *   "display_name": "鸢尾花",
 *   "task_type": "classification",
 *   "target": "species",
 *   "n_samples": 150,
 *   "n_features": 4,
 *   "source_format": "csv",
 *   "fields": [ { "name": "sepal_length", "dtype": "...", "missing": 0,
 *                 "unique_sample": 35, "samples": [5.1, 4.9, ...] }, ... ],
 *   "missing_total": 0,
 *   "label_distribution": { "Iris-setosa": 50, ... }   // 仅分类任务
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
