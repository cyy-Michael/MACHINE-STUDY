// 界面显示用的中文标签映射（与后端交互的数据仍保持英文原值）

// 任务类型
export const taskNames = {
  classification: '分类',
  regression: '回归',
  clustering: '聚类',
  dimensionality_reduction: '降维',
}

// registry 的 task_type 可能是 "classification / regression" 这样的组合串
export const formatTasks = (t) =>
  (t || '').split('/').map((s) => taskNames[s.trim()] || s.trim()).join(' / ')

// 算法超参数名（键与 backend/algorithms/registry.py 的 default_params 对齐）
export const paramNames = {
  k: '近邻数 (k)',
  n_estimators: '树的数量',
  max_depth: '最大深度',
  max_features: '分裂特征数',
  min_samples_split: '最小分裂样本数',
  learning_rate: '学习率',
  n_iters: '迭代次数',
  reg_lambda: '正则化系数',
  C: '惩罚系数 (C)',
  var_smoothing: '方差平滑',
  n_clusters: '簇数量 (k)',
  max_iter: '最大迭代次数',
  tol: '收敛容差',
  random_state: '随机种子',
  task: '任务模式',
  n_components: '主成分数',
}
