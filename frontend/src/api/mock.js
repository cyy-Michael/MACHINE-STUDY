// Mock 数据：后端 API 未就绪时使用，接口就绪后自动回退到真实请求
// 字段结构已与 backend/api/main.py 对齐（registry.list_datasets / describe_dataset）

const algorithms = [
  { name: 'linear_regression', display_name: '线性回归', task_type: 'regression', implemented_in_house: true, default_params: {} },
  { name: 'logistic_regression', display_name: '逻辑回归', task_type: 'classification', implemented_in_house: true, default_params: { learning_rate: 0.1, n_iters: 1000, reg_lambda: 0.0 } },
  { name: 'knn', display_name: 'K-近邻', task_type: 'classification', implemented_in_house: true, default_params: { k: 5 } },
  { name: 'decision_tree', display_name: '决策树（CART）', task_type: 'classification', implemented_in_house: true, default_params: { max_depth: 5, min_samples_split: 2 } },
  { name: 'svm', display_name: '支持向量机（线性核）', task_type: 'classification', implemented_in_house: true, default_params: { C: 1.0, learning_rate: 0.001, n_iters: 2000 } },
  { name: 'kmeans', display_name: 'K-平均聚类', task_type: 'clustering', implemented_in_house: true, default_params: { n_clusters: 3, max_iter: 300, tol: 1e-4, random_state: 42 } },
  { name: 'random_forest', display_name: '随机森林', task_type: 'classification / regression', implemented_in_house: true, default_params: { n_estimators: 50, max_depth: 8, max_features: 'sqrt', task: 'classification' } },
  { name: 'gaussian_naive_bayes', display_name: '高斯朴素贝叶斯', task_type: 'classification', implemented_in_house: true, default_params: { var_smoothing: 1e-9 } },
  { name: 'pca', display_name: '主成分分析（降维）', task_type: 'dimensionality_reduction', implemented_in_house: true, default_params: { n_components: 2 } },
  { name: 'gradient_boosting', display_name: '梯度增强', task_type: 'classification / regression', implemented_in_house: true, default_params: { n_estimators: 50, learning_rate: 0.05, max_depth: 2, task: 'regression' } },
]

const datasetList = [
  { name: 'iris', display_name: '鸢尾花', task_type: 'classification', n_samples: 150, n_features: 4, target: 'species', description: '', source_format: 'csv' },
  { name: 'phishing', display_name: '钓鱼网站', task_type: 'classification', n_samples: 11055, n_features: 30, target: 'Result', description: '', source_format: 'arff' },
  { name: 'airquality', display_name: '空气质量', task_type: 'regression', n_samples: 9471, n_features: 15, target: 'C6H6(GT)', description: '', source_format: 'csv' },
]

const datasetDetails = {
  iris: {
    name: 'iris',
    display_name: '鸢尾花',
    n_samples: 150,
    n_features: 4,
    task_type: 'classification',
    target: 'species',
    source_format: 'csv',
    fields: [
      { name: 'sepal_length', dtype: 'float', missing: 0, unique_sample: 35, samples: [5.1, 4.9, 4.7] },
      { name: 'sepal_width', dtype: 'float', missing: 0, unique_sample: 23, samples: [3.5, 3.0, 3.2] },
      { name: 'petal_length', dtype: 'float', missing: 0, unique_sample: 43, samples: [1.4, 1.3, 1.5] },
      { name: 'petal_width', dtype: 'float', missing: 0, unique_sample: 22, samples: [0.2, 0.1, 0.3] },
      { name: 'species', dtype: 'target', missing: 0, unique_sample: 3, samples: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'] },
    ],
    missing_total: 0,
    label_distribution: { 'Iris-setosa': 50, 'Iris-versicolor': 50, 'Iris-virginica': 50 },
  },
  phishing: {
    name: 'phishing',
    display_name: '钓鱼网站',
    n_samples: 11055,
    n_features: 30,
    task_type: 'classification',
    target: 'Result',
    source_format: 'arff',
    fields: [
      { name: 'having_IP_Address', dtype: 'int', missing: 0, unique_sample: 2, samples: [-1, 1] },
      { name: 'URL_Length', dtype: 'int', missing: 0, unique_sample: 3, samples: [1, 0, -1] },
      { name: 'Result', dtype: 'target', missing: 0, unique_sample: 2, samples: [-1, 1] },
    ],
    missing_total: 0,
    label_distribution: { '-1': 4898, '1': 6157 },
  },
  airquality: {
    name: 'airquality',
    display_name: '空气质量',
    n_samples: 9471,
    n_features: 15,
    task_type: 'regression',
    target: 'C6H6(GT)',
    source_format: 'csv',
    fields: [
      { name: 'CO(GT)', dtype: 'float', missing: 0, unique_sample: 120, samples: [2.6, 2.0, 2.2] },
      { name: 'C6H6(GT)', dtype: 'target', missing: 0, unique_sample: 200, samples: [11.9, 9.4, 10.5] },
    ],
    missing_total: 0,
  },
}

export const mock = { algorithms, datasetList, datasetDetails }
