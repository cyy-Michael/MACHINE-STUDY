// Mock 数据：后端 API 未就绪时使用，接口就绪后自动回退到真实请求
// 数据来源：backend/algorithms/registry.py（算法）和 backend/data/local/（数据集）

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
  { name: 'iris', path: 'backend/data/local/iris.csv', format: 'csv' },
  { name: 'phishing', path: 'backend/data/local/phishing.arff', format: 'arff' },
  { name: 'air_quality', path: 'backend/data/local/air_quality.csv', format: 'csv' },
]

const datasetDetails = {
  iris: {
    name: 'iris',
    n_samples: 150,
    n_features: 4,
    task_type: 'classification',
    label_column: 'species',
    columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
    head: [
      { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: 'Iris-setosa' },
      { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: 'Iris-setosa' },
      { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: 'Iris-setosa' },
      { sepal_length: 4.6, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.2, species: 'Iris-setosa' },
      { sepal_length: 5.0, sepal_width: 3.6, petal_length: 1.4, petal_width: 0.2, species: 'Iris-setosa' },
    ],
    missing: { sepal_length: 0, sepal_width: 0, petal_length: 0, petal_width: 0 },
    label_distribution: { 'Iris-setosa': 50, 'Iris-versicolor': 50, 'Iris-virginica': 50 },
  },
  phishing: {
    name: 'phishing',
    n_samples: 11055,
    n_features: 30,
    task_type: 'classification',
    label_column: 'Result',
    columns: ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report', 'Result'],
    head: [
      { having_IP_Address: -1, URL_Length: 1, Shortining_Service: 1, having_At_Symbol: 1, double_slash_redirecting: -1, Prefix_Suffix: -1, having_Sub_Domain: -1, SSLfinal_State: -1, Result: -1 },
      { having_IP_Address: 1, URL_Length: 1, Shortining_Service: 1, having_At_Symbol: 1, double_slash_redirecting: 1, Prefix_Suffix: -1, having_Sub_Domain: 0, SSLfinal_State: 1, Result: -1 },
    ],
    missing: {},
    label_distribution: { '-1': 4898, '1': 6157 },
  },
  air_quality: {
    name: 'air_quality',
    n_samples: 9471,
    n_features: 15,
    task_type: 'regression',
    label_column: 'C6H6(GT)',
    columns: ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'],
    head: [
      { Date: '10/03/2004', Time: '18.00.00', 'CO(GT)': 2.6, 'PT08.S1(CO)': 1360, 'NMHC(GT)': 150, 'C6H6(GT)': 11.9, 'PT08.S2(NMHC)': 1046, 'NOx(GT)': 166, 'PT08.S3(NOx)': 1056, 'NO2(GT)': 113, 'PT08.S4(NO2)': 1692, 'PT08.S5(O3)': 1268, T: 13.6, RH: 48.9, AH: 0.7578 },
      { Date: '10/03/2004', Time: '19.00.00', 'CO(GT)': 2.0, 'PT08.S1(CO)': 1292, 'NMHC(GT)': 112, 'C6H6(GT)': 9.4, 'PT08.S2(NMHC)': 955, 'NOx(GT)': 103, 'PT08.S3(NOx)': 1174, 'NO2(GT)': 92, 'PT08.S4(NO2)': 1559, 'PT08.S5(O3)': 972, T: 13.3, RH: 47.7, AH: 0.7255 },
    ],
    missing: {},
    label_distribution: {},
  },
}

export const mock = { algorithms, datasetList, datasetDetails }
