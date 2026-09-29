# 一、接口文档

# ML 平台后端接口文档

> 
> 当前已完成接口：`GET /api/algorithms`
> 服务基础地址：`http://127.0.0.1:8000`
> 调试页面：`http://127.0.0.1:8000/docs`（Swagger 自动生成，可在线测试）

## 接口 1：获取全部算法列表

- 请求路径：`/api/algorithms`
- 请求方法：`GET`
- 接口描述：获取平台内置机器学习算法元信息，支持按任务类型筛选
- 请求参数（Query 查询参数，全部可选）

表格

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| task_type | string | 否 | 任务类型筛选，可选值：`classification`（分类）、`regression`（回归）；不传则返回全部算法 |

### 请求示例

```
# 获取全部算法
GET http://127.0.0.1:8000/api/algorithms

# 只获取分类算法
GET http://127.0.0.1:8000/api/algorithms?task_type=classification
```

### 返回示例（200 OK）

```
[
  {
    "name": "linear_regression",
    "display_name": "线性回归",
    "task_type": "regression",
    "implemented_in_house": true,
    "default_params": {}
  },
  {
    "name": "logistic_regression",
    "display_name": "逻辑回归",
    "task_type": "classification",
    "implemented_in_house": true,
    "default_params": {
      "learning_rate": 0.1,
      "n_iters": 1000,
      "reg_lambda": 0
    }
  },
  {
    "name": "knn",
    "display_name": "K-近邻",
    "task_type": "classification",
    "implemented_in_house": true,
    "default_params": {
      "k": 5
    }
  }
]
```

### 返回字段说明

表格

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| name | string | 算法内部标识名，后端调用时使用 |
| display_name | string | 前端展示用中文名称 |
| task_type | string | 任务类型：`classification`分类 / `regression`回归 |
| implemented_in_house | bool | 是否为本项目手写实现 |
| default_params | object | 算法默认超参 |

### 异常说明

- 传入非法 task_type：返回空数组，服务不会崩溃。

---

> 
> 待开发接口（预留，后续队友模块完成后补充）
> 
> 
> 1. `GET /api/datasets` 获取数据集列表
> 2. `POST /api/experiments` 创建训练实验
> 3. WebSocket 实验训练实时日志推送