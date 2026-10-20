import axios from 'axios'
import { ElMessage } from 'element-plus'

// axios 统一封装
// - baseURL 设为 /api，配合 vite.config.js 的代理转发到后端
// - 响应拦截器统一处理错误，避免每个页面重复写 try/catch
const request = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // 后端统一错误结构为 { error: { type, message } }（见 backend/api/main.py），
    // detail 是 FastAPI 默认格式，留作兜底
    const data = error.response?.data
    const detail = data?.error?.message || data?.detail || error.message || '请求失败'
    ElMessage.error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    return Promise.reject(error)
  }
)

export default request
