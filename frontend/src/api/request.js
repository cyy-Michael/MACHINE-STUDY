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
    const detail = error.response?.data?.detail || error.message || '请求失败'
    ElMessage.error(typeof detail === 'string' ? detail : JSON.stringify(detail))
    return Promise.reject(error)
  }
)

export default request
