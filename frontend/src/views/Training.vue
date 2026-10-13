<template>
  <el-card>
    <template #header>训练进度</template>
    <el-progress :percentage="progress" :status="status" />
    <el-input
      v-model="logs"
      type="textarea"
      :rows="12"
      readonly
      style="margin-top: 16px; font-family: monospace"
    />
    <el-button
      v-if="done"
      type="primary"
      style="margin-top: 16px"
      @click="$router.push({ name: 'result', params: { experimentId } })"
    >
      查看结果
    </el-button>
  </el-card>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const experimentId = route.params.experimentId

const progress = ref(0)
const status = ref('')
const logs = ref('')
const done = ref(false)

let ws = null

onMounted(() => {
  // 接口位置：WS /ws/experiments/{id}
  // 注意：路径 /ws 已由 vite.config.js 代理到后端的 ws://localhost:8000
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(`${protocol}://${location.host}/ws/experiments/${experimentId}`)

  ws.onmessage = (event) => {
    // 期望后端推送格式：{ "type": "progress"|"log"|"done"|"error", ... }
    let msg
    try {
      msg = JSON.parse(event.data)
    } catch {
      msg = { type: 'log', message: event.data }
    }
    if (msg.type === 'progress') {
      progress.value = msg.percent || progress.value
    } else if (msg.type === 'log') {
      logs.value += (msg.message || '') + '\n'
    } else if (msg.type === 'done') {
      progress.value = 100
      status.value = 'success'
      done.value = true
    } else if (msg.type === 'error') {
      status.value = 'exception'
      logs.value += `错误: ${msg.message}\n`
    }
  }

  ws.onerror = () => {
    logs.value += '[mock] WebSocket 连接失败（后端未启动），使用 mock 结果。\n'
    // mock：模拟训练完成，2 秒后允许查看结果
    setTimeout(() => {
      progress.value = 100
      status.value = 'success'
      done.value = true
      logs.value += '[mock] 训练完成（mock）。\n'
    }, 2000)
  }
})

onBeforeUnmount(() => {
  ws?.close()
})
</script>
