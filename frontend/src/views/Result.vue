<template>
  <el-card v-loading="loading">
    <template #header>实验结果</template>
    <div v-if="result">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="算法">{{ result.algorithm }}</el-descriptions-item>
        <el-descriptions-item label="任务">{{ result.task_type }}</el-descriptions-item>
        <el-descriptions-item label="参数">{{ JSON.stringify(result.parameters) }}</el-descriptions-item>
        <el-descriptions-item label="预测耗时(s)">{{ result.prediction_time_seconds }}</el-descriptions-item>
      </el-descriptions>

      <h3>评价指标</h3>
      <el-table :data="metricRows" border>
        <el-table-column prop="name" label="指标" width="160" />
        <el-table-column prop="value" label="值" />
      </el-table>

      <!-- 混淆矩阵（分类任务） -->
      <template v-if="result.metrics?.confusion_matrix">
        <h3>混淆矩阵</h3>
        <el-table :data="confusionRows" border>
          <el-table-column prop="label" label="真实\预测" width="120" />
          <el-table-column
            v-for="(lbl, idx) in result.metrics.labels"
            :key="idx"
            :label="String(lbl)"
          >
            <template #default="{ row }">
              {{ row.values[idx] }}
            </template>
          </el-table-column>
        </el-table>
      </template>

      <!-- 后续可在此处加入 ECharts：预测曲线、聚类散点、算法对比等 -->
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getExperiment } from '../api/experiments'

const route = useRoute()
const experimentId = route.params.experimentId

const result = ref(null)
const loading = ref(false)

onMounted(async () => {
  loading.value = true
  try {
    // 接口位置：GET /api/experiments/{id}
    // 返回结构: { id, status, result: {...}, logs, error }
    const response = await getExperiment(experimentId)
    result.value = response.result
  } catch {
    // 错误已统一提示
  } finally {
    loading.value = false
  }
})

// 把 metrics 平铺成表格行
const metricRows = computed(() => {
  if (!result.value?.metrics) return []
  const skip = ['confusion_matrix', 'labels']
  return Object.entries(result.value.metrics)
    .filter(([k]) => !skip.includes(k))
    .map(([k, v]) => ({ name: k, value: typeof v === 'number' ? v.toFixed(4) : JSON.stringify(v) }))
})

// 混淆矩阵行结构
const confusionRows = computed(() => {
  const m = result.value?.metrics?.confusion_matrix
  const labels = result.value?.metrics?.labels
  if (!m || !labels) return []
  return labels.map((label, i) => ({ label, values: m[i] }))
})
</script>
