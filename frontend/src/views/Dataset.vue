<template>
  <div>
    <el-row :gutter="20">
      <el-col :span="8">
        <el-card>
          <template #header>数据集列表</template>
          <!-- 接口位置：GET /api/datasets，见 src/api/datasets.js -->
          <el-table :data="datasets" v-loading="loading" @row-click="onSelect" highlight-current-row>
            <el-table-column prop="name" label="名称" />
            <el-table-column label="任务" width="140">
              <template #default="{ row }">{{ taskNames[row.task_type] || row.task_type }}</template>
            </el-table-column>
            <el-table-column prop="source_format" label="格式" width="80" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="16">
        <el-card v-if="detail">
          <template #header>{{ detail.name }} 详情</template>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="样本数">{{ detail.n_samples }}</el-descriptions-item>
            <el-descriptions-item label="特征数">{{ detail.n_features }}</el-descriptions-item>
            <el-descriptions-item label="任务类型">{{ taskNames[detail.task_type] || detail.task_type }}</el-descriptions-item>
            <el-descriptions-item label="标签列">{{ detail.target }}</el-descriptions-item>
          </el-descriptions>
          <template v-if="detail.label_distribution">
            <h4>标签分布</h4>
            <el-tag v-for="(count, label) in detail.label_distribution" :key="label" style="margin: 4px">
              {{ label }}: {{ count }}
            </el-tag>
          </template>
          <h4>字段列表</h4>
          <el-table :data="detail.fields || []" size="small" border>
            <el-table-column prop="name" label="字段" />
            <el-table-column prop="dtype" label="类型" width="110" />
            <el-table-column prop="missing" label="缺失" width="80" />
            <el-table-column label="样例值">
              <template #default="{ row }">{{ (row.samples || []).slice(0, 3).join(', ') }}</template>
            </el-table-column>
          </el-table>
        </el-card>
        <el-card v-else>
          <el-empty description="请选择左侧数据集" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { listDatasets, getDataset } from '../api/datasets'
import { taskNames } from '../utils/labels'

const datasets = ref([])
const detail = ref(null)
const loading = ref(false)

onMounted(async () => {
  loading.value = true
  try {
    // 接口位置：GET /api/datasets
    datasets.value = await listDatasets()
  } catch {
    // 错误已由 request 拦截器统一提示
  } finally {
    loading.value = false
  }
})

async function onSelect(row) {
  try {
    // 接口位置：GET /api/datasets/{name}
    detail.value = await getDataset(row.name)
  } catch {
    detail.value = null
  }
}
</script>
