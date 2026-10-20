<template>
  <div>
    <el-card>
      <template #header>实验配置</template>
      <el-form :model="form" label-width="120px">
        <el-form-item label="数据集">
          <el-select v-model="form.dataset" placeholder="选择数据集" style="width: 100%">
            <el-option
              v-for="d in datasets"
              :key="d.name"
              :label="`${d.name}（${taskNames[d.task_type] || d.task_type}）`"
              :value="d.name"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="任务类型">
          <el-radio-group v-model="form.task_type">
            <el-radio label="classification">分类</el-radio>
            <el-radio label="regression">回归</el-radio>
            <el-radio label="clustering">聚类</el-radio>
            <el-radio label="dimensionality_reduction">降维</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item label="算法">
          <el-select v-model="form.algorithm" placeholder="选择算法" style="width: 100%">
            <el-option
              v-for="a in filteredAlgorithms"
              :key="a.name"
              :label="`${a.display_name}（${formatTasks(a.task_type)}）`"
              :value="a.name"
            />
          </el-select>
          <!-- 选中算法后的任务类型提示 -->
          <div v-if="selectedAlgo" class="algo-hint">
            <el-tag size="small">适用任务：{{ formatTasks(selectedAlgo.task_type) }}</el-tag>
            <el-tag
              v-if="['classification', 'regression'].includes(form.task_type)"
              size="small"
              type="success"
            >
              当前按「{{ taskNames[form.task_type] }}」模式训练
            </el-tag>
            <el-tag v-if="taskMismatch" size="small" type="danger">
              与所选数据集（{{ taskNames[selectedDataset.task_type] || selectedDataset.task_type }}）不匹配
            </el-tag>
          </div>
        </el-form-item>

        <el-form-item label="参数">
          <div class="params">
            <div v-for="(val, key) in form.params" :key="key" class="param-row">
              <span class="param-key">{{ key }}</span>
              <!-- task 是字符串枚举（classification/regression），不能用数字框 -->
              <el-select v-if="key === 'task'" v-model="form.params[key]" style="flex: 1">
                <el-option label="分类" value="classification" />
                <el-option label="回归" value="regression" />
              </el-select>
              <el-input-number v-else-if="typeof val === 'number'" v-model="form.params[key]" />
              <el-input v-else v-model="form.params[key]" />
            </div>
            <div v-if="Object.keys(form.params).length === 0" class="param-hint">该算法无参数</div>
          </div>
        </el-form-item>

        <el-form-item label="测试集比例">
          <el-slider v-model="form.test_size" :min="0.1" :max="0.5" :step="0.05" show-input />
        </el-form-item>

        <el-form-item label="随机种子">
          <el-input-number v-model="form.random_state" :min="0" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" :loading="submitting" @click="onSubmit">开始训练</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { listAlgorithms } from '../api/algorithms'
import { listDatasets } from '../api/datasets'
import { createExperiment } from '../api/experiments'
import { taskNames, formatTasks } from '../utils/labels'

const router = useRouter()

const datasets = ref([])
const algorithms = ref([])

const form = ref({
  dataset: '',
  task_type: 'classification',
  algorithm: '',
  params: {},
  test_size: 0.2,
  random_state: 42,
})

const submitting = ref(false)

// 根据任务类型过滤算法（registry 的 task_type 字段可能是 "classification" 或 "classification / regression"）
const filteredAlgorithms = computed(() =>
  algorithms.value.filter((a) => a.task_type.includes(form.value.task_type))
)

const selectedAlgo = computed(() =>
  algorithms.value.find((a) => a.name === form.value.algorithm)
)
const selectedDataset = computed(() =>
  datasets.value.find((d) => d.name === form.value.dataset)
)
// 算法与数据集任务冲突（聚类/降维算法不限数据集，跳过判断）
const taskMismatch = computed(() => {
  const a = selectedAlgo.value
  const d = selectedDataset.value
  if (!a || !d || ['clustering', 'dimensionality_reduction'].includes(d.task_type)) return false
  return !a.task_type.includes(d.task_type)
})

// 切换任务类型时重置算法选择，避免残留与任务不匹配的算法
watch(
  () => form.value.task_type,
  () => {
    form.value.algorithm = ''
    form.value.params = {}
  }
)

// 切换算法时加载其默认参数；task 参数跟随当前任务类型
watch(
  () => form.value.algorithm,
  (name) => {
    const algo = algorithms.value.find((a) => a.name === name)
    const params = algo ? JSON.parse(JSON.stringify(algo.default_params || {})) : {}
    if ('task' in params && ['classification', 'regression'].includes(form.value.task_type)) {
      params.task = form.value.task_type
    }
    form.value.params = params
  }
)

onMounted(async () => {
  // 接口位置：GET /api/datasets
  datasets.value = await listDatasets()
  // 接口位置：GET /api/algorithms
  algorithms.value = await listAlgorithms()
})

async function onSubmit() {
  if (!form.value.dataset || !form.value.algorithm) {
    ElMessage.warning('请选择数据集和算法')
    return
  }
  submitting.value = true
  try {
    // 接口位置：POST /api/experiments
    const res = await createExperiment(form.value)
    ElMessage.success('实验已提交')
    // 跳转到训练进度页，experiment_id 由后端返回
    router.push({ name: 'training', params: { experimentId: res.experiment_id } })
  } catch {
    // 错误已统一提示
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.params { display: flex; flex-direction: column; gap: 8px; }
.param-row { display: flex; align-items: center; gap: 12px; }
.param-key { width: 140px; color: #606266; }
.param-hint { color: #909399; }
.algo-hint { display: flex; gap: 6px; margin-top: 6px; }
</style>
