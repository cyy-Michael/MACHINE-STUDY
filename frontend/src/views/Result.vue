<template>
  <el-card v-loading="loading">
    <template #header>
      <div class="header-row">
        <span>实验结果</span>
        <el-radio-group v-model="viewMode" size="small">
          <el-radio-button value="chart">图表视图</el-radio-button>
          <el-radio-button value="table">表格数据</el-radio-button>
        </el-radio-group>
      </div>
    </template>
    <el-alert v-if="errorMsg" type="error" :title="errorMsg" :closable="false" />
    <div v-if="result">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="算法">{{ algoLabel }}</el-descriptions-item>
        <el-descriptions-item label="任务">{{ taskNames[result.task_type] || result.task_type }}</el-descriptions-item>
        <el-descriptions-item label="训练耗时(s)">{{ result.train_time_seconds }}</el-descriptions-item>
        <el-descriptions-item label="预测耗时(s)">{{ result.predict_time_seconds }}</el-descriptions-item>
        <!-- 参数逐项单独显示，中文标签 -->
        <el-descriptions-item
          v-for="(v, k) in result.parameters"
          :key="k"
          :label="paramNames[k] || k"
        >
          {{ k === 'task' ? (taskNames[v] || v) : v }}
        </el-descriptions-item>
      </el-descriptions>

      <!-- ══ 表格视图（原始数据） ══ -->
      <div v-show="viewMode === 'table'">
        <h3>评价指标</h3>
        <el-table :data="metricRows" border>
          <el-table-column prop="name" label="指标" width="160" />
          <el-table-column prop="value" label="值" />
        </el-table>

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
      </div>

      <!-- ══ 图表视图（可视化） ══ -->
      <div v-show="viewMode === 'chart'">
        <h3>评价指标</h3>
        <div class="chart-row">
          <div ref="metricsEl" class="chart"></div>
          <div v-if="isClassification" ref="radarEl" class="chart"></div>
        </div>

        <template v-if="result.metrics?.confusion_matrix">
          <h3>混淆矩阵</h3>
          <div ref="cmEl" class="chart"></div>
          <h3>标签分布：真实 vs 预测</h3>
          <div ref="distEl" class="chart"></div>
        </template>

        <!-- 回归：预测散点 + 残差分布 -->
        <template v-if="viz.true_vs_pred">
          <h3>真实值 vs 预测值</h3>
          <div ref="scatterEl" class="chart"></div>
          <h3>残差分布（真实 − 预测）</h3>
          <div ref="residEl" class="chart"></div>
        </template>
        <!-- 分类/聚类/降维：2D 投影散点 -->
        <template v-else-if="viz.scatter_2d">
          <h3>预测结果分布（2D 投影）</h3>
          <div ref="scatterEl" class="chart"></div>
        </template>

        <!-- PCA：各主成分方差解释率 -->
        <template v-if="explainedVariance">
          <h3>主成分方差解释率</h3>
          <div ref="varEl" class="chart"></div>
        </template>
      </div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import * as echarts from 'echarts'
import { getExperiment } from '../api/experiments'
import { listAlgorithms } from '../api/algorithms'
import { taskNames, paramNames } from '../utils/labels'

const route = useRoute()
const experimentId = route.params.experimentId

const result = ref(null)
const errorMsg = ref('')
const loading = ref(false)
const viewMode = ref('chart') // chart=图表视图 table=表格数据

const metricsEl = ref(null)
const radarEl = ref(null)
const cmEl = ref(null)
const distEl = ref(null)
const scatterEl = ref(null)
const residEl = ref(null)
const varEl = ref(null)
let charts = []

// 指标名的中文显示
const metricNames = {
  accuracy: '准确率', precision_macro: '精确率(宏)', recall_macro: '召回率(宏)', f1_macro: 'F1(宏)',
  mae: 'MAE', mse: 'MSE', rmse: 'RMSE', r2: 'R²',
}

const isClassification = computed(() => result.value?.task_type === 'classification')

// 算法英文名 → 中文显示名（来自 /api/algorithms 的 display_name）
const algoMap = ref({})
const algoLabel = computed(() => {
  const name = result.value?.algorithm || ''
  const display = algoMap.value[name]
  return display ? `${display}（${name}）` : name
})

// metrics 里的数值型指标（跳过 confusion_matrix / labels 等数组）
const numericMetrics = computed(() =>
  Object.entries(result.value?.metrics || {}).filter(([, v]) => typeof v === 'number')
)

// 后端按任务类型给的可视化数据：
// 回归 { true_vs_pred: [[真实, 预测], ...] }；分类/聚类 { scatter_2d: [[x, y, 标签], ...], projection }
const viz = computed(() => result.value?.visualization || {})

// PCA 的方差解释率（若后端提供）
const explainedVariance = computed(() => {
  const evr = result.value?.metrics?.explained_variance_ratio
  return Array.isArray(evr) && evr.length && evr.every((v) => typeof v === 'number') ? evr : null
})

// ── 表格视图数据 ──
const metricRows = computed(() => {
  if (!result.value?.metrics) return []
  const skip = ['confusion_matrix', 'labels', 'label_names']
  return Object.entries(result.value.metrics)
    .filter(([k]) => !skip.includes(k))
    .map(([k, v]) => ({
      name: metricNames[k] || k,
      value: typeof v === 'number' ? v.toFixed(4) : JSON.stringify(v),
    }))
})
const confusionRows = computed(() => {
  const m = result.value?.metrics?.confusion_matrix
  const labels = result.value?.metrics?.labels
  if (!m || !labels) return []
  return labels.map((label, i) => ({ label, values: m[i] }))
})

function makeChart(el, option) {
  if (!el) return
  const c = echarts.init(el)
  c.setOption(option)
  charts.push(c)
}

function renderCharts() {
  // 1. 评价指标条形图
  const entries = numericMetrics.value
  if (entries.length) {
    // 回归指标量级差异大（如 MSE≈25万 vs R²≈0.3），比值悬殊时切对数刻度，避免小值看不见
    const values = entries.map(([, v]) => v)
    const useLog = values.every((v) => v > 0) && Math.max(...values) / Math.min(...values) > 100
    makeChart(metricsEl.value, {
      tooltip: {},
      grid: { left: 60, right: 20, top: 30, bottom: 30, containLabel: true },
      xAxis: { type: 'category', data: entries.map(([k]) => metricNames[k] || k) },
      yAxis: useLog
        ? { type: 'log', name: '对数刻度', nameLocation: 'middle', nameGap: 50 }
        : { type: 'value' },
      series: [{
        type: 'bar',
        data: entries.map(([, v]) => Number(v.toFixed(4))),
        itemStyle: { color: '#409EFF' },
        label: { show: true, position: 'top' },
        barMaxWidth: 50,
      }],
    })
  }

  // 2. 分类指标雷达图（指标都在 0~1，适合雷达展示）
  if (isClassification.value && entries.length) {
    makeChart(radarEl.value, {
      tooltip: {},
      radar: {
        indicator: entries.map(([k]) => ({ name: metricNames[k] || k, max: 1 })),
        radius: '65%',
      },
      series: [{
        type: 'radar',
        data: [{
          value: entries.map(([, v]) => Number(v.toFixed(4))),
          name: '指标',
          areaStyle: { color: 'rgba(64, 158, 255, 0.25)' },
          lineStyle: { color: '#409EFF' },
          label: { show: true, formatter: (p) => p.value },
        }],
      }],
    })
  }

  // 3. 混淆矩阵热力图（行=真实，列=预测）
  const cm = result.value?.metrics?.confusion_matrix
  const labels = result.value?.metrics?.labels || result.value?.metrics?.label_names
  if (cm && labels) {
    const data = []
    let maxV = 0
    cm.forEach((row, i) => row.forEach((v, j) => {
      data.push([j, i, v])
      if (v > maxV) maxV = v
    }))
    makeChart(cmEl.value, {
      tooltip: { formatter: (p) => `真实 ${labels[p.value[1]]} → 预测 ${labels[p.value[0]]}：${p.value[2]}` },
      grid: { left: 110, right: 70, top: 30, bottom: 55, containLabel: true },
      xAxis: { type: 'category', data: labels.map(String), name: '预测', nameLocation: 'middle', nameGap: 28 },
      yAxis: { type: 'category', data: labels.map(String), name: '真实', nameLocation: 'middle', nameGap: 75, inverse: true },
      visualMap: {
        min: 0, max: maxV || 1, calculable: true,
        orient: 'vertical', right: 0, top: 'center',
        inRange: { color: ['#f0f7ff', '#409EFF'] },
      },
      series: [{ type: 'heatmap', data, label: { show: true } }],
    })

    // 4. 标签分布对比：真实（行和）vs 预测（列和）
    const trueCounts = cm.map((row) => row.reduce((a, b) => a + b, 0))
    const predCounts = labels.map((_, j) => cm.reduce((sum, row) => sum + row[j], 0))
    makeChart(distEl.value, {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 60, containLabel: true },
      xAxis: { type: 'category', data: labels.map(String), axisLabel: { rotate: labels.length > 4 ? 30 : 0 } },
      yAxis: { type: 'value' },
      series: [
        { name: '真实', type: 'bar', data: trueCounts, itemStyle: { color: '#67C23A' }, barMaxWidth: 40 },
        { name: '预测', type: 'bar', data: predCounts, itemStyle: { color: '#409EFF' }, barMaxWidth: 40 },
      ],
    })
  }

  // 5. 散点图
  if (viz.value.true_vs_pred) {
    // 回归：真实值 vs 预测值，红色虚线是"完美预测"对角线
    const pts = viz.value.true_vs_pred
    const flat = pts.flat()
    const min = Math.min(...flat)
    const max = Math.max(...flat)
    makeChart(scatterEl.value, {
      tooltip: { formatter: (p) => `真实 ${p.value[0]}<br/>预测 ${p.value[1]}` },
      grid: { left: 70, right: 40, top: 30, bottom: 55, containLabel: true },
      xAxis: { type: 'value', name: '真实值', nameLocation: 'middle', nameGap: 30 },
      yAxis: { type: 'value', name: '预测值', nameLocation: 'middle', nameGap: 50 },
      series: [
        { type: 'scatter', data: pts, symbolSize: 7, itemStyle: { color: '#409EFF', opacity: 0.6 } },
        { type: 'line', data: [[min, min], [max, max]], symbol: 'none', lineStyle: { color: '#F56C6C', type: 'dashed' }, tooltip: { show: false } },
      ],
    })

    // 6. 残差分布直方图（残差 = 真实 − 预测，越集中在 0 附近越好）
    const residuals = pts.map(([t, p]) => t - p)
    const rMin = Math.min(...residuals)
    const rMax = Math.max(...residuals)
    const binCount = 10
    const width = (rMax - rMin) / binCount || 1
    const bins = Array(binCount).fill(0)
    residuals.forEach((r) => {
      const idx = Math.min(binCount - 1, Math.floor((r - rMin) / width))
      bins[idx] += 1
    })
    makeChart(residEl.value, {
      tooltip: { formatter: (p) => `区间 ${p.name}<br/>样本数 ${p.value}` },
      grid: { left: 50, right: 20, top: 30, bottom: 60, containLabel: true },
      xAxis: {
        type: 'category',
        data: bins.map((_, i) => (rMin + width * (i + 0.5)).toFixed(2)),
        name: '残差',
        nameLocation: 'middle',
        nameGap: 40,
        axisLabel: { rotate: 30 },
      },
      yAxis: { type: 'value', name: '样本数', nameLocation: 'middle', nameGap: 45 },
      series: [{ type: 'bar', data: bins, itemStyle: { color: '#E6A23C' }, barMaxWidth: 50 }],
    })
  } else if (viz.value.scatter_2d) {
    // 分类/聚类/降维：2D 投影散点，按预测标签分组着色
    const groups = {}
    viz.value.scatter_2d.forEach(([x, y, lbl]) => {
      ;(groups[lbl] ||= []).push([x, y])
    })
    const names = Object.keys(groups).sort((a, b) => Number(a) - Number(b))
    const pc = viz.value.projection === 'pca'
    makeChart(scatterEl.value, {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 70, right: 40, top: 40, bottom: 55, containLabel: true },
      xAxis: { type: 'value', name: pc ? 'PC1' : '特征1', nameLocation: 'middle', nameGap: 30 },
      yAxis: { type: 'value', name: pc ? 'PC2' : '特征2', nameLocation: 'middle', nameGap: 50 },
      series: names.map((n) => ({
        name: `类别 ${n}`,
        type: 'scatter',
        data: groups[n],
        symbolSize: 7,
        itemStyle: { opacity: 0.7 },
      })),
    })
  }

  // 7. PCA 方差解释率
  if (explainedVariance.value) {
    makeChart(varEl.value, {
      tooltip: {},
      grid: { left: 60, right: 20, top: 30, bottom: 30, containLabel: true },
      xAxis: { type: 'category', data: explainedVariance.value.map((_, i) => `PC${i + 1}`) },
      yAxis: { type: 'value', max: 1 },
      series: [{
        type: 'bar',
        data: explainedVariance.value.map((v) => Number(v.toFixed(4))),
        itemStyle: { color: '#909399' },
        label: { show: true, position: 'top' },
        barMaxWidth: 60,
      }],
    })
  }
}

onMounted(async () => {
  loading.value = true
  try {
    // 接口位置：GET /api/experiments/{id}
    // 返回结构: { experiment_id, status, result: {...}, logs, error, traceback }
    const [response, algos] = await Promise.all([
      getExperiment(experimentId),
      listAlgorithms(),
    ])
    algoMap.value = Object.fromEntries(algos.map((a) => [a.name, a.display_name]))
    if (response.status === 'failed') {
      errorMsg.value = response.error || '训练失败'
    } else {
      result.value = response.result
      await nextTick() // 等 v-if 把图表容器渲染出来再初始化 ECharts
      renderCharts()
    }
  } catch {
    // 错误已统一提示
  } finally {
    loading.value = false
  }
})

onBeforeUnmount(() => {
  charts.forEach((c) => c.dispose())
  charts = []
})
</script>

<style scoped>
.header-row { display: flex; justify-content: space-between; align-items: center; }
.chart { width: 100%; height: 340px; }
.chart-row { display: flex; gap: 16px; }
.chart-row .chart { flex: 1; min-width: 0; }
</style>
