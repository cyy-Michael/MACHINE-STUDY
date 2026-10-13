<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card>
          <template #header>机器学习实验与可视化平台</template>
          <p>本平台支持在网页上选择数据集、配置算法参数、启动训练并查看指标与图表。</p>
          <el-space>
            <el-button type="primary" @click="$router.push('/experiment')">开始实验</el-button>
            <el-button @click="$router.push('/dataset')">查看数据集</el-button>
          </el-space>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="8">
        <el-card>
          <template #header>支持任务</template>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="分类">逻辑回归、KNN、决策树、SVM、朴素贝叶斯、随机森林、梯度增强</el-descriptions-item>
            <el-descriptions-item label="回归">线性回归、随机森林、梯度增强</el-descriptions-item>
            <el-descriptions-item label="聚类/降维">K-Means、PCA</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>评价指标</template>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="分类">Accuracy、Precision、Recall、F1、混淆矩阵</el-descriptions-item>
            <el-descriptions-item label="回归">MAE、MSE、RMSE、R²</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header>技术栈</template>
          <el-tag>FastAPI</el-tag>
          <el-tag>NumPy</el-tag>
          <el-tag>Vue 3</el-tag>
          <el-tag>Element Plus</el-tag>
          <el-tag>ECharts</el-tag>
          <el-tag>WebSocket</el-tag>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
// 首页暂不调用后端接口
</script>

<style scoped>
.home .el-card { margin-bottom: 0; }
</style>
