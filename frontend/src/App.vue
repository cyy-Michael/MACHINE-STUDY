<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="logo">机器学习实验与可视化平台</div>
      <el-menu
        mode="horizontal"
        :router="true"
        :default-active="$route.path"
        class="app-menu"
      >
        <el-menu-item index="/">首页</el-menu-item>
        <el-menu-item index="/dataset">数据集</el-menu-item>
        <el-menu-item index="/experiment">实验配置</el-menu-item>
      </el-menu>
    </el-header>
    <el-main class="app-main">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup>
</script>

<style scoped>
.app-container { min-height: 100vh; }
.app-header { display: flex; align-items: center; border-bottom: 1px solid #e4e7ed; }
.logo { font-size: 18px; font-weight: 600; margin-right: 40px; }
.app-menu { flex: 1; border-bottom: none; }
.app-main { padding: 24px; background: #f5f7fa; }
</style>
