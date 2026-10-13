import { createRouter, createWebHistory } from 'vue-router'

// 页面按 frontend/README.md 规划的 5 个视图组织
const routes = [
  { path: '/', name: 'home', component: () => import('../views/Home.vue') },
  { path: '/dataset', name: 'dataset', component: () => import('../views/Dataset.vue') },
  { path: '/experiment', name: 'experiment', component: () => import('../views/Experiment.vue') },
  { path: '/training/:experimentId', name: 'training', component: () => import('../views/Training.vue'), props: true },
  { path: '/result/:experimentId', name: 'result', component: () => import('../views/Result.vue'), props: true },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
