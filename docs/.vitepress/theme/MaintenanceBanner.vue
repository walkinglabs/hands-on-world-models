<template>
  <div class="course-maintenance" role="status">
    <strong>当前不建议学习本课程。</strong>
    <span>内容正在集中重写，请在 2026 年 8 月 24 日后再来查看。</span>
  </div>
</template>

<style scoped>
.course-maintenance {
  position: fixed;
  inset: 0 0 auto;
  z-index: var(--vp-z-index-layout-top);
  box-sizing: border-box;
  display: flex;
  min-height: var(--vp-layout-top-height);
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px 16px;
  color: #5d2509;
  font-size: 14px;
  line-height: 1.5;
  text-align: center;
  background: #fff1e8;
  border-bottom: 1px solid #efb28f;
  pointer-events: none;
}

@media (max-width: 639px) {
  .course-maintenance {
    flex-direction: column;
    gap: 0;
    font-size: 13px;
  }
}

.dark .course-maintenance {
  color: #ffd8c2;
  background: #402014;
  border-bottom-color: #7c3b23;
}
</style>
