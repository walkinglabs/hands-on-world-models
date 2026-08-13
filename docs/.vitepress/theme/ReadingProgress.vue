<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";

const progress = ref(0);
const visible = ref(false);

function updateProgress() {
  const available = document.documentElement.scrollHeight - window.innerHeight;
  progress.value =
    available > 0 ? Math.round((window.scrollY / available) * 100) : 0;
  visible.value = window.scrollY > 120;
}

function returnToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

onMounted(() => {
  window.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();
});

onBeforeUnmount(() => {
  window.removeEventListener("scroll", updateProgress);
});
</script>

<template>
  <Transition name="hwm-progress-fade">
    <button
      v-if="visible"
      class="hwm-reading-progress"
      type="button"
      :title="`阅读进度 ${progress}%，点击回到顶部`"
      aria-label="回到顶部"
      @click="returnToTop"
    >
      <svg viewBox="0 0 44 44" aria-hidden="true">
        <circle class="hwm-progress-track" cx="22" cy="22" r="18" />
        <circle
          class="hwm-progress-value"
          cx="22"
          cy="22"
          r="18"
          :style="{ strokeDashoffset: 113.1 - (progress / 100) * 113.1 }"
        />
      </svg>
      <span>{{ progress }}%</span>
    </button>
  </Transition>
</template>
