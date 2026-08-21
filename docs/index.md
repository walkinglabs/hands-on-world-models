---
head:
  - - meta
    - http-equiv: refresh
      content: 0; url=./guide/world-model-intro.html
---

<!-- 首页重定向到导论；保留此文件以维持站点入口 -->

<script setup>
import { onMounted } from "vue";

onMounted(() => {
  window.location.replace("./guide/world-model-intro.html");
});
</script>
