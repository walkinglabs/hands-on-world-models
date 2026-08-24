<template>
  <div class="pwm">
    <div class="pwm-hint">
      用键盘方向键（或点击按钮）移动。右侧是<b>模型想象</b>：它根据「当前格子 + 你按的方向」预测你会到哪——
      这个预测来自它至今见过的一切转移。刚开始它会猜错；玩着玩着，它就学会了。
    </div>
    <div class="pwm-row">
      <div class="pwm-panel">
        <div class="pwm-title">真实世界</div>
        <canvas ref="real" width="156" height="156" class="pwm-canvas"></canvas>
      </div>
      <div class="pwm-panel">
        <div class="pwm-title">模型想象（预测下一步）</div>
        <canvas ref="imag" width="156" height="156" class="pwm-canvas"></canvas>
      </div>
    </div>
    <div class="pwm-controls">
      <button class="pwm-btn pwm-up" @click="act('up')">↑</button>
      <div class="pwm-mid">
        <button class="pwm-btn" @click="act('left')">←</button>
        <button class="pwm-btn pwm-reset" @click="reset" title="重置世界和模型">重置</button>
        <button class="pwm-btn" @click="act('right')">→</button>
      </div>
      <button class="pwm-btn pwm-down" @click="act('down')">↓</button>
    </div>
    <div class="pwm-stats">
      <span>经历转移：<b>{{ totalTransitions }}</b> / 36</span>
      <span>最近 10 次预测命中：<b>{{ hits }}/{{ attempts }}</b></span>
      <span v-if="ended" class="pwm-ended">{{ endedText }}</span>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from "vue";

const real = ref(null);
const imag = ref(null);

// 与课程 1.1 相同的世界：3×3，陷阱 (1,1)，目标 (2,2)，起点 (0,0)
const GOAL = [2, 2];
const TRAP = [1, 1];
const DELTA = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };

let agent = [0, 0];
let ended = ref(false);
let endedText = ref("");
let totalTransitions = ref(0);
let hits = ref(0);
let attempts = ref(0);

// 表格世界模型：counts[agent][action] -> {next: count}
// 键用字符串 "r,c"，与 notebook 里的计数转移表一一对应
const counts = {};

function key(pos) {
  return pos[0] + "," + pos[1];
}

function step(pos, action) {
  const [dr, dc] = DELTA[action];
  const nr = pos[0] + dr;
  const nc = pos[1] + dc;
  if (nr < 0 || nr > 2 || nc < 0 || nc > 2) return [pos[0], pos[1]]; // 撞墙不动
  return [nr, nc];
}

function predict(pos, action) {
  const table = counts[key(pos)] && counts[key(pos)][action];
  if (!table) return null; // 从未见过：模型一无所知
  let best = null;
  let bestN = -1;
  let sum = 0;
  for (const k in table) {
    sum += table[k];
    if (table[k] > bestN) {
      bestN = table[k];
      best = k;
    }
  }
  return { pos: best.split(",").map(Number), conf: bestN / sum };
}

function learn(pos, action, next) {
  const k = key(pos);
  if (!counts[k]) counts[k] = {};
  if (!counts[k][action]) counts[k][action] = {};
  const nk = key(next);
  counts[k][action][nk] = (counts[k][action][nk] || 0) + 1;
}

function drawReal() {
  draw(real.value, { agent, goal: true });
}

function drawImag(pred) {
  draw(imag.value, pred ? { agent: pred.pos, goal: true, ghost: pred.conf } : null);
}

function draw(canvas, state) {
  const ctx = canvas.getContext("2d");
  const cell = 52;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // 背景
  ctx.fillStyle = "rgba(128,128,128,0.12)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const x = c * cell;
      const y = r * cell;
      if (state && r === TRAP[0] && c === TRAP[1]) {
        ctx.fillStyle = "#d64545";
        ctx.fillRect(x, y, cell - 2, cell - 2);
        pixel(ctx, x + 18, y + 16, "#7c1f1f", [
          [0,0],[4,0],[12,0],[16,0],
          [4,4],[8,4],[12,4],
          [0,8],[4,8],[8,8],[12,8],[16,8],
          [4,12],[12,12],
          [0,16],[4,16],[12,16],[16,16],
        ]); // 骷髅像素点阵
      } else if (state && r === GOAL[0] && c === GOAL[1]) {
        ctx.fillStyle = "#3f9e63";
        ctx.fillRect(x, y, cell - 2, cell - 2);
        pixel(ctx, x + 14, y + 12, "#eaf5ec", [
          [0,4],[4,4],[8,4],[12,4],[16,4],
          [4,0],[4,8],[12,0],[12,8],[16,0],[16,8],
          [8,12],[8,16],[8,20],
        ]); // 旗帜像素点阵
      } else {
        ctx.fillStyle = "rgba(128,128,128,0.10)";
        ctx.fillRect(x, y, cell - 2, cell - 2);
      }
    }
  }
  if (!state) {
    ctx.fillStyle = "#8a8a8a";
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("？ 未知", canvas.width / 2, canvas.height / 2 + 5);
    return;
  }
  // 智能体像素小人
  const [ar, ac] = state.agent;
  const ax = ac * cell + 14;
  const ay = ar * cell + 12;
  const body = state.ghost != null ? "#8f7ab5" : "#2f6fb0";
  pixel(ctx, ax, ay, body, [
    [4,0],[8,0],
    [0,4],[4,4],[8,4],[12,4],
    [0,8],[4,8],[8,8],[12,8],
    [4,12],[8,12],
    [2,16],[6,16],[10,16],
  ]);
  if (state.ghost != null) {
    ctx.fillStyle = "#8f7ab5";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("置信 " + Math.round(state.ghost * 100) + "%", ac * cell + cell / 2 - 1, ar * cell + cell - 6);
  }
}

function pixel(ctx, ox, oy, color, dots) {
  ctx.fillStyle = color;
  for (const [dx, dy] of dots) ctx.fillRect(ox + dx, oy + dy, 4, 4);
}

function act(action) {
  if (ended.value) return;
  const pred = predict(agent, action);
  const next = step(agent, action);
  attempts.value++;
  if (pred && pred.pos[0] === next[0] && pred.pos[1] === next[1]) hits.value++;
  learn(agent, action, next);
  if (key(agent) !== key(next)) totalTransitions.value++;
  agent = next;
  drawReal();
  if ((agent[0] === GOAL[0] && agent[1] === GOAL[1]) || (agent[0] === TRAP[0] && agent[1] === TRAP[1])) {
    ended.value = true;
    const win = agent[0] === GOAL[0] && agent[1] === GOAL[1];
    endedText.value = win
      ? "🏁 到达目标！重置后模型会保留学到的一切。"
      : "💀 掉进陷阱。重置再试——注意看模型是否已经记住这个坑。";
    drawImag(null);
    return;
  }
  drawImag(predict(agent, "up")); // 展示一个“接下来最可能去哪”的样例预测
  // 更贴近课程语义：显示对刚执行动作的预测结果已用于对照，这里预演下一步所有动作中最自信的
  let bestPred = null;
  for (const a of ["up", "down", "left", "right"]) {
    const p = predict(agent, a);
    if (p && (!bestPred || p.conf > bestPred.conf)) bestPred = { pos: p.pos, ghost: p.conf };
  }
  drawImag(bestPred);
}

function reset() {
  agent = [0, 0];
  ended.value = false;
  drawReal();
  drawImag(null);
}

function onKey(e) {
  const map = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" };
  if (map[e.key]) {
    e.preventDefault();
    act(map[e.key]);
  }
}

onMounted(() => {
  window.addEventListener("keydown", onKey);
  drawReal();
  drawImag(null);
});

onUnmounted(() => {
  window.removeEventListener("keydown", onKey);
});
</script>

<style scoped>
.pwm {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 20px 0;
  background: var(--vp-c-bg-soft);
}
.pwm-hint {
  font-size: 0.9em;
  color: var(--vp-c-text-2);
  margin-bottom: 12px;
  line-height: 1.7;
}
.pwm-row {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}
.pwm-panel {
  text-align: center;
}
.pwm-title {
  font-size: 0.85em;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 6px;
}
.pwm-canvas {
  border-radius: 8px;
  image-rendering: pixelated;
  border: 1px solid var(--vp-c-divider);
}
.pwm-controls {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  margin-top: 12px;
}
.pwm-mid {
  display: flex;
  gap: 4px;
}
.pwm-btn {
  min-width: 44px;
  height: 34px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-size: 15px;
}
.pwm-btn:hover {
  border-color: var(--vp-c-brand);
}
.pwm-reset {
  font-size: 12px;
  padding: 0 10px;
}
.pwm-stats {
  display: flex;
  gap: 20px;
  justify-content: center;
  margin-top: 12px;
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  flex-wrap: wrap;
}
.pwm-ended {
  color: var(--vp-c-brand);
  font-weight: 600;
}
</style>
