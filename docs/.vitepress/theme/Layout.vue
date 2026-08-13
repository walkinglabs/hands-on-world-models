<script setup>
import DefaultTheme from "vitepress/theme";
import { useData } from "vitepress";
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import MaintenanceBanner from "./MaintenanceBanner.vue";
import ReadingProgress from "./ReadingProgress.vue";

const { frontmatter, isDark } = useData();

const FONT_SIZE_KEY = "hwm-doc-font-size";
const LINE_HEIGHT_KEY = "hwm-doc-line-height";
const DOC_WIDTH_KEY = "hwm-doc-width";
const SIDEBAR_KEY = "hwm-sidebar-collapsed";

const fontSize = ref(17);
const lineHeight = ref(1.7);
const docWidth = ref(980);
const settingsOpen = ref(false);
const sidebarCollapsed = ref(false);
const allGroupsExpanded = ref(false);

const showReaderTools = computed(() => frontmatter.value.layout !== "home");
const groupButtonLabel = computed(() =>
  allGroupsExpanded.value ? "全部收起" : "全部展开",
);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value)));
}

function applyReadingSettings() {
  if (typeof document === "undefined") return;
  document.documentElement.style.setProperty(
    "--hwm-doc-font-size",
    `${clamp(fontSize.value, 15, 20)}px`,
  );
  document.documentElement.style.setProperty(
    "--hwm-doc-line-height",
    String(clamp(lineHeight.value, 1.55, 2)),
  );
  document.documentElement.style.setProperty(
    "--vp-doc-content-max-width",
    `${clamp(docWidth.value, 760, 1180)}px`,
  );
}

function saveReadingSettings() {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(FONT_SIZE_KEY, String(fontSize.value));
  localStorage.setItem(LINE_HEIGHT_KEY, String(lineHeight.value));
  localStorage.setItem(DOC_WIDTH_KEY, String(docWidth.value));
}

function resetReadingSettings() {
  fontSize.value = 17;
  lineHeight.value = 1.7;
  docWidth.value = 980;
}

function setAppearance(value) {
  isDark.value = value;
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value;
}

function applySidebarState() {
  if (typeof document === "undefined") return;
  document.body.classList.toggle(
    "hwm-sidebar-collapsed",
    sidebarCollapsed.value,
  );
  localStorage.setItem(SIDEBAR_KEY, String(sidebarCollapsed.value));
}

function sidebarGroups() {
  if (typeof document === "undefined") return [];
  return Array.from(document.querySelectorAll(".VPSidebarItem.collapsible"));
}

function syncGroupState() {
  const groups = sidebarGroups();
  allGroupsExpanded.value =
    groups.length > 0 &&
    groups.every((group) => !group.classList.contains("collapsed"));
}

function toggleAllSidebarGroups() {
  const shouldExpand = !allGroupsExpanded.value;
  for (const group of sidebarGroups()) {
    const collapsed = group.classList.contains("collapsed");
    if ((shouldExpand && collapsed) || (!shouldExpand && !collapsed)) {
      group.querySelector(":scope > .item")?.click();
    }
  }
  nextTick(syncGroupState);
}

function closeSettings(event) {
  if (!settingsOpen.value) return;
  if (event.key === "Escape") settingsOpen.value = false;
}

watch([fontSize, lineHeight, docWidth], () => {
  applyReadingSettings();
  saveReadingSettings();
});

watch(sidebarCollapsed, applySidebarState);

onMounted(() => {
  fontSize.value = clamp(localStorage.getItem(FONT_SIZE_KEY) || 17, 15, 20);
  lineHeight.value = clamp(
    localStorage.getItem(LINE_HEIGHT_KEY) || 1.7,
    1.55,
    2,
  );
  docWidth.value = clamp(localStorage.getItem(DOC_WIDTH_KEY) || 980, 760, 1180);
  sidebarCollapsed.value = localStorage.getItem(SIDEBAR_KEY) === "true";
  applyReadingSettings();
  applySidebarState();
  nextTick(syncGroupState);
  document.addEventListener("keydown", closeSettings);
});

onBeforeUnmount(() => {
  document.removeEventListener("keydown", closeSettings);
});
</script>

<template>
  <DefaultTheme.Layout>
    <template #layout-top>
      <MaintenanceBanner />
    </template>

    <template v-if="showReaderTools" #sidebar-nav-before>
      <Teleport defer to=".VPSidebar">
        <div class="hwm-sidebar-toolbar">
          <button
            class="hwm-sidebar-action hwm-expand-action"
            type="button"
            :aria-expanded="allGroupsExpanded"
            :title="groupButtonLabel"
            @click="toggleAllSidebarGroups"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path v-if="allGroupsExpanded" d="m4 10 4-4 4 4M4 14l4-4 4 4" />
              <path v-else d="m4 2 4 4 4-4M4 6l4 4 4-4" />
            </svg>
            <span>{{ groupButtonLabel }}</span>
          </button>

          <div class="hwm-sidebar-toolbar-end">
            <button
              class="hwm-sidebar-action hwm-icon-action"
              type="button"
              :title="isDark ? '切换到浅色' : '切换到深色'"
              :aria-label="isDark ? '切换到浅色' : '切换到深色'"
              @click="setAppearance(!isDark)"
            >
              <svg v-if="isDark" viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path
                  d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
                />
              </svg>
              <svg v-else viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M20.5 14.5A8.4 8.4 0 0 1 9.5 3.5 8.5 8.5 0 1 0 20.5 14.5Z"
                />
              </svg>
            </button>

            <button
              class="hwm-sidebar-action hwm-icon-action"
              type="button"
              title="阅读与外观设置"
              aria-label="阅读与外观设置"
              :aria-expanded="settingsOpen"
              @click="settingsOpen = !settingsOpen"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="3" />
                <path
                  d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"
                />
              </svg>
            </button>
          </div>
        </div>
      </Teleport>
    </template>
  </DefaultTheme.Layout>

  <ClientOnly>
    <button
      v-if="showReaderTools"
      class="hwm-sidebar-toggle"
      :class="{ collapsed: sidebarCollapsed }"
      type="button"
      :title="sidebarCollapsed ? '展开目录' : '收起目录'"
      :aria-label="sidebarCollapsed ? '展开目录' : '收起目录'"
      @click="toggleSidebar"
    >
      <svg viewBox="0 0 12 12" aria-hidden="true">
        <path v-if="sidebarCollapsed" d="m4 1 5 5-5 5" />
        <path v-else d="M8 1 3 6l5 5" />
      </svg>
    </button>
  </ClientOnly>

  <ClientOnly>
    <ReadingProgress v-if="showReaderTools" />
  </ClientOnly>

  <ClientOnly>
    <Teleport to="body">
      <div v-if="settingsOpen" class="hwm-settings-panel">
        <div class="hwm-settings-heading">
          <strong>阅读设置</strong>
          <button type="button" @click="resetReadingSettings">恢复默认</button>
        </div>

        <label>
          <span
            >字号 <b>{{ fontSize }}px</b></span
          >
          <input
            v-model.number="fontSize"
            type="range"
            min="15"
            max="20"
            step="1"
          />
        </label>
        <label>
          <span
            >行距 <b>{{ Number(lineHeight).toFixed(2) }}</b></span
          >
          <input
            v-model.number="lineHeight"
            type="range"
            min="1.55"
            max="2"
            step="0.05"
          />
        </label>
        <label>
          <span
            >正文宽度 <b>{{ docWidth }}px</b></span
          >
          <input
            v-model.number="docWidth"
            type="range"
            min="760"
            max="1180"
            step="20"
          />
        </label>
      </div>
    </Teleport>
  </ClientOnly>
</template>
