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

const MIN_FONT_SIZE = 15;
const MAX_FONT_SIZE = 20;
const DEFAULT_FONT_SIZE = 17;
const MIN_LINE_HEIGHT = 1.55;
const MAX_LINE_HEIGHT = 2;
const DEFAULT_LINE_HEIGHT = 1.7;
const MIN_DOC_WIDTH = 780;
const MAX_DOC_WIDTH = 1280;
const DEFAULT_DOC_WIDTH = 980;

const fontSize = ref(DEFAULT_FONT_SIZE);
const lineHeight = ref(DEFAULT_LINE_HEIGHT);
const docWidth = ref(DEFAULT_DOC_WIDTH);
const settingsOpen = ref(false);
const settingsMenu = ref(null);
const settingsPanel = ref(null);
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
    `${clamp(fontSize.value, MIN_FONT_SIZE, MAX_FONT_SIZE)}px`,
  );
  document.documentElement.style.setProperty(
    "--hwm-doc-line-height",
    String(clamp(lineHeight.value, MIN_LINE_HEIGHT, MAX_LINE_HEIGHT)),
  );
  document.documentElement.style.setProperty(
    "--vp-doc-content-max-width",
    `${clamp(docWidth.value, MIN_DOC_WIDTH, MAX_DOC_WIDTH)}px`,
  );
}

function saveReadingSettings() {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(FONT_SIZE_KEY, String(fontSize.value));
  localStorage.setItem(LINE_HEIGHT_KEY, String(lineHeight.value));
  localStorage.setItem(DOC_WIDTH_KEY, String(docWidth.value));
}

function resetReadingSettings() {
  fontSize.value = DEFAULT_FONT_SIZE;
  lineHeight.value = DEFAULT_LINE_HEIGHT;
  docWidth.value = DEFAULT_DOC_WIDTH;
}

function adjustFontSize(amount) {
  fontSize.value = clamp(fontSize.value + amount, MIN_FONT_SIZE, MAX_FONT_SIZE);
}

function adjustLineHeight(amount) {
  lineHeight.value = Number(
    clamp(
      Number(lineHeight.value) + amount,
      MIN_LINE_HEIGHT,
      MAX_LINE_HEIGHT,
    ).toFixed(2),
  );
}

function adjustDocWidth(amount) {
  docWidth.value = clamp(docWidth.value + amount, MIN_DOC_WIDTH, MAX_DOC_WIDTH);
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
  const groups = sidebarGroups();
  const orderedGroups = shouldExpand ? groups : groups.slice().reverse();

  for (const group of orderedGroups) {
    const collapsed = group.classList.contains("collapsed");
    if ((shouldExpand && collapsed) || (!shouldExpand && !collapsed)) {
      group.querySelector(":scope > .item > .caret")?.click();
    }
  }

  if (!shouldExpand) {
    requestAnimationFrame(() => {
      document.querySelector(".VPSidebar > .nav")?.scrollTo({ top: 0 });
    });
  }

  nextTick(syncGroupState);
}

function closeSettingsOnEscape(event) {
  if (!settingsOpen.value) return;
  if (event.key === "Escape") settingsOpen.value = false;
}

function closeSettingsOnOutsideClick(event) {
  if (
    settingsOpen.value &&
    !settingsMenu.value?.contains(event.target) &&
    !settingsPanel.value?.contains(event.target)
  ) {
    settingsOpen.value = false;
  }
}

watch([fontSize, lineHeight, docWidth], () => {
  applyReadingSettings();
  saveReadingSettings();
});

watch(sidebarCollapsed, applySidebarState);

onMounted(() => {
  fontSize.value = clamp(
    localStorage.getItem(FONT_SIZE_KEY) || DEFAULT_FONT_SIZE,
    MIN_FONT_SIZE,
    MAX_FONT_SIZE,
  );
  lineHeight.value = clamp(
    localStorage.getItem(LINE_HEIGHT_KEY) || DEFAULT_LINE_HEIGHT,
    MIN_LINE_HEIGHT,
    MAX_LINE_HEIGHT,
  );
  docWidth.value = clamp(
    localStorage.getItem(DOC_WIDTH_KEY) || DEFAULT_DOC_WIDTH,
    MIN_DOC_WIDTH,
    MAX_DOC_WIDTH,
  );
  sidebarCollapsed.value = localStorage.getItem(SIDEBAR_KEY) === "true";
  applyReadingSettings();
  applySidebarState();
  nextTick(syncGroupState);
  document.addEventListener("keydown", closeSettingsOnEscape);
  document.addEventListener("pointerdown", closeSettingsOnOutsideClick);
});

onBeforeUnmount(() => {
  document.removeEventListener("keydown", closeSettingsOnEscape);
  document.removeEventListener("pointerdown", closeSettingsOnOutsideClick);
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

          <div ref="settingsMenu" class="hwm-sidebar-toolbar-end">
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

            <Teleport to="body">
              <Transition name="hwm-settings-fade">
                <section
                  v-if="settingsOpen"
                  ref="settingsPanel"
                  class="hwm-settings-panel"
                  aria-label="阅读与外观设置"
                >
                  <div class="hwm-settings-heading">
                    <strong>阅读与外观</strong>
                    <button type="button" @click="resetReadingSettings">
                      恢复默认
                    </button>
                  </div>

                  <div class="hwm-settings-group">
                    <div class="hwm-settings-label">
                      <span>外观</span>
                      <b>{{ isDark ? "深色" : "浅色" }}</b>
                    </div>
                    <div class="hwm-settings-actions hwm-appearance-actions">
                      <button
                        type="button"
                        :class="{ active: !isDark }"
                        aria-label="切换到浅色模式"
                        title="浅色模式"
                        @click="setAppearance(false)"
                      >
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <circle cx="12" cy="12" r="4" />
                          <path
                            d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
                          />
                        </svg>
                      </button>
                      <button
                        type="button"
                        :class="{ active: isDark }"
                        aria-label="切换到深色模式"
                        title="深色模式"
                        @click="setAppearance(true)"
                      >
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path
                            d="M20.5 14.5A8.4 8.4 0 0 1 9.5 3.5 8.5 8.5 0 1 0 20.5 14.5Z"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div class="hwm-settings-group">
                    <div class="hwm-settings-label">
                      <span>字号</span>
                      <b>{{ fontSize }}px</b>
                    </div>
                    <div class="hwm-settings-actions">
                      <button type="button" @click="adjustFontSize(-1)">
                        A-
                      </button>
                      <button
                        type="button"
                        @click="fontSize = DEFAULT_FONT_SIZE"
                      >
                        默认
                      </button>
                      <button type="button" @click="adjustFontSize(1)">
                        A+
                      </button>
                    </div>
                    <input
                      v-model.number="fontSize"
                      type="range"
                      :min="MIN_FONT_SIZE"
                      :max="MAX_FONT_SIZE"
                      step="1"
                      aria-label="字号"
                    />
                  </div>

                  <div class="hwm-settings-group">
                    <div class="hwm-settings-label">
                      <span>行距</span>
                      <b>{{ Number(lineHeight).toFixed(2) }}</b>
                    </div>
                    <div class="hwm-settings-actions">
                      <button type="button" @click="adjustLineHeight(-0.05)">
                        更紧
                      </button>
                      <button
                        type="button"
                        @click="lineHeight = DEFAULT_LINE_HEIGHT"
                      >
                        默认
                      </button>
                      <button type="button" @click="adjustLineHeight(0.05)">
                        更松
                      </button>
                    </div>
                    <input
                      v-model.number="lineHeight"
                      type="range"
                      :min="MIN_LINE_HEIGHT"
                      :max="MAX_LINE_HEIGHT"
                      step="0.05"
                      aria-label="行距"
                    />
                  </div>

                  <div class="hwm-settings-group">
                    <div class="hwm-settings-label">
                      <span>正文宽度</span>
                      <b>{{ docWidth }}px</b>
                    </div>
                    <div class="hwm-settings-actions">
                      <button type="button" @click="adjustDocWidth(-20)">
                        更窄
                      </button>
                      <button
                        type="button"
                        @click="docWidth = DEFAULT_DOC_WIDTH"
                      >
                        默认
                      </button>
                      <button type="button" @click="adjustDocWidth(20)">
                        更宽
                      </button>
                    </div>
                    <input
                      v-model.number="docWidth"
                      type="range"
                      :min="MIN_DOC_WIDTH"
                      :max="MAX_DOC_WIDTH"
                      step="20"
                      aria-label="正文宽度"
                    />
                  </div>
                </section>
              </Transition>
            </Teleport>
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
</template>
