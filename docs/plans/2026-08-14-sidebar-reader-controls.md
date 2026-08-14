# Sidebar Reader Controls Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring the unified sidebar collapse control and reader settings experience from hands-on-modern-rl into the world-models documentation site.

**Architecture:** Keep the existing VitePress custom layout and its localStorage-backed preferences. Replace its simple settings card with an accessible anchored popover, and make the sidebar toolbar use the same compact unified-control visual pattern as the reference site. No new runtime dependency is necessary.

**Tech Stack:** VitePress, Vue 3 Composition API, CSS custom properties, browser localStorage.

---

### Task 1: Implement the reader-control behavior

**Files:**

- Modify: `docs/.vitepress/theme/Layout.vue`

**Step 1: Add the required reactive state and helpers**

Implement light/dark controls plus bounded increment, reset, and range handlers for font size, line height, and content width. Retain the existing localStorage keys and Escape-to-close behavior.

**Step 2: Replace the settings card markup**

Render the controls in a compact, labelled popover anchored to the sidebar settings button. Provide keyboard-accessible buttons and labels for every control.

**Step 3: Check client-side behavior**

Run `npm run build`; manually confirm the Vue compiler accepts the template and state bindings.

### Task 2: Match the sidebar toolbar treatment

**Files:**

- Modify: `docs/.vitepress/theme/style.css`

**Step 1: Update toolbar and popover styles**

Use the reference layout’s compact single-row grouping, subtle border and hover states, and light/dark surfaces while preserving this site’s teal brand token.

**Step 2: Implement responsive behavior**

Keep controls hidden on the VitePress mobile layout and constrain the popover to the viewport.

**Step 3: Check built CSS**

Run `npm run build` and inspect the generated CSS output for the reader-control selectors.

### Task 3: Verify and deliver

**Files:**

- Verify: `docs/.vitepress/theme/Layout.vue`
- Verify: `docs/.vitepress/theme/style.css`

**Step 1: Run formatting verification**

Run `npm run format:check` and format only the files changed for this feature if needed.

**Step 2: Build the documentation site**

Run `npm run build` and ensure it exits successfully.

**Step 3: Commit and push**

Stage only the reader-control implementation and this plan, create a focused commit, then push `main` to `origin`.
