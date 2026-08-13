import DefaultTheme from "vitepress/theme";
import { h } from "vue";
import MaintenanceBanner from "./MaintenanceBanner.vue";
import "./style.css";

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      "layout-top": () => h(MaintenanceBanner),
    }),
};
