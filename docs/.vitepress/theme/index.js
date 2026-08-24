import DefaultTheme from "vitepress/theme";
import Layout from "./Layout.vue";
import PlayWorldModel from "./components/PlayWorldModel.vue";
import "./style.css";

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component("PlayWorldModel", PlayWorldModel);
  },
};
