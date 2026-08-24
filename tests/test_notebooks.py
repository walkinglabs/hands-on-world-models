import json
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NotebookSmokeTest(unittest.TestCase):
    def execute_notebook(self, relative_path):
        path = ROOT / relative_path
        notebook = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)

        namespace = {"__name__": "__notebook_smoke__"}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            code = compile(source, f"{path.name}:cell-{index}", "exec")
            with redirect_stdout(StringIO()):
                exec(code, namespace)

    def test_reinvent_notebook_runs(self):
        self.execute_notebook(
            "notebooks/01_reinvent/invent-a-world-model.ipynb"
        )

    def test_foundation_notebooks_run(self):
        paths = [
            "notebooks/02_foundations/see-remember-compress.ipynb",
            "notebooks/02_foundations/space-plan-train.ipynb",
            "notebooks/03_data/learn-a-table-world.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    def test_evaluation_notebook_runs(self):
        self.execute_notebook(
            "notebooks/09_evaluation/test-a-world-model.ipynb"
        )

    def test_project_templates_are_valid(self):
        paths = [
            "notebooks/projects/learnable-world-template.ipynb",
            "notebooks/projects/route-template.ipynb",
            "notebooks/projects/next-model-template.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "安装 requirements-neural.txt 后运行神经 Notebook smoke",
    )
    def test_decision_notebooks_run(self):
        paths = [
            "notebooks/04_decision/learn-a-latent-world.ipynb",
            "notebooks/04_decision/act-in-imagination.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "安装 requirements-neural.txt 后运行第 5、6 章 Notebook smoke",
    )
    def test_video_and_jepa_notebooks_run(self):
        paths = [
            "notebooks/05_interactive_video/compress-and-predict-video.ipynb",
            "notebooks/05_interactive_video/make-video-controllable.ipynb",
            "notebooks/06_jepa/learn-video-features.ipynb",
            "notebooks/06_jepa/test-and-control-features.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "安装 requirements-neural.txt 后运行第 7、8 章 Notebook smoke",
    )
    def test_robot_and_spatial_notebooks_run(self):
        paths = [
            "notebooks/07_robot/build-a-tiny-vla.ipynb",
            "notebooks/07_robot/check-actions-before-moving.ipynb",
            "notebooks/08_spatial/from-camera-to-space.ipynb",
            "notebooks/08_spatial/build-a-small-4d-world.ipynb",
            "notebooks/08_spatial/predict-driving-space.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)


if __name__ == "__main__":
    unittest.main()
