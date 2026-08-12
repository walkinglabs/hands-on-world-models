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

    def test_f0_is_valid_and_all_code_cells_run(self):
        self.execute_notebook(
            "notebooks/00_reinvent/F0-invent-a-world-model.ipynb"
        )

    def test_foundation_notebooks_run(self):
        paths = [
            "notebooks/01_foundations/F1-see-remember-compress.ipynb",
            "notebooks/01_foundations/F2-space-plan-train.ipynb",
            "notebooks/02_first_model/F3-learn-a-table-world.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    def test_z0_notebook_runs(self):
        self.execute_notebook(
            "notebooks/08_evaluation/Z0-test-a-world-model.ipynb"
        )

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "安装 requirements-neural.txt 后运行神经 Notebook smoke",
    )
    def test_route_a_notebooks_run(self):
        paths = [
            "notebooks/03_decision/A1-learn-a-latent-world.ipynb",
            "notebooks/03_decision/A2-act-in-imagination.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "安装 requirements-neural.txt 后运行路线 B/C Notebook smoke",
    )
    def test_routes_bc_notebooks_run(self):
        paths = [
            "notebooks/04_interactive_video/B1-compress-and-predict-video.ipynb",
            "notebooks/04_interactive_video/B2-make-video-controllable.ipynb",
            "notebooks/05_jepa/C1-learn-video-features.ipynb",
            "notebooks/05_jepa/C2-test-and-control-features.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "安装 requirements-neural.txt 后运行路线 D/E Notebook smoke",
    )
    def test_routes_de_notebooks_run(self):
        paths = [
            "notebooks/06_robot/D1-build-a-tiny-vla.ipynb",
            "notebooks/06_robot/D2-check-actions-before-moving.ipynb",
            "notebooks/07_spatial/E1-from-camera-to-space.ipynb",
            "notebooks/07_spatial/E2a-build-a-small-4d-world.ipynb",
            "notebooks/07_spatial/E2b-predict-driving-space.ipynb",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.execute_notebook(path)


if __name__ == "__main__":
    unittest.main()
