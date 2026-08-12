import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NotebookSmokeTest(unittest.TestCase):
    def test_f0_is_valid_and_all_code_cells_run(self):
        path = ROOT / "notebooks/00_reinvent/F0-invent-a-world-model.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 20)

        namespace = {"__name__": "__notebook_smoke__"}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            code = compile(source, f"{path.name}:cell-{index}", "exec")
            exec(code, namespace)


if __name__ == "__main__":
    unittest.main()
