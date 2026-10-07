"""Execute and save the notebook using the isolated ZoomMap research kernel."""

from pathlib import Path
import argparse
import os

import nbformat
from nbclient import NotebookClient


PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("notebook", nargs="?", type=Path,
                    default=PROJECT / "notebooks" / "general_example.ipynb",
                    help="Notebook to execute (default: notebooks/general_example.ipynb)")
NOTEBOOK = parser.parse_args().notebook.resolve()
if not NOTEBOOK.is_file() or NOTEBOOK.suffix != ".ipynb":
    parser.error(f"Notebook not found: {NOTEBOOK}")
(PROJECT / "data").mkdir(exist_ok=True)
for name, folder in [("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")]:
    os.environ.setdefault(name, str(PROJECT / "data" / folder))

notebook = nbformat.read(NOTEBOOK, as_version=4)


def started(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)


class StreamingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


client = StreamingNotebookClient(
    notebook, timeout=1800, kernel_name="zoommap",
    resources={"metadata": {"path": str(PROJECT)}},
    on_cell_start=started,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, NOTEBOOK)
print(f"Executed and saved {NOTEBOOK}", flush=True)
