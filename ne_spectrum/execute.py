"""Execute the NE-spectrum notebook in its separate upstream-openTSNE kernel."""

from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient

EXPERIMENT = Path(__file__).resolve().parent
NOTEBOOK = EXPERIMENT / "general_example.ipynb"
for name, folder in [("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")]:
    os.environ.setdefault(name, str(EXPERIMENT / "data" / folder))

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
    notebook, timeout=1800, kernel_name="zoommap-ne-spectrum",
    resources={"metadata": {"path": str(EXPERIMENT)}}, on_cell_start=started,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, NOTEBOOK)
print(f"Executed and saved {NOTEBOOK}", flush=True)
