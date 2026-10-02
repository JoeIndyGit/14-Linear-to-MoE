"""Validate and execute notebooks in a fresh, explicitly selected Python kernel."""
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute_notebook(notebook, workdir, timeout=180):
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="linear-moe-kernel-") as spec_dir:
        spec = Path(spec_dir) / "linear-moe"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Linear to MoE (current Python)", "language": "python"}))
        class LocalKernelManager(AsyncKernelManager):
            def __init__(self, **kwargs):
                kwargs.update(kernel_spec_manager=KernelSpecManager(kernel_dirs=[spec_dir]),
                    transport="tcp" if os.name == "nt" else "ipc",
                    ip="127.0.0.1" if os.name == "nt" else str(Path(spec_dir) / "kernel"))
                super().__init__(**kwargs)

        client = NotebookClient(notebook, kernel_name="linear-moe",
            kernel_manager_class=LocalKernelManager, timeout=timeout,
            resources={"metadata": {"path": str(Path(workdir).resolve())}},
            allow_errors=False, record_timing=True)
        client.execute()
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count == i for i, c in enumerate(code_cells, 1))
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    return notebook
