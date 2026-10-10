"""Thực thi notebook C3 bằng chính Python hiện tại; không tự train lại."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from src.paths import ROOT
path = ROOT/'notebooks/05_distilbert_huy.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=600, kernel_name='python3',
    resources={'metadata':{'path':str(ROOT)}}, allow_errors=False)
client.on_cell_start = lambda cell, cell_index: print(f'Cell {cell_index+1}/{len(notebook.cells)}: {cell.cell_type}', flush=True)
client.km = client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client.execute()
nbformat.validate(notebook)
executed = ROOT/'reports/notebooks/05_distilbert_huy.executed.ipynb'
executed.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, executed)
body, _ = HTMLExporter().from_notebook_node(notebook)
output = ROOT/'reports/c3_distilbert/distilbert_huy.html'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(body, encoding='utf-8')
print('PASS: notebook executed and HTML exported.')
