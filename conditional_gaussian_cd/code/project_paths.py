"""Filesystem locations shared by the simulations and notebook builders."""
from pathlib import Path
import os
import re
from urllib.parse import urlsplit,urlunsplit

CODE=Path(__file__).resolve().parent
ROOT=CODE.parent
NOTEBOOKS=ROOT/'notebooks'

NOTEBOOK_ALIASES={
    'higher_order_cd_and_squeezing.ipynb':'archive/modulated_swap200.ipynb',
    'coupler_phase_transients.ipynb':'archive/coupler_phase_transients.ipynb',
    'runs/audit/pre_swap200/higher_order_cd_and_squeezing.ipynb':'archive/modulated_strong_sideband.ipynb',
    'runs/audit/pre_story_notebooks/conditional_gaussian_gates.ipynb':'archive/earlier_gaussian_gates.ipynb',
    'runs/audit/pre_story_notebooks/fast_conditional_displacement.ipynb':'archive/earlier_cd.ipynb',
}


def notebook_path(name):
    """Resolve a current or historical notebook name after the folder move."""
    name=str(name)
    if Path(name).is_absolute():return Path(name)
    if name.startswith('notebooks/'):return ROOT/name
    return NOTEBOOKS/NOTEBOOK_ALIASES.get(name,name)


def document_path(name):
    return notebook_path(name) if str(name).endswith('.ipynb') else ROOT/name


BOOTSTRAP=r'''# Locate code/ when this notebook is opened from notebooks/ or archive/.
from pathlib import Path as _Path
import sys as _sys
_project=next(p for p in [_Path.cwd(),*_Path.cwd().parents]
              if (p/'code/project_paths.py').is_file())
if str(_project/'code') not in _sys.path:
    _sys.path.insert(0,str(_project/'code'))
from project_paths import ROOT as PROJECT_ROOT, CODE, NOTEBOOKS
'''


def prepare_notebook(notebook,path):
    """Keep imports and relative document links valid in the notebook folder."""
    first=next(c for c in notebook.cells if c.cell_type=='code')
    if '# Locate code/' not in first.source:first.source=BOOTSTRAP+'\n'+first.source
    for cell in notebook.cells:
        if cell.cell_type=='code':
            cell.source=cell.source.replace('ROOT=Path.cwd()','ROOT=PROJECT_ROOT')
            cell.source=cell.source.replace('ROOT = Path.cwd()','ROOT = PROJECT_ROOT')
        elif cell.cell_type=='markdown':
            def link(match):
                url=urlsplit(match.group(1))
                if url.scheme or url.netloc or not url.path:return match.group(0)
                if url.path.startswith(('runs/','code/')) or url.path in ('README.md','WORK_STATUS.md'):target=ROOT/url.path
                elif url.path.endswith('.ipynb') and not url.path.startswith('../'):target=notebook_path(url.path)
                else:return match.group(0)
                relative=os.path.relpath(target,Path(path).parent)
                return ']('+urlunsplit(('', '', relative, url.query, url.fragment))+')'
            cell.source=re.sub(r'\]\(([^\s)]+)\)',link,cell.source)
    return notebook
