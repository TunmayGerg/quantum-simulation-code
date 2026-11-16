"""Execute and save both notebooks with counted code cells and embedded figures."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import argparse,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def execute(name):
    path=notebook_path(name);notebook=nbformat.read(path,4)
    expected=sum(c.cell_type=='code' for c in notebook.cells);completed=[]
    def start(cell,cell_index,**kwargs):
        if cell.cell_type=='code':print('START',name,'cell',cell_index,flush=True)
    def finish(cell,cell_index,**kwargs):
        if cell.cell_type!='code':return
        for output in cell.get('outputs',[]):
            if output.output_type=='error':
                print('\n'.join(output.get('traceback',[])),flush=True)
                raise AssertionError(f'{name} cell {cell_index}: {output.evalue}')
            if output.output_type=='stream':print(output.text.rstrip(),flush=True)
        completed.append(cell_index)
        print('COMPLETED',name,'cell',cell_index,len(completed),'/',expected,flush=True)
    client=NotebookClient(notebook,kernel_name='python3',timeout=1800,
                          resources={'metadata':{'path':str(path.parent)}},
                          on_cell_start=start,on_cell_executed=finish)
    client.execute()
    assert len(completed)==expected and expected>0
    figures=sum(any(key.startswith('image/') for key in output.get('data',{}))
                for c in notebook.cells if c.cell_type=='code' for output in c.outputs)
    minimum=6 if name=='coupler_phase_transients.ipynb' else 5
    assert figures>=minimum,(name,figures)
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
    nbformat.validate(notebook);nbformat.write(notebook,path)
    result=dict(file=name,cells=len(notebook.cells),executed_code_cells=expected,
                embedded_figures=figures,errors=0)
    print('EXECUTED NOTEBOOK',result,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks',nargs='*',default=['conditional_gaussian_gates.ipynb','fast_conditional_displacement.ipynb'])
    results=[execute(name) for name in parser.parse_args().notebooks]
    manifest=ROOT/'runs/audit/notebook_execution.json'
    old=json.loads(manifest.read_text()) if manifest.exists() else []
    combined={row['file']:row for row in old}
    combined.update({row['file']:row for row in results})
    manifest.write_text(json.dumps(list(combined.values()),indent=2)+'\n')
