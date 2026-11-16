"""Check Markdown/notebook math and local links, without creating HTML files."""

from project_paths import ROOT, CODE, NOTEBOOKS, notebook_path, document_path
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit

import mistune
import nbformat



def walk(tokens):
    for token in tokens:
        yield token
        yield from walk(token.get('children', []))


def check(path):
    if path.suffix == '.ipynb':
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        sections = [c.source for c in notebook.cells if c.cell_type == 'markdown']
    else:
        sections = [path.read_text()]
    equations = []
    links = 0
    for source in sections:
        assert source.count('$$') % 2 == 0
        assert '\\[' not in source and '\\]' not in source
        tokens = walk(mistune.create_markdown(renderer='ast', plugins=['math', 'table'])(source))
        for token in tokens:
            if token['type'] in ('inline_math', 'block_math'):
                equations.append(dict(text=token['raw'], display=token['type'] == 'block_math'))
            if token['type'] in ('link', 'image'):
                url = urlsplit(token['attrs']['url'])
                if not url.scheme and not url.netloc and url.path:
                    assert (path.parent/unquote(url.path)).exists(), url.path
                    links += 1
    assert len(equations) > 0
    bundles = sorted((Path.home()/'.vscode-server/bin').glob('*/node'))
    node = next(n for n in bundles if (n.parent/'node_modules/katex/dist/katex.js').exists())
    javascript = """
const fs = require('fs'); const katex = require(process.argv[1]);
const equations = JSON.parse(fs.readFileSync(0,'utf8'));
for (const e of equations) katex.renderToString(e.text,
 {displayMode:e.display,throwOnError:true,strict:'error'});
process.stdout.write(String(equations.length));
"""
    output = subprocess.run([str(node), '-e', javascript,
                             str(node.parent/'node_modules/katex/dist/katex.js')],
                            input=json.dumps(equations), text=True,
                            capture_output=True, check=True)
    assert int(output.stdout) == len(equations)
    result = dict(file=path.name, equations=len(equations),
                  display_equations=sum(e['display'] for e in equations),
                  local_links_checked=links, html_files_written=0)
    print('MARKDOWN CHECK', result)
    return result


if __name__ == '__main__':
    reports = [check(document_path(name)) for name in
               ['README.md', 'conditional_gaussian_gates.ipynb',
                'fast_conditional_displacement.ipynb', 'coupler_phase_transients.ipynb',
                'higher_order_cd_and_squeezing.ipynb','phase_controlled_cd.ipynb']]
    assert len(reports) == 6
    (ROOT/'runs/audit/document_checks.json').write_text(json.dumps(reports, indent=2)+'\n')
    print('FINAL DOCUMENT COUNTS', len(reports), 'documents;',
          sum(r['equations'] for r in reports), 'equations;',
          sum(r['local_links_checked'] for r in reports), 'local links')
