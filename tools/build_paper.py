"""Build Word and XeLaTeX main-text artifacts from paper/method.md."""
from pathlib import Path
from docx import Document
from docx.shared import RGBColor

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'paper/method.md').read_text()
doc=Document()
tex=[r'\documentclass[11pt]{article}',r'\usepackage[margin=1in]{geometry}',
     r'\usepackage{fontspec}',r'\setmainfont{DejaVu Serif}',r'\begin{document}']
escapes={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
         '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
def latex(s):return ''.join(escapes.get(c,c) for c in s.replace('`',''))
in_scope=False
for block in source.strip().split('\n\n'):
    if block.startswith('# '):
        title=block[2:];doc.add_heading(title,0);tex.append(r'\section*{'+latex(title)+'}')
    elif block.startswith('## '):
        heading=block[3:];in_scope=heading=='Reproducibility scope'
        doc.add_heading(heading,1);tex.append(r'\subsection*{'+latex(heading)+'}')
    else:
        paragraph=doc.add_paragraph();run=paragraph.add_run(block.replace('`',''))
        run.font.color.rgb=RGBColor.from_string('C06000' if in_scope or block.startswith('This text') else '1756A9')
        tex.append(latex(block)+'\n')
doc.core_properties.title='Response-aligned navigation method — jas0644-response-v3'
doc.core_properties.author='Bluestone-work'
doc.core_properties.last_modified_by='Bluestone-work'
doc.save(ROOT/'paper/Method_response_aligned.docx')
tex.append(r'\end{document}')
(ROOT/'paper/method.tex').write_text('\n'.join(tex)+'\n')
