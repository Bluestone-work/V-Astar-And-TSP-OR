"""Seed-level means, sample SD and paired priority/IS effects; no pooled-seed SD."""
import csv
import json
from pathlib import Path
import statistics as st
import sys


def summarize(root):
    cells=('per_no_is','eq19_is','per_is','eq19_no_is')
    values={}
    raw=[]
    for cell in cells:
        for seed in (0,1,2):
            directory=root/'evaluations'/('%s_seed%d'%(cell,seed))
            if not (directory/'evaluation_complete.json').exists():
                raise RuntimeError('All 12 evaluations must be complete before inference: '+str(directory))
            for level in (1,2,3,4):
                rows=[json.loads(s) for s in (directory/('level%d.jsonl'%level)).read_text().splitlines()]
                if len(rows)!=300 or [r['scenario_seed'] for r in rows]!=list(range(10000,10300)):
                    raise ValueError('Require all 300 ordered held-out scenes: '+str(directory))
                result=dict(cell=cell,seed=seed,level=level,
                    SR=100*sum(r['success'] for r in rows)/300,
                    CR=100*sum(r['termination']=='collision' for r in rows)/300)
                values[(cell,seed,level)]=result
                raw.append(result)
    def stat(v):return dict(mean=st.mean(v),sample_sd=st.stdev(v),seed_values=v)
    means=[dict(cell=c,level=L,**{m:stat([values[(c,s,L)][m] for s in (0,1,2)])
           for m in ('SR','CR')}) for c in cells for L in (1,2,3,4)]
    contrasts=[]
    for L in (1,2,3,4):
        comparisons=[('Eq19-PER | IS off',{'eq19_no_is':1,'per_no_is':-1}),
                     ('Eq19-PER | IS on',{'eq19_is':1,'per_is':-1}),
                     ('IS on-off | PER',{'per_is':1,'per_no_is':-1}),
                     ('IS on-off | Eq19',{'eq19_is':1,'eq19_no_is':-1}),
                     ('Interaction',{'eq19_is':1,'eq19_no_is':-1,'per_is':-1,'per_no_is':1})]
        for name,coefficients in comparisons:
            for metric in ('SR','CR'):
                v=[sum(k*values[(c,s,L)][metric] for c,k in coefficients.items()) for s in (0,1,2)]
                result=stat(v);h=4.302652729911275*result['sample_sd']/(3**.5)
                contrasts.append(dict(contrast=name,level=L,metric=metric,**result,
                    ci95_low=result['mean']-h,ci95_high=result['mean']+h))
    output=dict(seed_results=raw,cells=means,paired_contrasts=contrasts,
        note='Sample SD ddof=1; paired seed differences; df=2 t intervals, exploratory and unadjusted for multiple comparisons')
    (root/'summary.json').write_text(json.dumps(output,indent=2))
    with (root/'seed_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    return output


if __name__=='__main__':summarize(Path(sys.argv[1]).resolve())
