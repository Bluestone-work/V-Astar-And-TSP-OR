"""Three-seed sampling fractions and execution metrics without pooling seeds."""
import argparse
import json
from pathlib import Path
import statistics


def seed_summary(values):
    values=[None if x is None else float(x) for x in values]
    complete=len(values)==3 and all(x is not None for x in values)
    return dict(seed_values=values,mean=statistics.mean(values) if complete else None,
                sample_sd=statistics.stdev(values) if complete else None,
                defined_seeds=sum(x is not None for x in values),
                note='Equal-weight three-training-seed summary; ddof=1; undefined rates remain null')


def collect(training_runs=None,evaluation_dirs=None):
    result={}
    if training_runs:
        counters=[];seeds=[]
        for run in map(Path,training_runs):
            metadata=json.loads((run/'run_metadata.json').read_text());seeds.append(metadata['seed'])
            counters.append(metadata['sampled_event_counts'])
        if sorted(seeds)!=[0,1,2]:raise ValueError('Require independent training seeds 0,1,2')
        pairs=sorted(zip(seeds,counters));counters=[c for _,c in pairs]
        total=[sum(c.values()) for c in counters]
        result['sampling']={event:dict(**seed_summary([c.get(event,0)/n if n else None
            for c,n in zip(counters,total)]),pooled_fraction=(sum(c.get(event,0) for c in counters)/sum(total)
            if sum(total) else None)) for event in ('collision','success','waypoint_arrival')}
    if evaluation_dirs:
        bylevel={}
        for directory in map(Path,evaluation_dirs):
            if not (directory/'evaluation_complete.json').exists():raise ValueError('Incomplete evaluation')
            for level in (1,2,3,4):
                item=json.loads((directory/('level%d_summary.json'%level)).read_text())
                protocol=item['protocol']
                bylevel.setdefault(level,[]).append((protocol['training_seed'],item['summary']))
        metrics=('success_rate','task_throughput_per_wall_s','mean_global_replans','mean_planning_seconds',
                 'mean_steps_all','mean_steps_success','mean_path_length_success','mean_smoothness_success',
                 'deadlock_recovery_rate','deadlock_recoveries_per_episode')
        result['execution']={}
        for level,entries in bylevel.items():
            if sorted(s for s,_ in entries)!=[0,1,2]:raise ValueError('Require training seeds 0,1,2')
            summaries=[v for _,v in sorted(entries)]
            result['execution'][level]={m:seed_summary([v[m] for v in summaries]) for m in metrics}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-runs',nargs=3)
    p.add_argument('--evaluation-dirs',nargs=3)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not (a.training_runs or a.evaluation_dirs):p.error('Provide training runs and/or evaluation directories')
    Path(a.output).write_text(json.dumps(collect(a.training_runs,a.evaluation_dirs),indent=2,allow_nan=False)+'\n')
