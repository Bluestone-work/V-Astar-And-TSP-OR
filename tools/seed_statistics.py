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

def collect_continuous(evaluation_dirs):
    """Aggregate one Table XI(b) condition from the three complete seed runs."""
    results = [json.loads((Path(directory) / 'results.json').read_text())
               for directory in evaluation_dirs]
    results.sort(key=lambda row: row['protocol']['training_seed'])
    protocols = [dict(row['protocol']) for row in results]
    if [protocol.pop('training_seed') for protocol in protocols] != [0, 1, 2]:
        raise ValueError('Table XI(b) requires checkpoint labels 0, 1 and 2')
    if len({row['checkpoint_sha256'] for row in results}) != 3:
        raise ValueError('Table XI(b) requires three distinct checkpoints')
    for protocol in protocols:
        for key in ('checkpoint', 'output_dir'):
            protocol.pop(key)
    reference = protocols[0]
    if (not reference['group'] or reference['scenarios'] != 300
            or any(protocol != reference for protocol in protocols[1:])):
        raise ValueError('Use the same Table XI(b) group and 300 matched scenarios per seed')
    summaries = [row['summary'] for row in results]
    if any(summary['scenarios'] != 300 for summary in summaries):
        raise ValueError('Incomplete Table XI(b) evaluation')
    metrics = ('success_rate', 'collision_rate', 'planning_failure_rate',
               'refusal_rate', 'mean_tracking_error', 'timeout_rate')
    pooled_counts = {}
    for summary in summaries:
        for label, count in summary['terminations'].items():
            pooled_counts[label] = pooled_counts.get(label, 0) + count
    total = sum(summary['scenarios'] for summary in summaries)
    return {
        'group': reference['group'],
        'cross_seed': {name: seed_summary([row[name] for row in summaries]) for name in metrics},
        'pooled_terminations': pooled_counts,
        'pooled_static_collision_rate': pooled_counts.get('collision_static', 0) / total,
        'pooled_dynamic_collision_rate': pooled_counts.get('collision_dynamic', 0) / total,
        'pooled_false_arrival_rate': pooled_counts.get('false_arrival', 0) / total,
    }


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
                 'mean_wall_time_all_s','mean_wall_time_success_s','decision_latency_ms',
                 'cycle_latency_excluding_replans_ms','cycle_latency_including_replans_ms',
                 'mean_steps_all','mean_steps_success','mean_path_length_success','mean_smoothness_success',
                 'deadlock_recovery_rate','deadlock_recoveries_per_episode')
        result['execution']={}
        for level,entries in bylevel.items():
            if sorted(s for s,_ in entries)!=[0,1,2]:raise ValueError('Require training seeds 0,1,2')
            summaries=[v for _,v in sorted(entries)]
            result['execution'][level]={m:seed_summary([v.get(m) for v in summaries]) for m in metrics}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-runs',nargs=3)
    p.add_argument('--evaluation-dirs',nargs=3)
    p.add_argument('--continuous-dirs', nargs=3, help='Three complete seed runs for one Table XI(b) group')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not (a.training_runs or a.evaluation_dirs or a.continuous_dirs):
        p.error('Provide training runs or evaluation directories')
    result = collect(a.training_runs, a.evaluation_dirs)
    if a.continuous_dirs:
        result['continuous'] = collect_continuous(a.continuous_dirs)
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
