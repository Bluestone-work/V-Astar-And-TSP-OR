"""Run all four priority × IS cells with three fresh training seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CELLS = {'per_no_is': ('per', 'off'), 'eq19_is': ('manuscript', 'on'),
         'per_is': ('per', 'on'), 'eq19_no_is': ('manuscript', 'off')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/factorial')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError('Use a new output directory to avoid mixing experiments')
    output.mkdir(parents=True)
    jobs = []
    for seed in (0, 1, 2):
        for cell, (rule, is_mode) in CELLS.items():
            tag = '%s_seed%d' % (cell, seed)
            checkpoint = output/'training'/tag/'final.pth'
            train = [sys.executable, '-u', 'train.py', '--map-file', 'grid_map2.txt',
                     '--seed', str(seed), '--device', args.device, '--replay-mode', rule,
                     '--importance-sampling', is_mode, '--save-path', str(checkpoint),
                     '--log-dir', str(checkpoint.parent), '--checkpoint-interval', '200',
                     '--checkpoint-dir', str(checkpoint.parent/'checkpoints')]
            evaluate = [sys.executable, '-u', '-m', 'experiments.proposed_evaluation',
                        '--map-file', 'grid_map2.txt', '--training-map', 'grid_map2.txt',
                        '--checkpoint', str(checkpoint), '--seed', '0', '--scenarios', '300',
                        '--eval-seed', '10000', '--max-steps', '2000', '--levels', '1', '2', '3', '4',
                        '--levels-config', 'configs/levels.json', '--replay-mode', rule,
                        '--device', 'cpu', '--output-dir', str(output/'evaluations'/tag)]
            jobs.append(dict(cell=cell, seed=seed, train=train, evaluate=evaluate))
    (output/'plan.json').write_text(json.dumps(dict(jobs=jobs,
        config_sha256=hashlib.sha256((ROOT/'configs/method.json').read_bytes()).hexdigest(),
        protocol='jas0644-response-v3; all four cells freshly trained; historical results not reused'),indent=2))
    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
    def run(job):
        tag='%s_seed%d'%(job['cell'],job['seed'])
        for phase in ('train','evaluate'):
            with (output/(tag+'_'+phase+'.log')).open('w') as log:
                subprocess.run(job[phase],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed',tag,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run,jobs))
    subprocess.run([sys.executable,'tools/summarize_factorial.py',str(output)],cwd=ROOT,check=True)


if __name__=='__main__':main()
