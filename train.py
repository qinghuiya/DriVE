"""The single public experiment entry point."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
import yaml

from fedhodge.config import load_config
from fedhodge.engine.engine import METHODS
from fedhodge.engine.experiment import run_experiment
from fedhodge.settings import DEFAULTS, merge_config, validate_config


def main():
    p = argparse.ArgumentParser(description='Run one federated learning experiment.')
    p.add_argument('--config', type=Path, help='Optional YAML overrides; otherwise use built-in defaults')
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--dataset', choices=('cifar10', 'cifar100', 'tinyimagenet', 'synthetic','urbansound8k','shakespeare'))
    p.add_argument('--model', choices=('auto','resnet18', 'vit_tiny', 'tinycnn','audio_cnn','char_lstm'))
    p.add_argument('--data-root')
    p.add_argument('--output', type=Path, help='A NEW output directory; existing paths are refused')
    p.add_argument('--device', help='auto, cpu, cuda, or cuda:<index>')
    p.add_argument('--seed', type=int)
    p.add_argument('--split-seed', type=int, help='Independent fixed holdout seed, shared across training seeds')
    p.add_argument('--partition-seed', type=int, help='Client partition seed, separate from the holdout')
    p.add_argument('--phase', choices=('tune','confirm'), help='tune: validation only; confirm: validation and test at fixed evaluation rounds')
    p.add_argument('--rounds', type=int)
    p.add_argument('--num-clients', type=int)
    cohort = p.add_mutually_exclusive_group()
    cohort.add_argument('--clients-per-round', type=int)
    cohort.add_argument('--participation', type=float, help='Fraction in (0,1], rounded to a client count')
    local = p.add_mutually_exclusive_group()
    local.add_argument('--local-steps', type=int)
    local.add_argument('--local-epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--partition', choices=('iid', 'dirichlet'))
    p.add_argument('--alpha', type=float)
    p.add_argument('--lr', type=float)
    p.add_argument('--momentum', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--p', type=int, choices=(0, 1, 2), help='DriVE polynomial order')
    p.add_argument('--window', type=int, help='DriVE window size W')
    p.add_argument('--lambda-v', type=float, help='DriVE virtual integration horizon')
    p.add_argument('--eval-every', type=int)
    p.add_argument('--save-model', action='store_true')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='Override any known dotted setting, e.g. baseline.proximal_mu=0.1')
    p.add_argument('--dry-run', action='store_true', help='Validate and print config without loading data or training')
    a = p.parse_args()
    c = copy.deepcopy(DEFAULTS)
    try:
        if a.config:
            patch = load_config(a.config)
            if patch.get('training', {}).get('local_epochs') is not None and 'local_steps' not in patch['training']:
                c['training']['local_steps'] = None
            merge_config(c, patch)
        mapping = {'method': ('algorithm', 'name'), 'dataset': ('dataset', 'name'),
                   'model': ('model', 'name'), 'data_root': ('dataset', 'data_root'),
                   'device': ('experiment', 'device'), 'seed': ('experiment', 'seed'),
                   'phase': ('experiment','phase'), 'split_seed': ('dataset','split_seed'),
                   'partition_seed': ('dataset','partition_seed'),
                   'rounds': ('training', 'num_rounds'), 'num_clients': ('dataset', 'num_clients'),
                   'clients_per_round': ('training', 'clients_per_round'),
                   'batch_size': ('training', 'batch_size'), 'eval_every': ('training', 'eval_every'),
                   'p': ('drive', 'poly_order'), 'window': ('drive', 'window_size'),
                   'lambda_v': ('drive', 'lambda_v')}
        for flag, (section, key) in mapping.items():
            if getattr(a, flag) is not None:
                c[section][key] = getattr(a, flag)
        for flag in ('lr', 'momentum', 'weight_decay'):
            if getattr(a, flag) is not None:
                c['algorithm']['client_optimizer'][flag] = getattr(a, flag)
        for flag in ('local_steps', 'local_epochs'):
            if getattr(a, flag) is not None:
                c['training'][flag] = getattr(a, flag)
                c['training']['local_epochs' if flag == 'local_steps' else 'local_steps'] = None
        for flag in ('partition', 'alpha'):
            if getattr(a, flag) is not None:
                c['dataset']['partition']['type' if flag == 'partition' else flag] = getattr(a, flag)
        if a.save_model:
            c['experiment']['save_model'] = True
        for override in a.set:
            if '=' not in override:
                raise ValueError('--set requires KEY=VALUE')
            key, value = override.split('=', 1)
            patch = yaml.safe_load(value)
            for part in reversed(key.split('.')):
                patch = {part: patch}
            merge_config(c, patch)
        if a.participation is not None:
            if not 0 < a.participation <= 1:
                raise ValueError('participation must be in (0, 1]')
            c['training'].update(participation=a.participation,clients_per_round=None)
        elif a.clients_per_round is not None:
            c['training']['participation'] = None
        validate_config(c)
        if c['experiment']['phase']=='confirm' and a.config is None:
            raise ValueError('Confirmation requires --config pointing to the configuration selected on validation data')
    except (ValueError, TypeError, OSError, yaml.YAMLError) as exc:
        p.error(str(exc))
    if a.dry_run:
        print(yaml.safe_dump(c, sort_keys=False))
        return
    torch.set_num_threads(c['experiment']['threads'])
    output = a.output or Path('runs')/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    try:
        summary = run_experiment(c, output)
    except FileExistsError:
        p.error(f'Output directory already exists: {output}; choose a new path')
    print(json.dumps(summary, indent=2))
    print('Output:', output.resolve())


if __name__ == '__main__':
    main()
