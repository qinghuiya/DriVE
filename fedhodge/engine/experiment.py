"""Run a single experiment; no sweep, queue, selection, or plotting logic."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

from fedhodge.config import save_config
from fedhodge.data.factory import build_data
from fedhodge.models.factory import create_model
from fedhodge.settings import resolve_config
from fedhodge.utils.seed import seed_everything
from .engine import FederatedEngine


def run_experiment(config, output_dir):
    config = resolve_config(config)
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # Existing results are never overwritten.
    save_config(config, output/'config.yaml')
    seed_everything(config['experiment']['seed'], deterministic=config['experiment']['deterministic'])
    bundle = build_data(config)
    (output/'data_split.json').write_text(json.dumps({**bundle.manifest,
        'canonical_manifest_path':bundle.manifest_path},indent=2,ensure_ascii=False),encoding='utf8')
    model = create_model(config['model']['name'], num_classes=bundle.num_classes, **bundle.model_kwargs)
    engine = FederatedEngine(model, bundle.client_loaders, config)
    rounds = config['training']['num_rounds']
    eval_every = config['training']['eval_every']
    evaluations = {'validation': []}
    if config['experiment']['phase']=='confirm': evaluations['test'] = []
    with (output/'metrics.jsonl').open('x', encoding='utf8') as f:
        for r in range(rounds):
            result = engine.train_one_round(r)
            row = {'round': r+1, 'selected_clients': result.selected_clients,
                   'client_samples': result.client_samples, **result.metrics}
            if r == 0 or (r+1) % eval_every == 0 or r+1 == rounds:
                validation = engine.evaluate(bundle.validation_loader)
                row.update({f'val_{key}':value for key,value in validation.items()})
                evaluations['validation'].append((r+1,validation['accuracy']))
                status = f"round {r+1}/{rounds} val_accuracy={validation['accuracy']:.4f}"
                if config['experiment']['phase']=='confirm':
                    if bundle.test_loader is None: raise RuntimeError('Confirmation requires a held-out test loader')
                    test = engine.evaluate(bundle.test_loader)
                    row.update({f'test_{key}':value for key,value in test.items()})
                    evaluations['test'].append((r+1,test['accuracy']))
                    status += f" test_accuracy={test['accuracy']:.4f}"
                print(status,flush=True)
            f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
    summary = {'completed_rounds': rounds, 'method': config['algorithm']['name'],
               'seed': config['experiment']['seed'], 'phase':config['experiment']['phase'],
               'split_seed':config['dataset']['split_seed'],'split_sha256':bundle.manifest['split']['sha256'],
               'num_clients_used':len(bundle.client_loaders),'clients_per_round':engine.clients_per_round,
               'python': platform.python_version(),
               'torch': str(torch.__version__), 'device': str(engine.device)}
    boundary = rounds*(1-config['training']['late_fraction'])
    for split,values in evaluations.items():
        late = [accuracy for real_round,accuracy in values if real_round>boundary]
        summary[split] = {'final_accuracy':values[-1][1],
                          'late_mean_accuracy':sum(late)/len(late),
                          'late_evaluation_count':len(late),'late_round_exclusive_boundary':boundary}
    # Hyperparameter or checkpoint selection may use validation statistics only.
    summary['validation']['best_accuracy'] = max(accuracy for _,accuracy in evaluations['validation'])
    if config['experiment']['save_model']:
        torch.save(engine.global_model.state_dict(), output/'model.pt')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    return summary
