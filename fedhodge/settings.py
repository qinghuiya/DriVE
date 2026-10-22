"""Small explicit configuration surface shared by the CLI and Python API."""
from __future__ import annotations

import copy
import math

import torch

from .engine.engine import METHODS

DEFAULTS = {
    'experiment': {'seed': 0, 'device': 'auto', 'deterministic': True, 'save_model': False, 'threads': 4,
                   'phase': 'tune'},
    'dataset': {'name': 'cifar10', 'data_root': 'data', 'num_clients': None,
                'split_seed': 20261021, 'partition_seed': None, 'validation_fraction': .1, 'split_dir': None,
                'cache_on_device': False, 'loader_seed_mode': 'per_client',
                'partition': {'type': 'dirichlet', 'alpha': .5,
                              'min_samples_per_client': 1, 'allocation_mode': 'multinomial'}},
    'model': {'name': 'auto'},
    'training': {'num_rounds': 100, 'clients_per_round': 10, 'participation': None,
                 'local_steps': 5, 'local_epochs': None,
                 'batch_size': 20, 'eval_every': 5, 'eval_batch_size': 256,
                 'eval_num_workers': 0, 'client_loader_shuffle': True,
                 'client_loader_drop_last': False, 'late_fraction': .2},
    'audio': {'train_folds': [1,2,3,4,5,6,7,8], 'validation_folds': [9], 'test_folds': [10],
              'source_overlap_policy': 'drop_train',
              'sample_rate': 16000, 'duration': 4., 'n_mels': 64, 'n_fft': 1024,
              'hop_length': 320, 'cache_features': False},
    'text': {'sequence_length': 80},
    'algorithm': {'name': 'fedavg', 'client_optimizer': {'lr': .01, 'momentum': .9, 'weight_decay': .0005}},
    'drive': {'poly_order': 0, 'window_size': 10, 'lambda_v': 1.},
    'current_only': {'scale': 2.},
    'baseline': {'proximal_mu': .5, 'server_lr': 1., 'momentum_beta': .7,
                 'adam_lr': .01, 'adam_beta1': .9, 'adam_beta2': .99, 'adam_tau': .001,
                 'extrap_eta': 1., 'grads_eta_cap': 5., 'gmt_gamma': .5,
                 'gmt_tau': 3., 'gmt_beta': 10., 'gmt_ema_alpha': .95},
}


def merge_config(base, patch, path=''):
    if not isinstance(patch, dict):
        raise ValueError(f'{path or "config"} must be a mapping')
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f'Unknown setting: {path}{key}')
        if isinstance(base[key], dict):
            merge_config(base[key], value, path+key+'.')
        else:
            base[key] = value
    return base


def resolve_config(patch=None):
    config = merge_config(copy.deepcopy(DEFAULTS), patch or {})
    validate_config(config)
    return config


def validate_config(c):
    def integer(v, name, minimum=1):
        if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')

    def scalar(v, name, minimum=0., strict=False, maximum=None):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'{name} must be a finite number')
        if v < minimum or (strict and v == minimum) or (maximum is not None and v >= maximum):
            raise ValueError(f'Invalid range for {name}')

    def choice(v, options, name):
        if v not in options:
            raise ValueError(f'{name} must be one of {list(options)}')

    e, d, t = c['experiment'], c['dataset'], c['training']
    integer(e['seed'], 'seed', 0)
    integer(e['threads'], 'threads')
    choice(e['phase'], ('tune', 'confirm'), 'phase')
    if e['device'] == 'auto':
        e['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        device = torch.device(e['device'])
    except (TypeError, RuntimeError) as exc:
        raise ValueError('device must be cpu, cuda, or cuda:<index>') from exc
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device.type == 'cuda' and (not torch.cuda.is_available() or
            (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError('Requested CUDA device is unavailable')
    for section, keys in ((e, ['deterministic', 'save_model']),
                           (d, ['cache_on_device']),
                           (t, ['client_loader_shuffle', 'client_loader_drop_last']),
                           (c['audio'], ['cache_features'])):
        if any(not isinstance(section[k], bool) for k in keys):
            raise ValueError(f'{keys} must be booleans')
    choice(d['name'], ('cifar10', 'cifar100', 'tinyimagenet', 'synthetic','urbansound8k','shakespeare'), 'dataset.name')
    if c['model']['name'] == 'auto':
        c['model']['name'] = {'urbansound8k':'audio_cnn','shakespeare':'char_lstm'}.get(d['name'],'resnet18')
    choice(c['model']['name'], ('resnet18', 'vit_tiny', 'tinycnn','audio_cnn','char_lstm'), 'model.name')
    if d['name']=='shakespeare' and c['model']['name']!='char_lstm':
        raise ValueError('Shakespeare requires char_lstm')
    if d['name']=='urbansound8k' and c['model']['name']!='audio_cnn':
        raise ValueError('UrbanSound8K requires audio_cnn')
    if d['name'] not in ('shakespeare','urbansound8k') and c['model']['name'] in ('char_lstm','audio_cnn'):
        raise ValueError('Selected model does not match the image dataset')
    if not isinstance(d['data_root'], str) or not d['data_root']:
        raise ValueError('data_root must be a nonempty string')
    choice(d['loader_seed_mode'], ('per_client', 'reported'), 'loader_seed_mode')
    if d['cache_on_device'] and d['name'] not in ('cifar10', 'cifar100'):
        raise ValueError('Device caching is supported for CIFAR-10/100 only')
    integer(d['split_seed'],'split_seed',0)
    if d['partition_seed'] is None: d['partition_seed'] = e['seed']
    integer(d['partition_seed'],'partition_seed',0)
    scalar(d['validation_fraction'],'validation_fraction',strict=True,maximum=1.)
    if d['split_dir'] is not None and (not isinstance(d['split_dir'],str) or not d['split_dir']):
        raise ValueError('split_dir must be null or a nonempty path')
    if d['num_clients'] is None and d['name']!='shakespeare': d['num_clients']=100
    if d['num_clients'] is not None: integer(d['num_clients'], 'num_clients')
    part = d['partition']
    if d['name']=='shakespeare': part['type']='natural'
    choice(part['type'], ('iid', 'dirichlet','natural'), 'partition.type')
    if part['type']=='natural' and d['name']!='shakespeare':
        raise ValueError('Natural clients are supported for Shakespeare only')
    choice(part['allocation_mode'], ('multinomial',), 'allocation_mode')
    scalar(part['alpha'], 'alpha', strict=True)
    integer(part['min_samples_per_client'], 'min_samples_per_client')
    for key in ('num_rounds', 'batch_size', 'eval_every', 'eval_batch_size'):
        integer(t[key], key)
    if t['participation'] is None:
        integer(t['clients_per_round'],'clients_per_round')
    else:
        scalar(t['participation'],'participation',strict=True)
        if t['participation']>1 or t['clients_per_round'] is not None:
            raise ValueError('Use participation in (0,1] with clients_per_round=null, or a fixed client count')
    integer(t['eval_num_workers'], 'eval_num_workers', 0)
    scalar(t['late_fraction'],'late_fraction',strict=True,maximum=1.)
    if d['num_clients'] is not None and t['clients_per_round'] is not None and t['clients_per_round'] > d['num_clients']:
        raise ValueError('clients_per_round cannot exceed num_clients')
    if (t['local_steps'] is None) == (t['local_epochs'] is None):
        raise ValueError('Specify exactly one of local_steps and local_epochs')
    for key in ('local_steps', 'local_epochs'):
        if t[key] is not None:
            integer(t[key], key)
    choice(c['algorithm']['name'], METHODS, 'method')
    opt = c['algorithm']['client_optimizer']
    scalar(opt['lr'], 'client lr', strict=True)
    scalar(opt['momentum'], 'client momentum', maximum=1.)
    scalar(opt['weight_decay'], 'weight_decay')
    v = c['drive']
    integer(v['poly_order'], 'p', 0)
    choice(v['poly_order'], (0, 1, 2), 'p')
    if isinstance(v['poly_order'], bool):
        raise ValueError('p must be 0, 1, or 2')
    integer(v['window_size'], 'window_size', v['poly_order']+1)
    scalar(v['lambda_v'], 'lambda_v')
    scalar(c['current_only']['scale'], 'current_only.scale', strict=True)
    b = c['baseline']
    for key, value in b.items():
        scalar(value, 'baseline.'+key, strict=key not in ('proximal_mu', 'gmt_gamma') and 'beta' not in key and key != 'gmt_ema_alpha',
               maximum=1. if key in ('momentum_beta', 'adam_beta1', 'adam_beta2', 'gmt_ema_alpha') else None)
    scalar(b['gmt_beta'], 'baseline.gmt_beta', strict=True)
    if b['grads_eta_cap'] < 1.:
        raise ValueError('grads_eta_cap must be >= 1')
    audio = c['audio']
    choice(audio['source_overlap_policy'],('error','drop_train','official'),'audio.source_overlap_policy')
    folds = []
    for key in ('train_folds','validation_folds','test_folds'):
        value=audio[key]
        if not isinstance(value,list) or not value: raise ValueError(key+' must be a nonempty list')
        for fold in value: integer(fold,key)
        if max(value)>10 or len(set(value))!=len(value): raise ValueError('Invalid or repeated audio fold')
        folds.extend(value)
    if sorted(folds)!=list(range(1,11)): raise ValueError('Audio folds must be disjoint and cover 1..10 exactly')
    for key in ('sample_rate','n_mels','n_fft','hop_length'): integer(audio[key],'audio.'+key)
    scalar(audio['duration'],'audio.duration',strict=True)
    if audio['hop_length']>audio['n_fft']:raise ValueError('Audio hop_length cannot exceed n_fft')
    integer(c['text']['sequence_length'],'sequence_length')
    return c
