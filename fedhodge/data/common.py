"""Shared data bundles and auditable, deterministic split manifests."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class FederatedData:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    client_loaders: list[DataLoader]
    validation_loader: DataLoader
    test_loader: DataLoader | None
    num_classes: int
    in_channels: int = 3
    model_kwargs: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    manifest_path: str = ''
    client_ids: list[str] = field(default_factory=list)


def canonical_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')


def digest(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def save_manifest(config, payload):
    """Persist content-addressed indices without replacing a different manifest."""
    body = dict(payload)
    checksum = digest(body)
    result = {**body, 'sha256': checksum}
    folder = Path(config['dataset']['split_dir'] or Path(config['dataset']['data_root'])/'splits')
    folder = folder.expanduser().resolve()/config['dataset']['name']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(checksum+'.json')
    data = canonical_bytes(result)+b'\n'
    tmp = folder/('.'+uuid4().hex+'.tmp')
    try:
        tmp.write_bytes(data)
        try:
            # Same-directory hard link gives an atomic, exclusive publication.
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f'Existing split manifest failed integrity check: {path}')
    finally:
        tmp.unlink(missing_ok=True)
    return result, str(path)


def stratified_holdout(labels, fraction, seed):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError('Every class needs at least two samples for a stratified holdout')
        n = max(1, min(len(indices)-1, int(round(len(indices)*fraction))))
        validation.extend(indices[:n].tolist())
        training.extend(indices[n:].tolist())
    # Sorted source indices keep the split independent of later loader RNG.
    training, validation = sorted(training), sorted(validation)
    assert not set(training).intersection(validation)
    assert sorted(training+validation) == list(range(len(labels)))
    return training, validation


def evaluation_loader(dataset, config):
    return DataLoader(dataset, batch_size=config['training']['eval_batch_size'],
                      shuffle=False, num_workers=config['training']['eval_num_workers'],
                      generator=torch.Generator().manual_seed(config['dataset']['split_seed']))


def partition_manifest(config, indices):
    return {'seed': config['dataset']['partition_seed'],
            'type': config['dataset']['partition']['type'],
            'alpha': config['dataset']['partition']['alpha'],
            'client_indices': indices}
