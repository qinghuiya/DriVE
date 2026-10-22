"""UrbanSound8K official-fold protocol; no random splitting of audio slices."""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from .common import FederatedData, digest, save_manifest, evaluation_loader, partition_manifest
from .vision import iid_partition, dirichlet_partition, _make_loader


class UrbanSoundDataset(Dataset):
    def __init__(self, root, rows, settings):
        self.root, self.rows, self.settings = Path(root), rows, settings
        self.targets = [r['label'] for r in rows]
        self._cache = {}
        self._audio = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index in self._cache:
            return self._cache[index], self.targets[index]
        if self._audio is None:
            try:
                import torchaudio
            except (ImportError, OSError) as exc:
                raise RuntimeError('Install a torchaudio build matching your PyTorch build for UrbanSound8K') from exc
            self._audio = torchaudio
        ta = self._audio
        cfg = self.settings
        row = self.rows[index]
        path = self.root/'audio'/f"fold{row['fold']}"/row['filename']
        try:
            import soundfile as sf
            array, rate = sf.read(path, dtype='float32', always_2d=True)
            waveform = torch.from_numpy(array).mean(1, keepdim=True).T
        except ImportError:
            waveform, rate = ta.load(path)
            waveform = waveform.mean(0, keepdim=True)
        if rate != cfg['sample_rate']:
            waveform = ta.functional.resample(waveform, rate, cfg['sample_rate'])
        length = round(cfg['duration']*cfg['sample_rate'])
        waveform = waveform[:, :length]
        waveform = torch.nn.functional.pad(waveform, (0, max(0,length-waveform.shape[-1])))
        mel = ta.transforms.MelSpectrogram(sample_rate=cfg['sample_rate'], n_mels=cfg['n_mels'],
                    n_fft=cfg['n_fft'], hop_length=cfg['hop_length'])(waveform)
        features = ta.transforms.AmplitudeToDB(stype='power')(mel).to(torch.float32)
        # Per-recording normalization; no statistics are fitted on validation/test.
        features = ((features-features.mean())/(features.std()+1e-6)).detach()
        if cfg['cache_features']:
            self._cache[index] = features
        return features, self.targets[index]


def build_audio_data(config):
    cfg, settings = config['dataset'], config['audio']
    root = Path(cfg['data_root']).expanduser().resolve()
    if not (root/'metadata/UrbanSound8K.csv').is_file():
        root = root/'UrbanSound8K'
    csv_path = root/'metadata/UrbanSound8K.csv'
    if not csv_path.is_file():
        raise FileNotFoundError('Expected UrbanSound8K/metadata/UrbanSound8K.csv and audio/fold1..fold10')
    rows, seen = [], set()
    with csv_path.open(encoding='utf8', newline='') as f:
        for row in csv.DictReader(f):
            fold, name, label = int(row['fold']), row['slice_file_name'], int(row['classID'])
            identity = f'fold{fold}/{name}'
            if identity in seen or not 1 <= fold <= 10 or not 0 <= label < 10:
                raise ValueError('Duplicate or invalid UrbanSound8K metadata')
            seen.add(identity)
            source = row.get('fsID') or name.split('-')[0]
            path = root/'audio'/f'fold{fold}'/name
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({'fold':fold,'filename':name,'label':label,'source':source,'id':identity})
    rows.sort(key=lambda x:(x['fold'],x['filename']))
    groups = {key:[i for i,r in enumerate(rows) if r['fold'] in settings[key+'_folds']]
              for key in ('train','validation','test')}
    sources = {k:{rows[i]['source'] for i in ids} for k,ids in groups.items()}
    shared = (sources['train'] & (sources['validation'] | sources['test'])) | (sources['validation'] & sources['test'])
    exclusions = {'train':[], 'validation':[]}
    policy = settings['source_overlap_policy']
    if shared:
        if policy == 'error':
            raise ValueError(f'{len(shared)} original recording source(s) cross train/validation/test. '
                'Choose audio.source_overlap_policy=drop_train to exclude overlapping training/validation sources, '
                'or official to preserve folds and record this limitation explicitly.')
        if policy == 'drop_train':
            for key, forbidden in [('train',sources['validation'] | sources['test']),('validation',sources['test'])]:
                exclusions[key] = [i for i in groups[key] if rows[i]['source'] in forbidden]
                groups[key] = [i for i in groups[key] if rows[i]['source'] not in forbidden]
        else:
            warnings.warn('Official UrbanSound8K folds contain a source shared across data roles; '
                          'see data_split.json for the recorded overlap.', stacklevel=2)
    if any(not ids for ids in groups.values()):
        raise ValueError('UrbanSound8K train/validation/test folds must all contain recordings')
    dataset = UrbanSoundDataset(root, rows, settings)
    train, validation, test = [Subset(dataset,groups[k]) for k in ('train','validation','test')]
    p = cfg['partition']
    if p['type'] == 'iid':
        indices = iid_partition(train,cfg['num_clients'],cfg['partition_seed'])
    else:
        indices = dirichlet_partition(train,num_clients=cfg['num_clients'],num_classes=10,
            alpha=p['alpha'],seed=cfg['partition_seed'],min_samples_per_client=p['min_samples_per_client'])
    loaders = [_make_loader(train,ids,batch_size=config['training']['batch_size'],
               shuffle=config['training']['client_loader_shuffle'],
               seed=config['experiment']['seed']+cid,pin_memory=False,
               seed_mode=cfg['loader_seed_mode'],drop_last=config['training']['client_loader_drop_last'])
               for cid,ids in enumerate(indices)]
    split = {'protocol':'official_recording_folds_v1','dataset':'urbansound8k',
             'source_metadata_sha256':digest(rows),
             'source_overlap_policy':policy, 'original_cross_partition_sources':sorted(shared),
             'excluded_sample_ids':{k:[rows[i]['id'] for i in ids] for k,ids in exclusions.items()},
             'remaining_source_overlap':bool(shared) and policy=='official',
             'folds':{k:settings[k+'_folds'] for k in groups},
             'sample_ids':{k:[rows[i]['id'] for i in ids] for k,ids in groups.items()},
             'counts':{k:len(ids) for k,ids in groups.items()}}
    manifest,path = save_manifest(config,split)
    client_indices = [[groups['train'][i] for i in ids] for ids in indices]
    return FederatedData(train,validation,test,loaders,evaluation_loader(validation,config),
        evaluation_loader(test,config) if config['experiment']['phase']=='confirm' else None,
        num_classes=10,in_channels=1,manifest={'split':manifest,'client_partition':partition_manifest(config,client_indices)},
        manifest_path=path,client_ids=[str(i) for i in range(len(loaders))])
