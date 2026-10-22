"""Natural-client Shakespeare splits, cut before constructing character windows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .common import FederatedData, digest, save_manifest, evaluation_loader


def load_clients(folder):
    files = sorted(Path(folder).glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No LEAF JSON files in {folder}')
    clients = {}
    for file in files:
        payload = json.loads(file.read_text(encoding='utf8'))
        for client, data in payload.get('user_data',{}).items():
            if client in clients:
                raise ValueError(f'Duplicate natural client {client}')
            raw = data.get('raw', data.get('x') if 'y' not in data else None)
            if not isinstance(raw,list) or any(not isinstance(x,str) for x in raw):
                raise ValueError('Shakespeare needs original per-client text chunks (raw, or x without y). '
                                 'Pre-windowed x/y samples without source positions cannot be split safely.')
            clients[client] = raw
    return clients


def split_client_text(chunks, fraction, seed, client, sequence_length=80):
    """Return source-indexed character spans; never concatenate separate chunks."""
    valid = [i for i,text in enumerate(chunks) if len(text)>=2]
    total = sum(len(chunks[i]) for i in valid)
    if total < 4:
        return [(i,0,len(chunks[i])) for i in valid], [], total
    local_seed = int(hashlib.sha256(f'{seed}:{client}'.encode()).hexdigest()[:16],16)
    rng = np.random.default_rng(local_seed)
    rng.shuffle(valid)
    minimum = min(sequence_length+1,max(2,total//2))
    target_val = max(minimum,min(total-minimum,round(total*fraction)))
    boundary = total-target_val
    train, validation, offset = [], [], 0
    for i in valid:
        n = len(chunks[i])
        cut = max(0,min(n,boundary-offset))
        if cut >= 2: train.append((i,0,cut))
        if n-cut >= 2: validation.append((i,cut,n))
        offset += n
    # Very fragmented clients may have no useful suffix after discarding
    # one-character spans. Make a disjoint holdout inside their largest chunk.
    if not validation or not train:
        largest = max(valid,key=lambda i:len(chunks[i]))
        if len(chunks[largest]) >= 4:
            cut = max(2,min(len(chunks[largest])-2,round(len(chunks[largest])*(1-fraction))))
            train = [(i,0,len(chunks[i])) for i in valid if i!=largest]+[(largest,0,cut)]
            validation = [(largest,cut,len(chunks[largest]))]
        else:
            # No client is silently discarded; small-client membership is logged.
            train = [(i,0,len(chunks[i])) for i in valid[:-1]]
            validation = [(valid[-1],0,len(chunks[valid[-1]]))]
    return train,validation,total


class CharacterWindows(Dataset):
    def __init__(self, chunks, spans, vocabulary, length):
        self.chunks,self.spans,self.vocabulary,self.length = chunks,spans,vocabulary,length
        self.counts = np.asarray([max(1,end-start-length) for _,start,end in spans],dtype=np.int64)
        self.ends = np.cumsum(self.counts)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self,index):
        part = int(np.searchsorted(self.ends,index,side='right'))
        offset = index-(int(self.ends[part-1]) if part else 0)
        chunk,start,end = self.spans[part]
        text = self.chunks[chunk][start:end]
        sample = text[offset:offset+self.length+1]
        inputs,target = sample[:-1],sample[-1]
        ids = [self.vocabulary.get(ch,1) for ch in inputs]
        ids += [0]*(self.length-len(ids))
        return torch.tensor(ids,dtype=torch.long),torch.tensor(self.vocabulary.get(target,1),dtype=torch.long)


def build_text_data(config):
    cfg,text_cfg = config['dataset'],config['text']
    root = Path(cfg['data_root']).expanduser().resolve()
    if not (root/'train').is_dir(): root = root/'shakespeare'
    raw_train,raw_test = load_clients(root/'train'),load_clients(root/'test')
    clients = sorted(raw_train)
    if cfg['num_clients'] is not None: clients = clients[:cfg['num_clients']]
    plans,charset,small = {},set(),[]
    for client in clients:
        a,b,total = split_client_text(raw_train[client],cfg['validation_fraction'],cfg['split_seed'],client,
                                    text_cfg['sequence_length'])
        if not a:
            raise ValueError(f'Client {client} has no usable training text; fix the client list explicitly')
        if not b:small.append(client)
        # Exact duplicate original chunks across supplied train/test sources are
        # rejected. Source-position metadata remains necessary for auditing any
        # externally preprocessed files that may contain partial duplicates.
        train_hashes = {digest(x) for x in raw_train[client]}
        if any(digest(x) in train_hashes for x in raw_test.get(client,[])):
            raise ValueError(f'Identical original text chunk occurs in train and test for {client}')
        plans[client] = {'train_spans':a,'validation_spans':b,
                        'training_characters':sum(end-start for _,start,end in a),
                        'validation_characters':sum(end-start for _,start,end in b),
                        'discarded_short_span_characters':total-sum(end-start for _,start,end in a+b)}
        for i,start,end in a: charset.update(raw_train[client][i][start:end])
    vocabulary = {ch:i+2 for i,ch in enumerate(sorted(charset))}
    # Padding and unknown tokens occupy indices 0 and 1; held-out text never
    # determines the vocabulary or model output dimension.
    trains,validations,tests,loaders = [],[],[],[]
    for cid,client in enumerate(clients):
        plan = plans[client]
        train = CharacterWindows(raw_train[client],plan['train_spans'],vocabulary,text_cfg['sequence_length'])
        val = CharacterWindows(raw_train[client],plan['validation_spans'],vocabulary,text_cfg['sequence_length'])
        test_chunks = raw_test.get(client,[])
        test_spans = [(i,0,len(x)) for i,x in enumerate(test_chunks) if len(x)>=2]
        test = CharacterWindows(test_chunks,test_spans,vocabulary,text_cfg['sequence_length'])
        trains.append(train);validations.append(val);tests.append(test)
        generator = (torch.Generator().manual_seed(config['experiment']['seed']+cid)
                     if cfg['loader_seed_mode']=='per_client' else None)
        loaders.append(DataLoader(train,batch_size=min(config['training']['batch_size'],len(train)),
            shuffle=config['training']['client_loader_shuffle'],generator=generator,
            drop_last=config['training']['client_loader_drop_last']))
        plan.update(train_windows=len(train),validation_windows=len(val),test_windows=len(test),test_spans=test_spans)
    train,val,test = ConcatDataset(trains),ConcatDataset(validations),ConcatDataset(tests)
    if not len(val) or not len(test):raise ValueError('The selected natural clients have no validation or test windows')
    split = {'protocol':'natural_client_raw_text_holdout_v1','dataset':'shakespeare',
             'split_seed':cfg['split_seed'],'validation_fraction':cfg['validation_fraction'],
             'source_text_identity_sha256':digest({'train':{k:raw_train[k] for k in clients},'test':{k:raw_test.get(k,[]) for k in clients}}),
             'vocabulary':vocabulary,'sequence_length':text_cfg['sequence_length'],
             'clients':plans,'small_clients_without_validation':small,
             'counts':{'train':len(train),'validation':len(val),'test':len(test)}}
    manifest,path = save_manifest(config,split)
    return FederatedData(train,val,test,loaders,evaluation_loader(val,config),
        evaluation_loader(test,config) if config['experiment']['phase']=='confirm' else None,
        num_classes=len(vocabulary)+2,in_channels=1,model_kwargs={'vocab_size':len(vocabulary)+2},
        manifest={'split':manifest,'natural_client_ids':clients},manifest_path=path,client_ids=clients)
