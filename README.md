# DriVE

A compact PyTorch framework for federated learning and server-side virtual
evolution. It provides method implementations, a shared client-training engine,
and **one entry point: `train.py`**.

This is a framework release. Dataset files, pretrained weights, experimental
results, private experiment queues, hyperparameter-selection pipelines,
communication/cost measurement tools, and plotting scripts are not included.
The defaults and commands below illustrate the interface; they are not a
complete recipe for reproducing a paper's tables.

## Install

Python 3.10+ is required. Install a mutually compatible PyTorch/torchvision build
for your CPU or CUDA environment using the [PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/qinghuiya/DriVE.git
cd DriVE
python -m pip install -e .
```

## Run one experiment

```bash
python train.py --method fedavg --dataset cifar10 --model resnet18 --seed 1
python train.py --method drive --p 0 --window 10 --lambda-v 1 --seed 1
```

Common controls:

```bash
python train.py --config configs/cifar10.yaml --method drive --p 1 \
  --window 10 --lambda-v 1 --rounds 100 --num-clients 100 \
  --clients-per-round 10 --local-steps 5 --batch-size 20 \
  --alpha 0.5 --lr 0.01 --momentum 0.9 --weight-decay 0.0005 \
  --device cuda:0 --data-root ./data --output ./runs/example
```

The `\` continuation above is for POSIX shells; on Windows, put the command on
one line. Use `--local-epochs` instead of `--local-steps` for epoch-based local
training. `--participation` can replace `--clients-per-round` (the fraction is
rounded to a count after loading the selected client population, at least one).
`--partition iid` selects IID splits;
otherwise class-wise Dirichlet/multinomial allocation is used.

Every known configuration field can be changed with `--set KEY=VALUE` (YAML
values). Unknown keys and invalid ranges are rejected. Order of precedence is:
built-in defaults, YAML, named flags, then `--set`. `--participation`, when used,
is resolved against the actual client count after loading the data.

```bash
python train.py --method fedprox --set baseline.proximal_mu=0.1
python train.py --method fedavg --set dataset.cache_on_device=true --dry-run
python train.py --help
```

An offline, small CPU interface check (random data; no scientific result):

```bash
python train.py --dataset synthetic --model tinycnn --device cpu --num-clients 4 --clients-per-round 2 --rounds 3 --local-steps 2 --eval-every 1
```

CIFAR-10/100 are downloaded through torchvision if absent. Tiny-ImageNet must be
extracted separately into `data/tiny-imagenet-200/{train,val}`; select
`--dataset tinyimagenet`. Models: CIFAR-style ResNet18 (3x3 stem, no ImageNet
max-pool), a small ViT, AudioCNN, character LSTM, and `tinycnn` for interface
checks. The default `--model auto` chooses ResNet18, AudioCNN, or character LSTM
from the dataset. Audio support needs a matching torchaudio installation and
optionally soundfile (`python -m pip install -e ".[audio]"`; keep torch and
torchaudio versions compatible).

## Training, validation, and test protocol

| Dataset | Training data | Validation data | Held-out test data |
| --- | --- | --- | --- |
| CIFAR-10 | 45,000 official training images | 5,000 official training images, 500/class | 10,000 official test images |
| CIFAR-100 | 45,000 official training images | 5,000 official training images, 50/class | 10,000 official test images |
| Tiny-ImageNet | 90,000 official training images | 10,000 official training images, 50/class | 10,000 official validation images |
| UrbanSound8K | Folds 1–8, excluding sources shared with held-out sets | Fold 9, excluding any test sources | Fold 10, unchanged |
| Shakespeare | About 90% of each client's original training text | About 10% of each client's original training text | Existing LEAF test portion |

The default holdout seed is **20261021**, independent of the training seed.
Image holdouts are stratified by class **before** client partitioning; only
training samples enter IID/Dirichlet client loaders. ResNet18 and ViT use the
same held-out images. `--split-seed` controls membership;
`--partition-seed` controls the subsequent client split (defaults to the training
seed when not specified). Changing N, participation, alpha, or the training seed
does not change the image holdout. Validation uses deterministic evaluation
transforms, not training augmentation. `dataset.validation_fraction` defaults
to 0.1 for images/text.

UrbanSound8K reads `UrbanSound8K/metadata/UrbanSound8K.csv` and `audio/fold*`.
It checks original recording IDs across training/validation/test roles.
The default `audio.source_overlap_policy=drop_train` excludes training clips
sharing a source with validation/test (and validation
clips sharing a test source), preserving test membership. `official` preserves
the official folds, warns about cross-role sources, and records the limitation;
`error` refuses to proceed when cross-role sources exist.
The chosen policy and every excluded clip are saved in the manifest. Only
training candidates from folds 1–8 can reach clients. This is a fixed-fold
protocol, not a claim to run official ten-fold cross-validation.

Shakespeare reads LEAF-style `shakespeare/train/*.json` and `test/*.json`,
preserving natural client IDs. Payloads must contain original text chunks in
`raw`, or `x` without precomputed `y`. Each training chunk is split into
source-indexed character spans before constructing windows; windows never cross
a train/validation boundary or a chunk boundary. Small clients may need more
than 10% held-out characters to retain a useful sequence; actual spans, counts,
and tiny clients without validation are logged. The vocabulary is fitted on
training spans only; held-out characters map to an unknown token. Pre-windowed
`x/y` JSON without raw text/source positions is rejected. Exact duplicate raw
chunks in supplied train/test portions are also rejected; externally prepared
source data still need provenance checks for partial overlaps. No synthetic
Dirichlet split is imposed on natural text clients. By default all clients are
used; `--num-clients` can explicitly limit the sorted natural-client list.

Every run saves the source-indexed split and a SHA-256 digest in
`data_split.json`. Canonical manifests are content-addressed under
`data/splits/<dataset>/` (or `dataset.split_dir`) and are checked on reuse.
Client partitions are recorded separately from the image holdout digest.

The default phase is **tune**: only validation is evaluated. Test loaders are
not supplied to the training loop. Freeze parameters using validation metrics,
then run **confirm** with the selected configuration file and fresh training
seeds:

```bash
python train.py --phase tune --dataset cifar10 --seed 0 --output runs/candidate
python train.py --phase confirm --config runs/candidate/config.yaml --seed 1 --output runs/confirmation
python train.py --config configs/urbansound8k.yaml --phase tune
python train.py --config configs/shakespeare.yaml --phase tune
```

Confirmation requires `--config`. This loads a chosen configuration; the
framework does not perform parameter search or certify how a human selected it.
Confirmation evaluates validation and test at the same predetermined rounds;
test results never drive automatic parameter/checkpoint selection or stopping.
All phases continue to train only on the training subset; validation is not
merged back into training. Use the same policy for every compared method.

Each invocation creates a **new** output directory containing `config.yaml`,
`data_split.json`, `metrics.jsonl`, and `summary.json`. Accuracy is a fraction in
[0, 1]. Metrics are named `val_accuracy`/`val_loss`; confirmation additionally
has `test_accuracy`/`test_loss`. Late accuracy is the mean at evaluation rounds
strictly after the first 80% of the fixed training budget (configurable through
`training.late_fraction`). Summary validation and test fields are separate;
tuning produces no test metrics. Add
`--save-model` to save the final global state. Existing output directories are
refused. There is no automatic sweep, retry, or resume. A seed alone does not
guarantee bitwise equality across devices and library versions.

## Methods and conventions

| `--method` | Implementation |
| --- | --- |
| `fedavg` | Processed-sample-weighted client endpoint aggregation. |
| `drive` | DriVE polynomial orders `--p 0`, `1`, `2`. |
| `fedprox` | Local CE plus a quadratic proximal term to the incoming global parameters. |
| `fedavgm` | Unnormalized server momentum: `m = beta*m + delta`; step `server_lr*m`. |
| `fedadam` | Experimental Adam-style server variant described below. |
| `fedgmt` | Teacher distillation, persistent per-client duals, and a server dual correction. |
| `fedexprox_constant` | Inexact local proximal solve with constant extrapolation. |
| `fedexprox_grads` | Inexact solve with sample-weighted, capped GraDS extrapolation. |
| `current_only` | Scale the current real parameter displacement, with no history. |

All methods share a sequential client loop and fresh SGD state at every client
participation. Parameters **and** BatchNorm buffers are reset before each
client. Short loaders repeat until exactly K local steps are completed.
Floating model states are averaged by processed-sample counts; integer buffers
are taken from the client with the largest count (first on ties).

**DriVE.** Each round records only the real aggregated parameter displacement
`delta_t`. The newest W real displacements are fit by a degree-p polynomial at
times `0,-1,...,-(W-1)` and integrated over `[0,lambda_v]`. The resulting vector
is added after the real FedAvg endpoint. The current real displacement is
stored before checking eligibility: rounds 1 through W-1 use the base optimizer
alone, and the first eligible correction is after real round W (zero-based
index `t0 = W-1`). Thus W=30 starts at round 30. For p0, the correction is
`lambda_v * mean(history)`.
Virtual corrections do not enter history, train clients, change BatchNorm
buffers, or advance client optimizer state. `lambda_v=0` disables corrections.

**Baseline adaptations.** These are implementations within this shared engine,
not bundled author repositories. In `fedadam`, `m = beta1*m + (1-beta1)*delta`,
`v = beta2*v + (1-beta2)*m*m`, and the step is
`adam_lr*m/(sqrt(v)+adam_tau)`, without bias correction. Its second moment is
computed from **m**, so it must not be described as canonical FedAdam.
FedExProx uses finite local SGD rather than an exact proximal oracle; GraDS uses
processed-sample weights and clamps extrapolation to `[1, grads_eta_cap]`.
FedGMT uses CE plus teacher KL and a linear dual term, adds each client's
endpoint displacement to its dual, then adds the mean of all client duals to
the server endpoint. The teacher parameters follow an EMA; its buffers are
copied. Duals are stored lazily on CPU and can still require substantial memory.

## Extend the framework

`fedhodge/engine/engine.py` owns client training and aggregation;
`fedhodge/engine/drive.py` owns polynomial virtual evolution. New server methods
can consume a real displacement and return a parameter correction without
duplicating the client-training loop. `fedhodge/settings.py` defines the public
configuration surface. The installed `drive-train` command is an alias of
the same `train.py` entry point.

MIT license for the framework; third-party notices are retained separately.
