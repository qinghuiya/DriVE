from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from .common import FederatedData, digest, save_manifest, stratified_holdout, partition_manifest


VisionDataBundle = FederatedData


class ResidentTensorDataset(Dataset):
    """A preprocessed vision dataset stored directly on the training device.

    This is intentionally a small, conservative optimization: the existing
    Subset/DataLoader pipeline is kept unchanged so client sampling and batch
    shuffling semantics remain the same. Only the repeated torchvision
    preprocessing and per-batch host->device copies are removed.
    """

    def __init__(self, images: torch.Tensor, targets: torch.Tensor):
        if images.shape[0] != targets.shape[0]:
            raise ValueError("images and targets must have the same length.")
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]



TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)


class TinyImageNetValDataset(Dataset):
    """Reader for the official flat Tiny-ImageNet validation layout."""

    def __init__(self, root: str | Path, class_to_idx: dict[str, int], transform=None):
        self.root = Path(root)
        self.transform = transform
        annotation_path = self.root / "val_annotations.txt"
        image_dir = self.root / "images"
        if not annotation_path.exists() or not image_dir.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet validation annotations or images are missing."
            )
        self.samples: list[tuple[Path, int]] = []
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            filename, class_name, *_ = line.split("\t")
            if class_name not in class_to_idx:
                raise ValueError(f"Unknown Tiny-ImageNet class {class_name!r}.")
            self.samples.append((image_dir / filename, class_to_idx[class_name]))
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _tiny_imagenet_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    candidates = [root, root / "tiny-imagenet-200", root / "tiny_imagenet_200"]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(
        "Tiny-ImageNet-200 was not found. Extract it under "
        f"{root / 'tiny-imagenet-200'} (expected train/ and val/)."
    )

def get_dataset(name: str, data_root: str | Path):
    if name == 'synthetic':
        # A fixed, balanced random dataset for interface checks only.
        # Its generator is independent of client/model RNG streams.
        gen = torch.Generator().manual_seed(1729)
        return (ResidentTensorDataset(torch.randn(128, 3, 32, 32, generator=gen), torch.arange(128) % 10),
                ResidentTensorDataset(torch.randn(32, 3, 32, 32, generator=gen), torch.arange(32) % 10), 10)
    from torchvision import datasets, transforms

    root = str(Path(data_root).expanduser())
    key = str(name).lower().replace("_", "-")
    cifar_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if key == "cifar10":
        return (
            datasets.CIFAR10(root, train=True, download=True, transform=cifar_transform),
            datasets.CIFAR10(root, train=False, download=True, transform=cifar_transform),
            10,
        )
    if key == "cifar100":
        return (
            datasets.CIFAR100(root, train=True, download=True, transform=cifar_transform),
            datasets.CIFAR100(root, train=False, download=True, transform=cifar_transform),
            100,
        )
    if key in {"tinyimagenet", "tiny-imagenet", "tiny-imagenet-200"}:
        tiny_root = _tiny_imagenet_root(data_root)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
            ]
        )
        train = datasets.ImageFolder(tiny_root / "train", transform=train_transform)
        validation_root = tiny_root / "val"
        class_dirs = [
            path
            for path in validation_root.iterdir()
            if path.is_dir() and path.name != "images"
        ]
        if class_dirs:
            validation = datasets.ImageFolder(validation_root, transform=eval_transform)
        else:
            validation = TinyImageNetValDataset(
                validation_root, train.class_to_idx, transform=eval_transform
            )
        if len(train.classes) != 200:
            raise ValueError("Tiny-ImageNet training split must contain 200 classes.")
        return train, validation, 200
    raise ValueError(f"Unsupported vision dataset: {name}")


def _labels(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return _labels(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    values = getattr(dataset, "targets", getattr(dataset, "labels", None))
    if values is None:
        values = [dataset[i][1] for i in range(len(dataset))]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def _cache_cifar_on_device(dataset: Dataset, device: torch.device) -> ResidentTensorDataset:
    """Vectorize the existing CIFAR ToTensor+Normalize transform once.

    The legacy/current transform is exactly:
      ToTensor(): uint8 [0,255] HWC -> float32 [0,1] CHW
      Normalize(mean=.5, std=.5): x -> (x-.5)/.5

    We apply the same operations to the whole dataset and keep the resulting
    tensors on ``device`` for the remainder of the run.
    """

    raw = getattr(dataset, "data", None)
    targets = getattr(dataset, "targets", None)
    if raw is None or targets is None:
        raise TypeError(
            "dataset.cache_on_device currently supports torchvision CIFAR-10/100 only."
        )

    array = np.asarray(raw)
    if array.ndim != 4 or array.shape[-1] not in (1, 3):
        raise ValueError(f"Unexpected CIFAR image array shape: {array.shape}")

    # torch.from_numpy is zero-copy on CPU. The following .to performs one
    # dataset-sized host->device transfer while converting uint8 -> float32.
    images = (
        torch.from_numpy(array)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    images.div_(255.0).sub_(0.5).div_(0.5)
    labels = torch.as_tensor(targets, dtype=torch.long, device=device)
    return ResidentTensorDataset(images, labels)


def iid_partition(dataset: Dataset, num_clients: int, seed: int) -> list[list[int]]:
    if not 1 <= num_clients <= len(dataset):
        raise ValueError('num_clients must be between 1 and the number of training samples')
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(dataset))
    return [chunk.tolist() for chunk in np.array_split(permutation, num_clients)]


def dirichlet_partition(
    dataset: Dataset,
    *,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
    allocation_mode: str = "multinomial",
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if num_clients < 1 or min_samples_per_client < 1 or num_clients * min_samples_per_client > len(dataset):
        raise ValueError('The dataset cannot provide the requested minimum samples per client')
    mode = str(allocation_mode or "multinomial").strip().lower()
    if mode not in {"multinomial", "unbiased"}:
        raise ValueError(
            "Only allocation_mode='multinomial' is supported. The historical "
            "'classwise'/'legacy' allocator was removed because it assigned every "
            "integer-rounding remainder to the final client."
        )
    labels = _labels(dataset)
    rng = np.random.default_rng(seed)
    clients: list[list[int]] = [[] for _ in range(num_clients)]

    # Class-wise allocation matches the DriVE CIFAR protocol.
    for cls in range(num_classes):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(indices), proportions)
        offset = 0
        for client_id, count in enumerate(counts):
            if count:
                clients[client_id].extend(indices[offset : offset + count].tolist())
            offset += count

    # Deterministic repair for rare empty clients.
    for client_id in range(num_clients):
        while len(clients[client_id]) < min_samples_per_client:
            donor = max(range(num_clients), key=lambda j: len(clients[j]))
            if len(clients[donor]) <= min_samples_per_client:
                raise RuntimeError("Unable to satisfy min_samples_per_client.")
            clients[client_id].append(clients[donor].pop())
    for indices in clients:
        rng.shuffle(indices)
    return clients


def _make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    drop_last: bool = False,
    seed_mode: str = "per_client",
):
    subset = Subset(dataset, list(indices))
    mode = str(seed_mode).strip().lower()
    if mode not in {"per_client", "reported"}:
        raise ValueError("dataset.loader_seed_mode must be 'per_client' or 'reported'.")
    # The legacy DriVE code used the global PyTorch RNG (reported mode).
    # per_client gives each loader its own deterministic generator.
    generator = (
        torch.Generator().manual_seed(seed) if mode == "per_client" else None
    )
    return DataLoader(
        subset,
        batch_size=min(batch_size, max(len(subset), 1)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_vision_data(config: dict) -> VisionDataBundle:
    dataset_cfg = config["dataset"]
    training = config["training"]
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 0))
    partition_seed = int(dataset_cfg['partition_seed'])
    device = torch.device(str(experiment.get("device", "cpu")))
    cache_on_device = bool(dataset_cfg.get("cache_on_device", False))
    pin_memory = bool(training.get("pin_memory", device.type == "cuda"))

    train, test, num_classes = get_dataset(
        dataset_cfg.get("name", "cifar10"),
        dataset_cfg.get("data_root", "data"),
    )
    if dataset_cfg['name']=='tinyimagenet' and hasattr(test,'class_to_idx'):
        if test.class_to_idx != train.class_to_idx:
            raise ValueError('Tiny-ImageNet validation and training class mappings differ')
    labels = _labels(train)
    train_indices, val_indices = stratified_holdout(
        labels, dataset_cfg['validation_fraction'], dataset_cfg['split_seed'])
    # Validation uses a deterministic view of the original training split.
    train_eval = copy.copy(train)
    if hasattr(test, 'transform'):
        train_eval.transform = test.transform
    if hasattr(train, 'samples'):
        source_ids = [str(Path(path).relative_to(Path(train.root))).replace('\\', '/')
                      for path, _ in train.samples]
    else:
        source_ids = list(range(len(train)))
    split = {
        'protocol': 'stratified_training_holdout_v1', 'dataset': dataset_cfg['name'],
        'split_seed': dataset_cfg['split_seed'], 'validation_fraction': dataset_cfg['validation_fraction'],
        'source_index_identity_sha256': digest({'ids': source_ids, 'labels': labels.tolist()}),
        'train_indices': train_indices, 'validation_indices': val_indices,
        'test_source': 'official_val' if dataset_cfg['name']=='tinyimagenet' else 'official_test',
        'test_indices': list(range(len(test))),
        'counts': {'train': len(train_indices), 'validation': len(val_indices), 'test': len(test)},
    }
    # Optional caching affects storage, never membership or evaluation transforms.
    if cache_on_device:
        train = _cache_cifar_on_device(train, device)
        train_eval = train
        test = _cache_cifar_on_device(test, device)
        pin_memory = False
    validation = Subset(train_eval, val_indices)
    train = Subset(train, train_indices)
    num_clients = int(dataset_cfg.get("num_clients", 100))
    partition = dataset_cfg.get("partition", {"type": "dirichlet", "alpha": 0.5})
    ptype = str(partition.get("type", "dirichlet")).lower()
    if ptype == "iid":
        client_indices = iid_partition(train, num_clients, partition_seed)
    else:
        client_indices = dirichlet_partition(
            train,
            num_clients=num_clients,
            num_classes=num_classes,
            alpha=float(partition.get("alpha", 0.5)),
            seed=partition_seed,
            min_samples_per_client=int(partition.get("min_samples_per_client", 1)),
            allocation_mode=str(partition.get("allocation_mode", "multinomial")),
        )

    batch_size = int(training.get("batch_size", 20))
    client_loaders = [
        _make_loader(
            train,
            indices,
            batch_size=batch_size,
            shuffle=bool(training.get("client_loader_shuffle", True)),
            seed=seed + client_id,
            pin_memory=pin_memory,
            drop_last=bool(training.get("client_loader_drop_last", False)),
            seed_mode=str(dataset_cfg.get("loader_seed_mode", "per_client")),
        )
        for client_id, indices in enumerate(client_indices)
    ]

    eval_workers = int(training.get("eval_num_workers", 0))
    if cache_on_device:
        # DataLoader workers cannot safely own CUDA tensors. Keeping num_workers
        # at zero also preserves the intended minimal-change benchmark.
        eval_workers = 0
    def eval_loader(dataset):
        return DataLoader(dataset, batch_size=training['eval_batch_size'], shuffle=False,
                          num_workers=eval_workers, pin_memory=pin_memory,
                          persistent_workers=eval_workers > 0,
                          generator=torch.Generator().manual_seed(dataset_cfg['split_seed']))
    # Keep the outer split SHA separate from the client partition, so changing
    # N/alpha/model/training seed cannot silently produce a different holdout.
    split_manifest, path = save_manifest(config, split)
    manifest = {'split': split_manifest, 'client_partition': partition_manifest(config,
                    [[train_indices[i] for i in ids] for ids in client_indices])}
    return VisionDataBundle(train_dataset=train, validation_dataset=validation,
        test_dataset=test, client_loaders=client_loaders, validation_loader=eval_loader(validation),
        test_loader=eval_loader(test) if experiment['phase']=='confirm' else None,
        num_classes=num_classes, in_channels=3,
        model_kwargs={'image_size': 64 if dataset_cfg['name']=='tinyimagenet' else 32},
        manifest=manifest, manifest_path=path,
        client_ids=[str(i) for i in range(len(client_loaders))])
