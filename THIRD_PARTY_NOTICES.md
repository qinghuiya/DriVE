# Third-party notices

## Algorithm references

The method modules are local framework implementations. Method names
identify algorithms or the documented adaptations below; they do not imply that
every implementation detail matches an author's original code or benchmark.

- FedAvg: McMahan et al., AISTATS 2017.
- FedProx: Li et al., MLSys 2020.
- Adaptive server optimization: Reddi et al., ICLR 2021. See README for this
  framework's specific momentum and second-moment conventions.
- FedGMT: gradient-matching/teacher-dual baseline; see its explicit update in
  `fedhodge/engine/engine.py`.
- FedExProx: finite-local-SGD approximation to extrapolated proximal updates,
  with constant or sample-weighted, capped GraDS extrapolation.

Datasets and third-party Python packages are distributed separately under their
own terms. No datasets, pretrained weights, or experimental outputs are included.
