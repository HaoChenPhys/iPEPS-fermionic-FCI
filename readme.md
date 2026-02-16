# iPEPS for Fermionic Fractional Chern Insulators

## Data

This repository hosts the data accompanying the paper: https://arxiv.org/abs/2512.20697

- `FCI_data/states/` contains iPEPS states stored in **JSON** format. The unit-cell structure and conventions are described in the paper.
- `obs/` contains the observables reported in the paper.

## Reading states

See `read_state.py`. The function `read_state` loads a state from disk and returns a `yastn.tn.fpeps.Peps` object (from **YASTN**): https://github.com/yastn/yastn

### Minimal example

```python
from read_state import read_state

filename="FCI_data/states/D9/t1_0.1_3x3_N3_D_9_chi_117_fullrank_cuda_state.json"
peps = read_state(filename)
```

## License

Unless otherwise noted, the data and accompanying materials in this repository are licensed under the
Creative Commons Attribution 4.0 International License (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

SPDX-License-Identifier: CC-BY-4.0