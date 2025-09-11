# Training Tutorial

This tutorial demonstrates how to run a simple training loop.

```bash
python train.py --config config.yaml
```

```{doctest}
>>> def train(cfg):
...     return cfg["epochs"]
>>> train({"epochs": 1})
1
```
