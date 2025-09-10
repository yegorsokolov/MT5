# Feature Development Guide

Create new features in the `features/` package and expose a `compute` function.

```bash
# add your feature
echo "def compute(df):\n    return df" > features/my_feature.py
```

```{doctest}
>>> 2 * 2
4
```
