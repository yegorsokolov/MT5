# Extending Features Tutorial

This tutorial shows how to create a custom feature.

```{doctest}
>>> class MyFeature:
...     def compute(self, x):
...         return x * 2
>>> MyFeature().compute(3)
6
```
