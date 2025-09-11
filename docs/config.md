# Configuration Options

| Parameter | Description | Default | Valid Range |
| --- | --- | --- | --- |
| seed | Random seed for reproducibility | **42** |  |
| risk_per_trade | Fraction of capital risked per trade | **required** | **> 0, <= 1** |
| symbols | List of trading symbols, e.g. ['EURUSD', 'GBPUSD'] | **required** |  |
| ddp | Enable DistributedDataParallel if true, auto-detect if null | **None** |  |
| cross_asset.max_pairs | Limit cross-asset feature generation to top-K pairs | **50** | *int* or **null** |
| cross_asset.reduce | Reduction strategy for cross-asset pairs: `top_k` keeps most correlated pairs, `pca` compresses interactions | **pca** | `top_k`, `pca` |
