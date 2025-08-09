# Configuration Options

| Parameter | Description | Default | Valid Range |
| --- | --- | --- | --- |
| seed | Random seed for reproducibility | **42** |  |
| risk_per_trade | Fraction of capital risked per trade | **required** | **> 0, <= 1** |
| symbols | List of trading symbols, e.g. ['EURUSD', 'GBPUSD'] | **required** |  |
| ddp | Enable DistributedDataParallel if true, auto-detect if null | **None** |  |
