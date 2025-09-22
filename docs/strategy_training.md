# Strategy Training

The strategy training utilities can be customised through a
`StrategyConfig` dataclass. These options influence how prompts are
constructed during strategy generation.

## Sample configuration

### YAML

```yaml
budget_limit: 10000
risk_tolerance: 0.05
custom_sections:
  notes: "Focus on liquid assets"
  constraints: "No leverage"
```

### JSON

```json
{
  "budget_limit": 10000,
  "risk_tolerance": 0.05,
  "custom_sections": {
    "notes": "Focus on liquid assets",
    "constraints": "No leverage"
  }
}
```

## CLI usage

Pass the configuration file to the RL training script to inject these
parameters into generated strategy prompts:

```bash
python -m mt5.train_rl --strategy-config my_strategy.yaml
```

When provided, the values from the configuration are embedded into the
prompt sections and any `custom_sections` are appended.
