# System Architecture

This overview shows how data flows through MT5 from ingestion to execution and outlines the plugin system.

## Data Flow

```
+-----------------+     +---------------+     +-----------------+     +----------------+     +------------------+
| Data Ingestion  +---->+ Feature Store +---->+ Training Pipeline+---->+ Strategy Engine+---->+ Execution Broker |
+-----------------+     +---------------+     +-----------------+     +----------------+     +------------------+
```

## Plugins

- Indicator plugins in `indicators/`
- Strategy plugins in `strategies/`
- Data source plugins under `plugins/`
- Broker integrations in `brokers/`

## Core Components

- Feature store
- Training pipeline
- Strategy engine
- Execution and risk modules

## Configuration Options

```{autoclass} config_models.AppConfig
:members:
:undoc-members:
:show-inheritance:
```
