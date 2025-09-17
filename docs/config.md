# Configuration Options

| Parameter | Description | Default | Valid Range |
| --- | --- | --- | --- |
| training.seed |  | **42** |  |
| training.model_type | Primary model architecture to train (lgbm, neural, cross_modal) | **lgbm** |  |
| training.use_pseudo_labels |  | **False** |  |
| training.use_focal_loss |  | **False** |  |
| training.focal_alpha |  | **0.25** | **>= 0.0** |
| training.focal_gamma |  | **2.0** | **>= 0.0** |
| training.num_leaves |  | **None** | **>= 2** |
| training.learning_rate |  | **None** | **> 0** |
| training.max_depth |  | **None** | **>= 1** |
| training.pt_mult |  | **0.01** | **> 0** |
| training.sl_mult |  | **0.01** | **> 0** |
| training.max_horizon |  | **10** | **>= 1** |
| training.balance_classes |  | **False** |  |
| training.time_decay_half_life |  | **None** | **>= 1** |
| training.drift.method | Concept drift detection method | **adwin** |  |
| training.drift.delta | Sensitivity of the detector | **0.002** | **> 0** |
| training.drift.threshold | Number of drift events required before reacting | **3** | **>= 1** |
| training.drift.cooldown | Seconds to wait before reacting again | **3600.0** | **>= 0.0** |
| training.feature_includes | Feature columns to always include in training | **[]** |  |
| training.feature_excludes | Feature columns to drop from training | **[]** |  |
| training.feature_families | Map of feature family name to inclusion flag | **{}** |  |
| training.use_feature_selector | Run analysis.feature_selector.select_features on candidates | **True** |  |
| training.feature_selector_top_k | Optional top-k cutoff when using the feature selector | **None** | **>= 1** |
| training.feature_selector_corr_threshold | Correlation threshold for dropping redundant features | **0.95** | **>= 0.0, <= 1.0** |
| features.latency_threshold |  | **0.0** | **>= 0.0** |
| features.features |  | **[]** |  |
| strategy.symbols |  | **required** |  |
| strategy.risk_per_trade |  | **required** | **> 0, <= 1** |
| strategy.session_position_limits |  | **{}** |  |
| strategy.default_position_limit |  | **1** | **>= 0** |
| strategy.use_kalman_smoothing |  | **False** |  |
| strategy.risk_profile.tolerance | Risk tolerance multiplier | **1.0** | **>= 0.0** |
| strategy.risk_profile.leverage_cap | Maximum allowed leverage for positions | **1.0** | **>= 0.0** |
| strategy.risk_profile.drawdown_limit | Fractional drawdown at which positions are force-closed | **0.0** | **>= 0.0** |
| services.service_cmds |  | **{}** |  |
| active_learning.k | Number of candidates to query | **10** | **>= 1** |
| active_learning.min_confidence | Skip examples whose model confidence falls below the threshold | **0.5** | **>= 0.0, <= 1.0** |
| active_learning.exploration_bias | Weight given to sampling unexplored regions | **0.0** | **>= 0.0** |
| mlflow.tracking_uri | Remote MLflow tracking URI; empty string keeps local logging | **** |  |
| mlflow.username | MLflow username when basic auth is enabled | **** |  |
| mlflow.password | MLflow password for basic auth | **** |  |
| alerting.slack_webhook | Slack webhook URL used for alert delivery | **None** |  |
| alerting.smtp.host |  | **** |  |
| alerting.smtp.port |  | **587** | **>= 0, <= 65535** |
| alerting.smtp.username |  | **** |  |
| alerting.smtp.password |  | **** |  |
| alerting.smtp.sender |  | **** |  |
| alerting.smtp.recipients |  | **[]** |  |
| alerting.smtp.use_tls |  | **True** |  |
