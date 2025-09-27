# Repository Synchronization Overview

This project is synchronized with the upstream repository hosted at [yegorsokolov/MT5](https://github.com/yegorsokolov/MT5). This is the source for upstream updates and the destination for uploading operational logs.

## Deployment Environment

* The target VPS environment used for deployments provides SSH access. This allows for secure automation of synchronization tasks and log uploads.
* Ensure that deployment scripts or CI pipelines are configured with the required SSH credentials before attempting to push updates or upload logs.

## Recommended Workflow

1. Fetch upstream changes from `https://github.com/yegorsokolov/MT5` before starting local development.
2. Apply project updates locally and verify the changes using the available test suites.
3. Use the provisioned SSH access on the deployment VPS to upload logs and coordinate rollouts.
4. After verification, push changes back to the upstream repository to keep the deployment environment and source control in sync.

Keeping the repository and deployment environment aligned ensures reliable monitoring and traceable change history for the MT5 project.
