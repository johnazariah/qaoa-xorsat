# Protocol Index

This folder contains the project's protocol documents for testing, experiments,
benchmarks, reproduction, and optimisation-result preservation.

Use this file as the entry point when deciding which document governs a given
activity.

## Files

| File | Role | Use when |
| --- | --- | --- |
| [`testing-benchmarking-policy.md`](./testing-benchmarking-policy.md) | Canonical policy document. | You need the operational rules for what counts as verification, experimentation, benchmarking, reproduction, and result preservation. |
| [`testing-protocol.md`](./testing-protocol.md) | Formal explanatory overview of the testing stack. | You want the readable walkthrough of how verification, experimentation, benchmarking, and reproduction fit together. |
| [`experimentation-benchmarking-protocol.md`](./experimentation-benchmarking-protocol.md) | Methodology for exploratory and benchmark-grade numerical runs. | You are planning or interpreting experiments or controlled benchmarks. |
| [`reproduction-protocol.md`](./reproduction-protocol.md) | Methodology for validating against trusted external targets. | You are checking whether the pipeline recovers literature-backed or historically trusted values. |
| [`optimization-data-protocol.md`](./optimization-data-protocol.md) | Archive and schema protocol for preserved optimisation runs. | You need to know how run artefacts are stored, what fields mean, or how historical optimisation data should be interpreted. |

## How These Documents Relate

1. Start with [`testing-benchmarking-policy.md`](./testing-benchmarking-policy.md) for the rules.
2. Read [`testing-protocol.md`](./testing-protocol.md) for the overall testing lifecycle.
3. Use [`experimentation-benchmarking-protocol.md`](./experimentation-benchmarking-protocol.md) for experiment and benchmark methodology.
4. Use [`reproduction-protocol.md`](./reproduction-protocol.md) when the goal is external validation.
5. Use [`optimization-data-protocol.md`](./optimization-data-protocol.md) when preserving or analysing optimisation archives.

## Related Documents Outside This Folder

- [`../testing-register.md`](../testing-register.md): inventory of implemented tests
- [`../results/optimization/`](../results/optimization/): preserved optimisation artefacts governed by the optimisation data protocol
