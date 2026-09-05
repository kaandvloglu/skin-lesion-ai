# AI Module

## Dataset Setup

The project uses the **MILK10K** skin lesion dataset for training.

### Folder structure

```text
ai/
└── data/
    └── MILK10K/
        ├── MILK10K_Training_Input/
        ├── MILK10K_Training_GroundTruth.csv
        ├── MILK10K_Training_Metadata.csv
        └── MILK10K_Training_Supplement.csv
```

### Setup steps

1. Download the **MILK10K** dataset.
2. Extract the files into `ai/data/MILK10K/`.
3. Verify that the folder structure matches the example above before running the training pipeline.

> The dataset is intentionally excluded from GitHub because of its size.