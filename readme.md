# AgentDS-BUS

**AgentDS-BUS: a fine-tuning-free agentic breast ultrasound malignancy
classification framework with decoupled segmentation and feature analysis**

## Overview

AgentDS-BUS is a fine-tuning-free framework for breast ultrasound malignancy
classification. It separates lesion localization and segmentation from imaging
feature analysis, then combines the resulting information for clinical decision
support with vision-language models.

The repository includes the AgentDS-BUS pipeline together with ROI-guided,
image-only, and feature-reasoned comparison workflows.

## Repository structure

```text
AgentDS-BUS/
├── bbox accuracy/
│   ├── Qwen3VL.py
│   ├── SAM3.py
│   ├── bbox_lingshu_medgemma.py
│   └── generate_sam3_masks.py
└── clinical decision support/
    ├── agentic_augmented.py
    ├── agentic_augmented_majority_voting.py
    ├── agentic_augmented_lingshu_medgemma.py
    ├── agentic_naive.py
    ├── end_to_end.py
    ├── lingshu_medgemma.py
    └── step_by_step_naive.py
```

- `bbox accuracy/` contains lesion localization, segmentation, and mask
  generation workflows.
- `clinical decision support/` contains AgentDS-BUS and the comparison
  classification pipelines.

## Usage

Install the model dependencies required by the selected workflow, then set the
model and dataset paths in the corresponding Python script.

For the consensus-gated AgentDS-BUS workflow, first generate the SAM3 masks:

```bash
python "bbox accuracy/generate_sam3_masks.py" \
  --data_root /path/to/data \
  --data_index_dir /path/to/index_tables \
  --sam3_model_path /path/to/sam3 \
  --resume
```

Then run the classification pipeline:

```bash
python "clinical decision support/agentic_augmented_majority_voting.py"
```

Other localization and classification workflows can be run directly after
configuring the paths in their corresponding scripts, for example:

```bash
python "bbox accuracy/Qwen3VL.py"
python "clinical decision support/agentic_augmented.py"
python "clinical decision support/agentic_naive.py"
python "clinical decision support/end_to_end.py"
```
