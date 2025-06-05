<p align="center">
  <h1 align="center">Human-in-Context: Unified Cross-Domain 3D Human Motion Modeling via In-Context Learning</h1>

# Installation and Preparation

## 1. Installation
```
conda create -n human_in_context python=3.9 anaconda
conda activate human_in_context
pip install -r requirements.txt
```

## 2. Download checkpoint and support data


Download all 3 files [here](https://drive.google.com/drive/folders/1UzFbrMtzAmvJKvuDi63yz72iSdQqGjvq?usp=drive_link), and put them such that the root directory looks like this:

```
./
│
├── ckpt/
│   └── hic_pretrained_ep69.bin
│
└── data/
    |
    └── support_data/
        |
        ├── anchor_collection/
        |   └── anchor_collection.pkl
        |
        └── mesh/
            └── SMPL_NEUTRAL.pkl
```

# Run Demo

```
python app.py
```

## Demo Instruction

Step 1. Select a query input (2d pose / 3D pose / mesh) from the dropdown menu.

Step 2 (Optional). Push the button to visualize query input.

Step 3. Select a task from the dropdown menu. The menu will pop up once you select a query input. Available tasks depend on the selected input.

Step 4. Push the ''Run Inference'' to start the inference on the selected input and task.