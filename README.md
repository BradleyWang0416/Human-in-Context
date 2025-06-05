<p align="center">
  <h1 align="center">Human-in-Context: Unified Cross-Domain 3D Human Motion Modeling via In-Context Learning</h1>

# 😃Run

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
    └── support_data/
        ├── anchor_collection/
        |   └── anchor_collection.pkl
        └── mesh/
            └── SMPL_NEUTRAL.pkl
```

## 4. Run the demo