# DamageCAT

A categorical typology-based building damage classification framework using satellite imagery and deep learning. This repository contains the implementation of the DamageCAT framework for building damage assessment from satellite imagery.

## Overview

DamageCAT is a deep learning framework for building damage assessment that:

- Classifies building damage into multiple categories
- Uses pre- and post-disaster satellite imagery
- Implements a transformer-based architecture for accurate damage assessment

## Requirements

- Python 3.11
- PyTorch
- torchvision
- numpy
- opencv-python (cv2)
- Pillow (PIL)
- scikit-learn
- matplotlib
- einops
- tifffile

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Data Preparation

The data should be organized in the following structure:

```bash
data/damagecat/
├── train/
│ ├── images/
│ │ ├── pre_0.png
│ │ ├── pre_1.png
│ │ ├── pre_2.png
│ │ ├── pre_3.png
│ │ └── ...
│ └── masks/
│ │ ├── pre_0.png
│ │ ├── pre_1.png
│ │ ├── pre_2.png
│ │ ├── pre_3.png
│ │ └── ...
└── test/
   ├── pre_0.png
   ├── pre_1.png
   ├── pre_2.png
   ├── pre_3.png
   └── ...
```

## Usage

### Training

To train the model, use the script in `scripts/run_cd.sh`:

```bash
bash scripts/run_cd.sh
```

Key parameters in the training script:

- `img_size`: Image size (default: 512)
- `batch_size`: Batch size (default: 8)
- `max_epochs`: Maximum training epochs (default: 200)
- `lr`: Learning rate (default: 0.001)
- `n_class`: Number of damage classes (default: 5)
- `net_G`: Network architecture (default: newUNetTrans)

### Evaluation

To evaluate the model and make predictions, make sure you have the pre-trained model in the `checkpoints/your_project_name` folder, have the test images in the `data/damagecat/test/` folder, and the use the script in `scripts/eval.sh`:

```bash
bash scripts/eval.sh
```

Key parameters in the evaluation script:

- `dataset`: Dataset name (default: DamageCAT)
- `data_name`: Data name (default: x)
- `batch_size`: Batch size (default: 8)

## Model Architecture

The framework uses a transformer-based architecture (newUNetTrans) that combines:

- U-Net backbone
- Transformer encoder-decoder
- Multi-scale feature fusion

## Pre-trained Models

[Link to pre-trained models will be added]

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{xiao2025damagecatdeeplearningtransformer,
      title={DamageCAT: A Deep Learning Transformer Framework for Typology-Based Post-Disaster Building Damage Categorization}, 
      author={Yiming Xiao and Ali Mostafavi},
      year={2025},
      eprint={2504.11637},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.11637}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work is based on the DAHiTra framework developed by Navjot Kaur. We would like to thank [@nka77](https://github.com/nka77/DAHiTra) for their pioneering work on transformer-based building damage assessment. Our implementation builds upon their codebase and extends it for our specific use case.

The original DAHiTra paper can be found at:

- Journal: [CACAIE](https://onlinelibrary.wiley.com/doi/10.1111/mice.12981)
- ArXiv: [2208.02205](https://arxiv.org/abs/2208.02205)
