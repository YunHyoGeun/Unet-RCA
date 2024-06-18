# UNet Cell Segmentation

This repository contains code for training and inference of a UNet model for cell segmentation using PyTorch. The input images and masks are divided into 256x256 patches, augmented, and used for training. The training process includes logging with TensorBoard and saving model checkpoints.

## Requirements

- Python 3.8 or higher
- Conda
- CUDA (if using GPU)

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/unet-cell-segmentation.git
cd unet-cell-segmentation
```

2. **Create and activate the conda environment:**

```bash
conda create -f environment.yml
conda activate torch
```

3. **Prepare your dataset:**

   - Place your input images in the directory: `/mnt/c/Users/user/Desktop/Unet/DIC_형광/Img`
   - Place your masks in the directory: `/mnt/c/Users/user/Desktop/Unet/DIC_형광/Mask`

## Training

To train the model, run:

```bash
python main.py train
```

This script will:

- Load and preprocess the dataset.
- Train the UNet model.
- Save checkpoints to `unet_checkpoint.pth`.
- Log training details to TensorBoard in the `runs/unet_experiment` directory.

You can monitor the training progress with TensorBoard:

```bash
tensorboard --logdir=runs/unet_experiment
```

## Inference

To perform inference using a trained model, run:

```bash
python main.py infer
```

This script will:

- Load the trained model from `unet_checkpoint.pth`.
- Perform inference on the dataset.
- Save or display the predicted masks.

## File Structure

```
unet-cell-segmentation/
│
├── main.py           # Script for everything
├── environment.yml   # Required Python packages
└── README.md         # This file
```

## Customization

You can customize various aspects of the training and inference process by modifying the parameters in `main.py`, such as:

- Batch size
- Number of epochs
- Learning rate
- Paths to input images and masks
- Augmentation settings

## Acknowledgements

This code is inspired by various UNet implementations available in the deep learning community. If you have any questions or encounter any issues, please feel free to open an issue or contact the author.

## License

This project is licensed under the MIT License - see the LICENSE file for details.