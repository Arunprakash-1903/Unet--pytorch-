# 🧬 UNet - PyTorch

A lightweight PyTorch implementation of the **U-Net** architecture for image segmentation tasks.  
This project demonstrates how to build, train, and test a U-Net model using clean, minimal code.


![Tux, the Linux mascot](https://viso.ai/wp-content/uploads/2024/04/uNet-architecture.png)
---

## 🧠 Features
- ✅ Implementation of the classic **U-Net** model (`unet.py`)
- 📂 Dataset loader for segmentation tasks (`dataset.py`)
- 🏋️ Training script with checkpoint saving (`train.py`)
- ⚙️ Utility functions for metrics and visualization (`utils.py`)
- 💡 Easy to modify and extend for any custom dataset

---

## ⚙️ Requirements

Make sure you have the following installed:

- Python 3.8+
- PyTorch
- NumPy
- Pillow (PIL)

Install dependencies:

```bash
pip install torch numpy pillow
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Arunprakash-1903/Unet--pytorch-
cd Unet--pytorch-
```

### 2️⃣ Prepare Your Dataset
Organize your dataset with **images** and **segmentation masks** in separate folders.  
Adjust paths in `dataset.py` if needed.

Example structure:
```
dataset/
 ├── images/
 │    ├── img1.png
 │    ├── img2.png
 └── masks/
      ├── mask1.png
      ├── mask2.png
```

Make sure each image has a corresponding mask file with the same filename.

---

## 🏋️ Training

Run the training script:
```bash
python train.py --data_dir path/to/dataset --epochs 50 --batch_size 8
```

You can modify hyperparameters (like learning rate, optimizer, loss function) inside `train.py` or pass them as arguments.

Example:
```bash
python train.py --data_dir ./dataset --epochs 100 --batch_size 4 --lr 0.0001
```

---

## 🧩 Project Structure
```
Unet--pytorch-
│
├── unet.py        # U-Net model architecture
├── dataset.py     # Dataset loading and preprocessing
├── train.py       # Training loop, loss, and optimizer
├── utils.py       # Helper functions (metrics, saving masks, etc.)
└── README.md      # Project documentation
```

---

## 🧪 Inference / Testing

Once training is done, you can use the saved model checkpoint for segmentation on new images.

Example:
```python
import torch
from unet import UNet
from PIL import Image
import numpy as np

# Load model
model = UNet()
model.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))
model.eval()

# Load and preprocess image
image = Image.open('test_image.png').convert('L')
image_tensor = torch.from_numpy(np.array(image) / 255.0).unsqueeze(0).unsqueeze(0).float()

# Run inference
with torch.no_grad():
    output = model(image_tensor)

# Convert output to image
output_mask = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255
Image.fromarray(output_mask).save('predicted_mask.png')
```

---


