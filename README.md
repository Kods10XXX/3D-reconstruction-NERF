# Parallel NeRF: 3D Scene Reconstruction

A Python and PyTorch project exploring **Neural Radiance Fields (NeRF)** for 3D scene reconstruction, with parallel image preprocessing, GPU-accelerated training, quantitative evaluation, and 360° visualization.

## Overview

NeRF is a neural rendering technique that learns a continuous representation of a 3D scene from multiple 2D images and their camera poses. Once trained, the model can synthesize new views from previously unseen camera positions.

This project implements a NeRF pipeline using the synthetic LEGO scene from the NeRF dataset. It also investigates the impact of CPU multiprocessing on image preprocessing and uses GPU acceleration for neural network training.

## Objectives

- Understand and implement the core components of a NeRF model.
- Compare sequential and parallel image preprocessing.
- Accelerate neural network training using a CUDA-enabled GPU.
- Evaluate reconstruction quality using PSNR and SSIM.
- Generate novel views and a 360° visualization of the reconstructed scene.

## Technologies

- **Language:** Python
- **Deep Learning:** PyTorch
- **GPU Computing:** CUDA
- **Image Processing:** OpenCV, NumPy, Matplotlib
- **Parallel Computing:** Python multiprocessing
- **Evaluation:** PSNR, SSIM
- **Environment:** Google Colab

## Methodology

### 1. Dataset preparation

The project uses the synthetic LEGO scene from the NeRF example dataset, including training images, test images, and camera transformation matrices.

Images are resized to 100 × 100 pixels to reduce computational cost during experimentation.

### 2. Sequential vs. parallel preprocessing

Two approaches are implemented to process the dataset:

- **Sequential:** images are processed one at a time.
- **Parallel:** image processing is distributed across multiple CPU workers.

The speedup is calculated as:

`Speedup = Sequential execution time / Parallel execution time`

### 3. NeRF implementation

The pipeline includes:

- Positional encoding for 3D coordinates and viewing directions.
- A multilayer neural network predicting RGB color and volume density.
- Ray generation from camera parameters.
- Sampling of 3D points along camera rays.
- Volume rendering to synthesize images.
- Model training and checkpoint saving.

### 4. Model evaluation and visualization

The trained model is evaluated on test views using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

The project also includes training-curve visualization, comparisons between rendered and reference views, and 360° scene visualization.

## Experimental Results

### Parallel preprocessing benchmark

The following measurements were obtained while processing 100 LEGO images:

| Configuration | Execution time |
|---|---:|
| Sequential | 1.563 s |
| Parallel — 4 workers | 1.213 s |
| Measured speedup | 1.29× |

The parallel implementation reduced the measured preprocessing time by approximately 22.4% in this experiment.

### NeRF training and evaluation

| Metric | Result |
|---|---:|
| Training iterations | 5,000 |
| Training images | 100 |
| Image resolution | 100 × 100 |
| Model parameters | 530,052 |
| Mean PSNR on evaluated test views | 22.08 dB |
| Mean SSIM on evaluated test views | 0.850 |

**Hardware used:** Google Colab Tesla T4 GPU, with approximately 15.6 GB of reported VRAM.

These results correspond to the recorded experiment and may vary with hardware, random initialization, and training configuration.

## Repository Structure

```text
nerf-parallel-3d-reconstruction/
├── README.md
├── NeRF_Parallele_Main.ipynb
├── src/
│   └── ...
├── evaluation/
│   └── ...
├── visualization/
│   └── ...
└── results/
    └── ...
```

The source directories and result files should be included when available. The notebook currently references additional modules and Google Drive paths, so the repository must include the required code and adapt the paths before it can run independently.

## Getting Started

### Requirements

- Python 3
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- tqdm
- imageio
- pytorch-msssim
- einops

A CUDA-compatible GPU is recommended for training.

### Running on Google Colab

1. Upload or open `NeRF_Parallele_Main.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Download and extract the required NeRF dataset.
4. Ensure that the project source modules and dataset paths match the notebook configuration.
5. Run the notebook cells in order.
6. Review the preprocessing benchmark, training metrics, rendered views, and 360° visualization.

**Note:** A clean execution requires the supporting Python modules referenced by the notebook, including the training, evaluation, and visualization components.

## Future Improvements

- Make the notebook fully reproducible without hard-coded Google Drive paths.
- Add configurable experiments across multiple scenes.
- Compare different numbers of CPU workers and image resolutions.
- Optimize GPU memory usage and training performance.
- Include reconstruction screenshots and quantitative comparison plots.
- Add automated environment setup and dependency versioning.

  ## Biomedical Extension: NeRF for 3D CT Reconstruction

The second notebook, [`Projet_NeRF_CT_Final.ipynb`](Projet_NeRF_CT_Final.ipynb), extends the original NeRF project to biomedical imaging. It explores how Neural Radiance Fields and GPU acceleration can be applied to the reconstruction and visualization of 3D medical volumes from simulated cone-beam CT projections.

### Objectives

- Adapt the NeRF pipeline to biomedical imaging.
- Generate simulated cone-beam projections from a CT volume.
- Accelerate model training using PyTorch, CUDA, and mixed-precision computation.
- Evaluate the training process and visualize the resulting 3D structures.

### Dataset and Experimental Setup

The experiments use CT data from the **Pancreas-CT dataset (NIH / TCIA)**.

| Parameter | Configuration |
|---|---|
| Input data | CT volume in DICOM format |
| Projection geometry | Cone-beam |
| Projection angles | 90 |
| Detector resolution | 160 × 160 |
| Volume resolution | 128³ |
| Training iterations | 8,000 |
| GPU | NVIDIA T4 |
| Framework | PyTorch |
| Numerical precision | FP16 mixed precision |

### Methodology

1. **CT data preparation:** Load the medical volume and prepare it for the reconstruction pipeline.
2. **Projection simulation:** Generate cone-beam projections using the Beer–Lambert attenuation model.
3. **NeRF training:** Train a neural network on the simulated projections using GPU acceleration and mixed-precision computation.
4. **Visualization and reconstruction:** Generate novel views, visualize reconstructed slices, and extract a 3D surface using the Marching Cubes algorithm.

### Preliminary Results

- The training MSE decreases from approximately \(10^5\) to approximately 20 over 8,000 iterations, with the loss still decreasing at the end of the observed training period.
- Synthesized projections reproduce the overall silhouette and large-scale structures of the reference data, while fine details remain smoothed.
- The pipeline produces novel-view visualizations, a reconstructed slice at \(z = 0\), and a rotating 3D scene.

These results are preliminary and demonstrate the feasibility of adapting the NeRF pipeline to simulated biomedical projection data. They do not establish clinical reconstruction accuracy.

### Technologies

`Python` · `PyTorch` · `CUDA` · `NeRF` · `Cone-Beam CT` · `DICOM` · `FP16 Mixed Precision` · `Marching Cubes` · `Google Colab`

### Notebook

Open [`Projet_NeRF_CT_Final.ipynb`](Projet_NeRF_CT_Final.ipynb) to explore the implementation, training experiments, loss evolution, and visualization results.

## Author

**Kods Bechri**  
Master's student in Data Science  
Interested in Artificial Intelligence, Deep Learning, and Parallel Computing.

[LinkedIn](https://www.linkedin.com/in/kods-bechri/) · [Portfolio](https://kods10xxx.github.io/Portfolio/)

## Acknowledgements

This project builds on the NeRF synthetic example dataset and explores the implementation of neural volumetric rendering using PyTorch.
