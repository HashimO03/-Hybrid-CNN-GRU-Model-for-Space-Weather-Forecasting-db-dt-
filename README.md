# 🌌 Hybrid CNN-GRU Model for Space Weather Forecasting (db/dt)
**University of Sheffield — BEng Computer Systems Engineering**
**Supervisor: Professor Michael Balikhin**

## Overview
A hybrid **CNN-GRU** deep learning model for one-minute-ahead forecasting of the geomagnetic field rate of change **(db/dt)**. Rapid db/dt variations induce geomagnetically induced currents (GICs) in power grids and pipelines — accurate short-horizon forecasts are critical for infrastructure protection.

Trained on **NASA OMNI** solar wind data and **INTERMAGNET** magnetometer readings (Abisko, Sweden), with geographic transfer testing on **Boulder, USA**.

---

## Architecture
Three 1D-Conv layers (5→32→64→128) with BatchNorm and Dropout extract cross-channel spatial features, feeding into a 2-layer GRU (hidden=128) for temporal modelling. Final linear layer outputs db/dt in nT/min. ~790k parameters, <20ms inference on GPU.

## Input Features
5 selected from 10 candidates via correlation analysis + gradient saliency: `VBs`, `Pressure`, `Bmag`, `Proton_Temp`, `By`

## Training
Adam optimizer · Smooth L1 (Huber) loss · ReduceLROnPlateau · Early stopping · 5-fold cross-validation · Mixed precision (float16) · NVIDIA RTX 3050

---

## Results

| Model | Abisko MAE | Abisko RMSE | Boulder MAE | Boulder RMSE |
|---|---|---|---|---|
| **CNN-GRU** | **2.094** | **3.835** | **0.229** | **0.342** |
| LSTM | 2.208 | 4.051 | — | — |
| MLP | 2.516 | 4.700 | — | — |

*Metrics in nT/min. CNN-GRU outperforms best baseline by ~5% on both metrics.*

---
<img width="676" height="318" alt="image" src="https://github.com/user-attachments/assets/51202b4c-1554-4685-8329-33d8ad8fb72a" />

## Stack
`Python` · `PyTorch` · `NumPy` · `Pandas` · `Scikit-learn` · `Matplotlib` · `CUDA 12.1`
