# Multi-Temporal Sentinel-2 Land Cover Classification in Boreal Southern Finland

End-to-end machine learning pipeline comparing classical baselines against a fine-tuned Prithvi-EO-2.0 geospatial foundation model for land cover classification on Sentinel-2 tile T35WMP (southern Finland), with CORINE Land Cover 2018 as ground truth.

This is the final project for **RAF622M Machine Learning for Earth Observation with Supercomputers**, University of Iceland.

**Author:** Ólafur Sær Sigursteinsson (oss27@hi.is)


## Project summary

- **Study area:** Sentinel-2 tile T35WMP (~110 x 110 km, southern Finland, centred at 61.5°N, 27.5°E)
- **Sensor:** Sentinel-2 L2A, four cloud-filtered acquisitions in 2018 (March, May, August, October)
- **Bands used:** B02, B03, B04, B08 (Blue, Green, Red, NIR. All native to 10 meters)
- **Labels:** CORINE Land Cover 2018, filtered to 8 classes with >= 200 samples
- **Patches:** 3 × 3 pixels at 10 meter resolution, total ~40k patches across train/val/test
- **Models compared:** Random Forest (naive + balanced), MLP (naive + balanced), Prithvi-EO-2.0 (fine-tuned)

For the full methodology and results, see the report PDF.


## Reproducing the pipeline on JURECA

### 1. JURECA environment setup

The following module sequence must be loaded in this order on every login:

```
module load Stages/2025
module load Intel/2024.2.0-CUDA-12
module load Python/3.12.3
module load PyTorch/2.5.1
source /p/project1/training2600/<USER>/envs/finalproject/bin/activate
```

The Python virtual environment includes:
- `torch==2.5.1+cu124`
- `torchvision==0.20.1+cu124`
- `terratorch==1.2.6`
- `lightning==2.6.1`
- `scikit-learn`, `imbalanced-learn`, `rasterio`, `numpy`, `matplotlib`, `seaborn`

NOTE: `torch` and `torchvision` must be installed in the venv (not the system) to avoid `libpython3.12.so` linker errors.

### 2. Dataset locations on JURECA

All data lives under `/p/project1/training2600/<USER>/`:

| Path | Contents 
| `data/`                         | Raw Sentinel-2 L2A scenes and the CORINE 2018 raster (the Checkpoint 1 outputs) 
| `training_data_checkpoint/`     | Patches and labels from Checkpoint 2 (`.npz` and `.npy` files, about 40k patches) 
| `Final_Project/processed_data/` | TerraTorch-formatted dataset (one `.npy` per patch, organised by class) 
| `Final_Project/checkpoints/`    | Best Prithvi checkpoint (`prithvi_best.pt`) saved during training (gets overwritten by a better one)
| `Final_Project/results/`        | All metrics, predictions, and figures 


### 3. Run order

The notebooks must be run in numerical order, as each relies on artifacts from the previous book:

|Step| Notebook                     | Inputs                           | Outputs 
|---|                            ---|                               ---|                                                            ---|
| 1 | `01_data_preparation.ipynb`   | `training_data_checkpoint/*.npz` | `Final_Project/processed_data/{train,val,test}/<class>/*.npy` 
| 2 | `02_baseline_models.ipynb`    | `processed_data/`                | `results/all_results.json` (baseline entries), trained model `.pkl` files 
| 3 | `03_prithvi_finetuning.ipynb` | `processed_data/`                | `checkpoints/prithvi_best.pt`, `results/prithvi_*.npy`, updated `all_results.json` 
| 4 | `04_figures.ipynb`            | All saved artifacts above        | `results/fig*.png` 

Total runtime: ~10 minutes for steps 1, 2, 4. Step 3 (Prithvi fine-tuning, 20 epochs) takes about ~30 minutes


### 4. Hardware

Training was performed on the JURECA `dc-gpu` partition (Quadro RTX 8000, 47.6 GB VRAM).


## Reproducing locally (without JURECA)

The pipeline can run on any machine with a CUDA-capable GPU and the dependencies above. The dataset paths in the notebooks would need to be adjusted from `/p/project1/training2600/<USER>/` to local paths.

For testing-only purposes, an RTX 5090 was used during development to verify code correctness before deploying to JURECA. A consumer GPU is sufficient for the full pipeline given that the dataset is small (~20k training patches, 16-channel 3 × 3 patches).


## Key results

| Model | Test OA | Test Macro F1 |
|                              ---|        ---|        ---|
| Random Forest (naive)           | 0.491     | 0.136     |
| MLP (naive)                     | 0.480     | 0.126     |
| Random Forest (balanced)        | 0.238     | 0.122     |
| MLP (balanced)                  | 0.210     | 0.122     |
| **Prithvi-EO-2.0 (fine-tuned)** | 0.324     | 0.137     |

All five models converged on macro F1 between 0.122 and 0.137 despite very different architectures, suggesting an information-driven rather than model-capacity ceiling. Please look at the report for a more detailed discussion.


## Reproducing without JURECA
Full reproduction outside JURECA requires re-downloading the four source Sentinel-2 scenes from the Copernicus Data Space Ecosystem and the CORINE 2018 raster from the copernicus Land Monitoring Service,
then running the local checkpoint 1&2 notebooks to build the patch dataset before the final project notebooks can run. The pipeline also assumes a CUDA-capable blackwell GPU (RTX 5090). So i 










