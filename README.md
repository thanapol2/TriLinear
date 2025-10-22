
<h2 align="center">TriLinear: Time Series Anomaly Detection Using Tricube Smoothing Decomposition and a Linear Forecasting Model</h2>
<div align="center">
  
![Python 3.12](https://img.shields.io/badge/python-3.12-green.svg?style=plastic)
![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg?style=plastic)
![ADMA 2025](https://img.shields.io/badge/ADMA-2025-blue.svg?style=plastic)


</div>

---

## 🆕 Main Recent Update
- **[03-SEP-2025]** Uploaded full dataset and VUS-ROC image for Kepler and M-Dwarfs dataset
- **[12-JUL-2025]** This paper was accepted in [ADMA 2025](https://adma2025.github.io/)
- **[22-MAY-2025]** Uploaded source code for the double-blind review phase of **ADMA 2025**.
- **[29-MAY-2025]** Uploaded example source codes for TriLinear


---

## 🔧 Requirements

- Python 3.12  
- matplotlib == 3.10.3 
- numpy == 1.26.4 
- pandas == 2.2.3
- torch == 2.7.0  
- periodicity-detection == 0.1.3  
- TSB-AD == 1.5  

---

## 📁 Folder Structure

    .
    ├── datasets/                   # Datasets used in this paper
    │     ├── NAB_KDD               # Benchmark datasets: NAB and KDD
    │     └── light_curves          # Sample light curve data (Kepler & M-dwarfs)
    ├── evaluation/                 # Code for evaluation and reproduction
    ├── figures/                    # Scripts to reproduce figures from the paper
    ├── src/                        # Core source files for TriLinear implementation
    └── README.md


## 📂 Dataset

- **NAB and KDD**: Available from the [TSB-AD GitHub repository](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets).  
- **light curve**:
  - Synthetic Kepler light curves are stored in the `Kepler/` folder. Each file follows the naming pattern: `[AA]_H[BB]_D[CC]_[DD]_[EE]`
  **Filename structure:**
    - `AA` – Original light curve name  
    - `BB` – Height of the injected flare  
    - `CC` – Duration of the flare  
    - `DD` – Index of the start of the rising phase  
    - `EE` – Index of the start of the decay phase
  - M-dwarfs**: This GitHub repository includes two sample files from 18 flares. For the full dataset and detailed analysis, please refer to the publication:t [Fast optical flares from M dwarfs detected by a one-second-cadence survey with Tomo-e Gozen](https://academic.oup.com/pasj/article/74/5/1069/6656381). 

---

## 📊 Evaluation

We provide reproducible evaluation code to compare **TriLinear** against existing TSAD methods across datasets.

### Metrics:
We evaluate methods using:
- **VUS-ROC** (Volume Under the Surface for ROC), Ref: [VUS-ROC](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)

### Settings:
- **Unsupervised** methods: Trained on the entire time series.
- **Semi-supervised** methods: Few-shot learning setup (**trained on 10–20% normal data**).
- **Reproducibility**: All experiments were conducted on a single NVIDIA RTX 5070 GPU. For testing purposes, the environment was also verified on an RTX 2060 GPU.

---

## BibTex
- If you plan to use or apply our source code, please cite our published paper.
```
@InProceedings{10.1007/978-981-95-3456-2_2,
	author="Phungtua-eng, Thanapol
	and Arima, Noriaki
	and Yamamoto, Yoshitaka",
	title="TriLinear: Time Series Anomaly Detection Using Tricube Smoothing Decomposition and a Linear Forecasting Model",
	booktitle="Advanced Data Mining and Applications",
	year="2026",
	publisher="Springer Nature Singapore",
	address="Singapore",
	pages="19--33",
	isbn="978-981-95-3456-2"
}


```

---
