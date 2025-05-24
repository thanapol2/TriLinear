
<h2 align="center">TriLinear: Time Series Anomaly Detection Using Tricube Smoothing Decomposition and a Linear Forecasting Model</h2>
<div align="center">
  
![Python 3.12](https://img.shields.io/badge/python-3.12-green.svg?style=plastic)
![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg?style=plastic)

</div>

---

## 🆕 Main Recent Update

- **[22-MAY-2024]** Uploaded source code for the double-blind review phase of **ADMA 2025**.

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
- **Kepler and M-dwarfs**: Only two samples are included in this release for the double-blind review phase. The full dataset will be made available after acceptance.

---

## 📌 Note

This version of **TriLinear** does not include encapsulated packaging or modular deployment features. A full version will be released after the double-blind review process is complete.

---
