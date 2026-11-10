# 🧠 Wilson–Cowan E–I Model: Theta Rhythm & Meditation

> A computational study of the simplified Wilson–Cowan excitatory–inhibitory model, applied to theta-band (4–8 Hz) oscillations observed in EEG during focused-attention meditation.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/scipy-%230C55A5.svg?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-%2311557c.svg?logo=matplotlib&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## 📖 Overview

The project builds a chain of **four claims**:

1. 🎯 **Fixed points & limit cycles** — a single E–I region settles to a steady state or sustains an oscillation depending on its parameters.
2. 🔀 **Bifurcations** — the transition between regimes occurs through identifiable saddle-node and Hopf bifurcations.
3. 🌊 **Theta band & meditation** — with appropriate time constants the oscillation falls in the theta band, and a "meditative" parameter setting shows stronger, more regular theta power than a "resting" setting.
4. 🔗 **Synchronization** — two coupled regions synchronize when the coupling strength is large enough.

> ⚠️ **Note**: The model is a mean-field simplification of Wilson & Cowan (1972) with no refractory term and no spatial extent. The comparison with EEG is **qualitative**, not a validation.

---

## ⚙️ Requirements

- 🐍 Python 3.10+
- 🔢 numpy
- 🧮 scipy ≥ 1.6
- 📊 matplotlib

Install:

```bash
pip install -r requirements.txt
```
