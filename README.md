# Machine Learning

This repository contains **pure Python-based machine learning implementations**, structured strictly around the actual code and experiments present in the repository.  
No external theory, no generic ML coverage — only what is implemented here.

The repository is organized into **four folders**, each serving a clearly defined purpose.

---

## 📁 supervised learning

This folder contains **supervised learning implementations using labeled data**.  
The focus here is strictly on **classification and regression**, and the folder collectively covers **6–7 different supervised models**, implemented from scratch or using core libraries.

### What is present here:
- **Classification models** (multiple algorithms)
- **Regression models** (multiple algorithms)
- Each model is implemented independently with:
  - Data preprocessing
  - Model training
  - Prediction
  - Performance evaluation

All implementations are algorithm-focused and demonstrate how different supervised models behave on data.

---

## 📁 unsupervised learning

This folder contains **only one unsupervised learning technique**, implemented and explored in depth.

### Technique used:
- **K-Means Clustering**

### What is done:
- Data clustering using K-Means
- Distance-based grouping of samples
- Observation of cluster formation and convergence behavior

No other unsupervised algorithms are included — this folder is **exclusively focused on K-Means**.

---

## 📁Model Development

This section focuses on **training dynamics and optimization**, rather than introducing new architectures.  
The primary contribution here is addressing **gradient-related issues through dynamic learning rate strategies**.

### Key idea
- Start with a **higher learning rate** for faster initial learning
- **Automatically decrease the learning rate** during later stages to improve stability and accuracy
- Learning rate adjustment is handled by internal logic and makes traning fast

### IMP Python files included
- `dynamic learn.py`
- `dynamic learn2.py`
- `dynamic learn3.py`

These files demonstrate different ways of implementing **dynamic learning rate behavior** during training.

---

## 📁Mini Projects

This folder contains **small, self-contained machine learning projects** designed to reflect **real-world application scenarios**.

Each project typically includes:
- Problem formulation
- Data handling
- Model training
- Evaluation and observation

---

Thank you for going through the repository.  
