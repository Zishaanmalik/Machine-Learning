# Machine Learning

This repository presents a structured collection of **machine learning implementations in Python**, focused on supervised learning, unsupervised learning, experimental model development, and applied mini projects.  
All work is implementation-driven and emphasizes understanding learning behavior through code.

---

## 📁Supervised Learning

Supervised learning deals with training models on **labeled data**, where the input–output relationship is known.  
In this repository, both **classification** and **regression** aspects are covered.

### Covered aspects
- **Classification models**
- **Regression models**

### Models implemented
- Logistic Regression / Classification
- Bagging  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- Support Vector Machine (SVM)

Each model is implemented and evaluated independently to study behavior across different supervised settings.

---

## 📁Unsupervised Learning

Unsupervised learning works with **unlabeled data**, aiming to discover hidden structures or patterns without predefined targets.

### Technique used
- **K-Means Clustering**

The implementation focuses on:
- Distance-based clustering
- Cluster formation and convergence
- Visual interpretation of clusters to understand data distribution

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

This section contains **practical, implementation-focused machine learning projects**.  
Each project is built as a **working prototype**, emphasizing end-to-end flow rather than isolated algorithms.

Key characteristics:
- Problem-oriented implementation
- Realistic data handling and preprocessing
- Model training and evaluation in a practical setup
- Focus on applying ML concepts in usable, real-world-style scenarios
  
---

Thank you for going through the repository.  
