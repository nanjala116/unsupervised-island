# 🏝️ Unsupervised Island  
**A Machine Learning Survival Challenge**

> _“You wake up on a strange island. No memory. No signal. Just seven datasets and a ticking clock.”_

Welcome to **Unsupervised Island**, a machine learning survival simulation where your ability to analyze, predict, and adapt determines whether you make it off the island—or not. This repo is your field guide through a jungle of unknown berries, cryptic expedition notes, and high-stakes decisions.

It’s not just about building models. It’s about making them matter.

---

## 📖 Storyline

You have **2 weeks** to figure out which berries are edible, which species are safe, and how to estimate berry size without a computer. The island is rich with data—but it’s raw, unlabeled, and dangerous.

Your only tools:  
- Seven datasets from a lost botanical expedition  
- A few scribbled notes on berry species  
- Your machine learning skills

Can you cluster the unknown? Classify the edible? Predict your next meal?

---

## 🧠 What to Expect

This challenge blends unsupervised learning, classification, regression, and rule-based modeling. You’ll be expected to:

### 🔍 Clustering
- Group berries by similarity using unsupervised techniques  
- Engineer new features to improve cluster quality  
- Explore different feature subsets and cluster counts

### 🍓 Edibility Prediction
- Build classifiers to distinguish edible vs. inedible plants  
- Compare direct prediction vs. species-first inference  
- Evaluate models across leaf, petal, and berry features

### 🌿 Species Classification
- Identify berries by species using expedition notes  
- Use species as a proxy for edibility  
- Test multi-class classification strategies

### 📏 Berry Size Estimation
- Predict average berry size using regression  
- Derive a formula usable in the wild (no computer required)  
- Explore feature importance and model interpretability

---

## 📦 Datasets

All datasets are located in the `/data` folder:

| Filename           | Description |
|--------------------|-------------|
| `clustering.json`  | Raw berry observations for unsupervised learning |
| `edible.json`      | Labeled data for edibility classification |
| `species.json`     | Expedition notes on five berry species |
| `berry.json`       | Berry-only features |
| `leaf.json`        | Leaf-only features |
| `petal.json`       | Petal-only features |
| `complete.json`    | Full feature set including soil and size |

---

## 📁 Repo Structure

```bash
📦 Unsupervised-Island/
├── data/
│   ├── clustering.json
│   ├── edible.json
│   ├── species.json
│   ├── berry.json
│   ├── leaf.json
│   ├── petal.json
│   └── complete.json
├── notebooks/
│   ├── 01_clustering.ipynb
│   ├── 02_edibility_prediction.ipynb
│   ├── 03_species_classification.ipynb
│   └── 04_berry_size_estimation.ipynb
├── survival_log.md
└── README.md
```

---

## 🧪 Tools & Libraries

- Python 3.8+  
- scikit-learn  
- pandas  
- matplotlib / seaborn  
- Optional: XGBoost, LightGBM, rule-based classifiers

---

## ✅ Final Deliverables

By the end of this challenge, you should produce:

- A clustering map of berry types  
- A rule-based classifier for edibility  
- A species classification model  
- A wilderness-ready formula for berry size

---

## 🚀 Getting Started

1. Clone the repo  
2. Explore the datasets in `/data`  
3. Start with `01_clustering.ipynb`  
4. Follow the storyline through each notebook  
5. Document your findings in `survival_log.md`

---

## 🧭 Tagline

**Survive the jungle. One prediction at a time.**

---
