Comprehensive toxicity classification analysis using BERT, rule-based baselines, and data augmentation techniques. Includes limited data learning, model distillation, and performance comparison across different approaches on the Jigsaw toxicity dataset.

## 🚀 Project Overview

**`nlp-toxicity-classification-analysis`** is an end-to-end study of automated toxicity detection on the Jigsaw Toxicity dataset. We explore the entire spectrum of approaches—from coin-flip baselines and handcrafted rule sets, through few-shot fine-tuning of BERT/DistilBERT with data-augmentation techniques, all the way to zero-shot large-language-model (LLM) inference and full-dataset training.  Additionally, we distil our best performing model into a lightweight version, documenting all the process. 

The goal is two-fold:

- **Business impact** – Provide a reproducible pipeline that helps content-moderation teams flag abusive comments quickly and reliably, keeping online communities healthy.
- **Engineering trade-offs** – Benchmark heavyweight, high-accuracy models against lightweight, distilled alternatives to inform real-world deployment where latency, memory, or cost constraints matter.

---

## ✨ Key Features

| #  | Feature                                                                 | Description                                                                                   |
|----|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 1  | **Exploratory Data Analysis (EDA)**                                     | Visualizations of class balance, word clouds, and multi-label overlap patterns.               |
| 2  | **Random & Rule-Based Baselines**                                       | Establish lower bounds and highlight limitations of simple heuristics for toxicity detection. |
| 3  | **Few-Shot DistilBERT (32 examples)**                                   | Demonstrates what can be learned from extremely limited labeled data.                         |
| 4  | **Data Augmentation Techniques**                                        | Includes back-translation, zero-shot labeling, and LLM-based synthetic comment generation.    |
| 5  | **LLM Zero-Shot Classification (Gemini 2.0 Flash)**                     | Uses powerful pre-trained models without any fine-tuning to classify toxic content.           |
| 6  | **Full-Dataset Fine-Tuning (Best Model)**                               | Achieves 98.29% accuracy and 97.43% F1 score using DistilBERT on the entire dataset.          |
| 7  | **Learning Curve Experiments (1% → 100%)**                              | Analyzes how performance scales with more labeled data.                                       |
| 8  | **Model Distillation & Quantization** *(coming soon)*                  | Converts the best model into a smaller, faster one suitable for real-time or low-power use.   |
| 9  | **Performance Dashboards & Confusion Matrices**                         | Highlights strengths and weaknesses of each modeling strategy.                                |
| 10 | **Reproducible Notebooks & Dependency Pinning**                         | Easy-to-follow structure for re-running experiments end-to-end.                              |
