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
| 7  | **Learning Curve Experiments (1% → 100%)**                              | Analyzes how performance scales with more labeled data and how different loss functions affect performance, conducting detailed Error Analysis. |
| 8  | **Model Distillation & Quantization** *(coming soon)*                  | Converts the best model into a smaller, faster one suitable for real-time or low-power use.   |
| 9  | **Performance Dashboards & Confusion Matrices**                         | Highlights strengths and weaknesses of each modeling strategy.                                |
| 10 | **Reproducible Notebooks & Dependency Pinning**                         | Easy-to-follow seeded structure for re-running experiments end-to-end.                              |

## 📚 Dataset and Classification Objective

We use the [Jigsaw Toxic Comment Classification dataset](https://huggingface.co/datasets/Arsive/toxicity_classification_jigsaw), which contains **25,960 user comments** from Wikipedia talk pages, annotated for multiple types of toxicity.

Each comment is labeled across **five toxicity subcategories**:
- `toxic`
- `severe_toxic`
- `obscene`
- `insult`
- `identity_hate`

This is a **multi-label classification problem** — a single comment may exhibit multiple types of toxicity simultaneously. For example, a comment can be both `toxic` and `insulting`.

![Label Distribution](figures/learning_curve.png)

### ⚖️ Label Distribution Challenges

While the overall binary label for `toxic` is **almost perfectly balanced** (~47% toxic, ~53% non-toxic), the five finer-grained subcategories are **extremely imbalanced**:

- `obscene` and `insult` are moderately frequent.
- `severe_toxic`, `identity_hate` are very rare (under 5% appearance).

This imbalance means that standard loss functions (like unweighted binary cross-entropy) would cause the model to ignore minority classes. To address this, we adopt **Weighted Binary Cross-Entropy Loss**, which gives higher importance to underrepresented labels and improves the model’s ability to detect subtle or rare toxicity patterns.

### 🏁 Objective

Build a robust, scalable **multi-label classifier** that can accurately detect all five toxicity categories — even with very limited labeled data — while comparing different modeling strategies (rule-based, transformer-based, LLM-based, and distilled models).


## 🗂️ Project Structure

The project is organized into four main parts, each building progressively toward a robust and deployable toxicity classification system.

### Part 1 — Setting Up the Problem

- **Task Definition and State of the Art**  
  Presents the motivation behind toxicity detection, outlines real-world applications, and surveys the current research landscape, including benchmark models and methodologies.

- **Dataset Exploration**  
  Offers a detailed overview of the dataset, including size, label distribution, text statistics, and challenges such as label imbalance and multi-label structure.

- **Random Classifier Benchmark**  
  Establishes a theoretical and empirical lower-bound benchmark using a random classifier to contextualize all future performance metrics.

- **Rule-Based Baseline Implementation**  
  Implements a handcrafted classifier based on lexical patterns and textual cues. Evaluates its precision-recall trade-off and limitations in detecting nuanced or subtle toxicity.

---

### Part 2 — Learning with Limited Labeled Data

- **Few-Shot BERT Modeling**  
  Trains a DistilBERT model on just 32 labeled examples, leveraging architectural tweaks like high dropout and class balancing to prevent overfitting.

- **Data Augmentation without LLMs**  
  Experiments with strategies such as back-translation and zero-shot weak labeling using traditional tools to artificially expand the dataset.

- **Zero-Shot Learning with LLMs**  
  Uses a general-purpose language model (Gemini 2.0 Flash) to classify toxic content directly via prompt-based inference, bypassing the need for labeled training data.

- **Synthetic Data Generation with LLMs**  
  Generates additional labeled examples using an LLM, then retrains BERT on the expanded dataset. Assesses the impact of synthetic data on model quality.

- **Technique Selection and Application**  
  Compares all limited-data strategies to identify the most effective approach. Applies it at scale and discusses how and why it works best.

---

### Part 3 — Full Dataset Training and Comparison

- **Progressive Fine-Tuning**  
  Trains DistilBERT models on increasing fractions of the dataset (1%, 10%, 25%, 50%, 75%, 100%) to study the relationship between dataset size and performance.

- **Learning Curve Analysis**  
  Visualizes how performance improves as more labeled data becomes available, identifying diminishing returns and critical inflection points.

- **Technique Integration**  
  Incorporates the best-performing limited-data augmentation techniques into the full-data training regime to assess scalability and synergy.

- **Comparative Evaluation**  
  Performs an in-depth comparison across all methods used throughout the project, highlighting trade-offs in accuracy, precision, recall, generalization, and practical feasibility.

---

### Part 4 — Model Distillation and Deployment Optimization

- **Model Distillation / Quantization** *(To be completed)*  
  Converts the best-performing large model into a smaller, faster student model using knowledge distillation or quantization techniques.

- **Speed vs. Accuracy Trade-Off**  
  Benchmarks inference time and accuracy of the compressed model relative to the original, providing deployment-ready performance metrics.

- **Failure Analysis and Future Directions**  
  Identifies shortcomings in the student model and proposes actionable improvements for both training and model design in constrained environments.



## 📈 Performance Summary

This section summarizes the performance of all key models and strategies evaluated during the project, using macro-averaged metrics where applicable. The focus is on the balance between precision and recall, robustness under low-resource settings, and final-state performance after full fine-tuning.

###  Best-Performing Model

The highest overall performance was achieved by a fully fine-tuned **DistilBERT model** trained on 100% of the dataset using **weighted binary cross-entropy loss** to address class imbalance.

| Metric     | Score (%) |
|------------|-----------|
| Accuracy   | **98.29** |
| Precision  | **97.31** |
| Recall     | **97.55** |
| F1 Score   | **97.43** |

---

| Method                              | Accuracy | Precision | Recall | F1 Score |
|-------------------------------------|----------|-----------|--------|----------|
| **Random Classifier**               | 50.00    | 47.10     | 50.00  | 48.50    |
| **Rule-Based Classifier**           | 60.30    | 84.00     | 19.00  | 31.00    |
| **DistilBERT (32 Examples)**        | 78.95    | 39.48     | 50.00  | 44.12    |
| **+ Back-Translation**              | 77.06    | 40.67     | 48.89  | 43.76    |
| **+ Zero-Shot Weak Labels (BERT)**  | 68.57    | 51.86     | 51.80  | 51.82    |
| **LLM Zero-Shot (Gemini 2.0 Flash)**| 82.83    | 74.95     | 85.37  | 77.46    |
| **+ LLM-Generated Synthetic Data**  | 54.77    | 52.05     | 53.06  | 48.98    |
| **DistilBERT (100% Data)** 🏆       | 98.29    | 97.31     | 97.55  | 97.43    |

---
