# Employee Churn Prediction App

An end-to-end Machine Learning pipeline and interactive web deployment framework designed to predict workforce attrition. This project transitions an experimental data science workflow into a robust, production-grade software artifact using `scikit-learn` pipelines and a dynamic `Streamlit` user interface.

---

## 📌 Project Overview
Employee turnover impacts organizational productivity, corporate culture, and overhead costs. This project builds a reliable machine vision and predictive classification engine to determine whether an employee is likely to leave or stay with the company (`1` for Expected to Leave, `0` for Expected to Stay).

The architecture transitions from manual data preparation steps (scaling, manual one-hot encoding) to a **fully integrated enterprise pipeline** that automates feature isolation, missing value imputation, variance standard scaling, and categorical encoding natively within a single serialized object.

---

## ⚙️ Pipeline Architecture & Design Pattern
Following robust engineering design choices, this project decouples data streaming and schema definitions from inference logic:

* **Separation of Concerns:** Data loading, data splitting, and validation checkpoints are kept strictly outside the pipeline configuration to completely prevent data leakage.
* **Feature Schema Dictionary (`my_feature_dict.pkl`):** Categorical variables (`Education`, `City`, `Gender`, `EverBenched`) and Numerical features (`Age`, `JoiningYear`, `ExperienceInCurrentDomain`, `PaymentTier`) are mapped dynamically. This layout allows for seamless structural changes or metadata expansions.
* **Multi-Stage Estimator (`pipeline_class.pkl`):** Preprocessing transformations are bundled directly with a state-of-the-art `RandomForestClassifier` into an isolated, multi-stage scikit-learn `Pipeline` object. The pipeline handles raw input features and processes them identically during training, test validation, and web app scoring.
* **Advanced Cross-Version Serializing:** Handled via `dill` to capture deep dependencies, internal states, and functional configurations smoothly.

---

## 🚀 Web Application UI
The serialized production pipeline is deployed through an interactive dashboard built using **Streamlit**. 

* **Dynamic Dropdowns:** Inputs are automatically populated by parsing the unique metadata profiles extracted from the exported training schema dictionary.
* **Production-Grade Hot-Fixing:** Features explicit, integrated runtime adjustments that reconcile underlying package dependency version mismatches (such as NumPy NEP-50 cast restrictions or newer Scikit-Learn `ColumnTransformer` internal shifts) natively at the script level.
* **Immediate Prediction Profiling:** Outputs instant classification statuses with contextual alerts, clearly marking potential attrition flags (`1`) or high-retention signals (`0`).

---

## 📂 Project Directory Structure

```text
├── Employee_Churn_Assignment_DanishulKhairi.ipynb  # Model engineering, validation, and pipeline export
├── mymain.py                                         # Streamlit application entry point (UI & Inference)
├── pipeline_class.pkl                              # Serialized multi-stage ML Pipeline (dill format)
├── class_feature_dict.pkl                          # Serialized column schema mappings (pickle format)
└── README.md                                       # Documentation