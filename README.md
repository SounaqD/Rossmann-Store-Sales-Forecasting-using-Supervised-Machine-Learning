# 🏪 Rossmann Store Sales Forecasting

This project aims to **predict daily sales for Rossmann stores** across Europe using **supervised machine learning** techniques. The dataset contains store, sales, and promotional information — with the goal of building robust forecasting models to help store managers plan better.

---

## 📘 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Feature Engineering](#feature-engineering)
- [Modeling & Evaluation](#modeling--evaluation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Files](#files)
- [License](#license)

---

## 🧠 Overview

Rossmann operates over **3,000 drug stores across Europe**, where store managers need to **forecast daily sales up to six weeks in advance**.  
Sales depend on multiple factors such as:
- Store type and assortment level  
- Promotions and holidays  
- Competition and proximity  
- Seasonal trends (month, day, year, etc.)

This project uses a **machine learning pipeline** approach for preprocessing, feature engineering, model training, and evaluation.

---

## 📂 Dataset

- **Train data:** `train.csv`  
  Includes daily sales, customer count, and promotions for each store.
- **Test data:** `test.csv`  
  Similar structure but without the target variable (`Sales`).
- **Store data:** `store.csv`  
  Contains additional attributes such as store type, assortment, and competition information.

**Data fields:**
- `Store`, `Date`, `Sales`, `Customers`, `Open`, `Promo`,  
  `StateHoliday`, `SchoolHoliday`, `StoreType`, `Assortment`,  
  `CompetitionDistance`, `Promo2`, etc.

---

## 🔄 Project Workflow

### 1️⃣ Import Libraries  
Essential Python packages for data handling, visualization, and modeling (`pandas`, `numpy`, `sklearn`, etc.).

### 2️⃣ Data Loading & Merging  
Merges `train`, `test`, and `store` data to form unified datasets.

### 3️⃣ Preprocessing  
- Removes closed stores and zero-sales entries  
- Handles missing values (`CompetitionDistance`, `Promo2`)  
- Converts categorical columns and types

### 4️⃣ Feature Engineering  
Created rich temporal and store-level features:
- **Date-based features:** `year`, `month`, `day`, `dayofweek`, `weekofyear`  
- **Cyclic encoding:** `month_sin`, `month_cos`  
- **Lag features:** previous 7, 14, 28-day sales  
- **Rolling statistics:** mean and std (7-day window)  
- **Weekend indicator:** `is_weekend`

### 5️⃣ Train/Validation Split  
Time-based split ensuring 6 weeks of data reserved for validation.

### 6️⃣ Preprocessing Pipeline  
Using `ColumnTransformer` with:
- **Numerical:** Imputation + StandardScaler  
- **Categorical:** OneHotEncoder  

### 7️⃣ Modeling  
Models trained and evaluated:
- Linear Regression  
- Lasso LARS  
- Decision Tree Regressor  
- Random Forest Regressor  
- K-Nearest Neighbors  
- Support Vector Regressor (SVR)

Evaluation metrics include:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- R² and Adjusted R²  

---

## 📊 Exploratory Data Analysis (EDA)

Comprehensive visual insights using **Matplotlib** and **Seaborn**:

1. 📈 **Sales trend over time**  
2. 🗓️ **Sales by day of week & month**  
3. 🎯 **Impact of promotions**  
4. 🏪 **Store type comparison**  
5. 🔍 **Competition distance effects**  
6. 🌞 **Seasonality and trend decomposition**  
7. 🔥 **Feature importance (Random Forest)**  
8. ⚖️ **Residual analysis & model diagnostics**  
9. 🧮 **Correlation heatmap**  
10. 📆 **Sales heatmap by day and month**

---

## 🧾 Results

| Model | MAE | RMSE | MAPE (%) | R² | Adj_R² |
|:------|----:|----:|---------:|---:|-------:|
| Linear Regression | ... | ... | ... | ... | ... |
| Lasso LARS | ... | ... | ... | ... | ... |
| Decision Tree | ... | ... | ... | ... | ... |
| **Random Forest** | **Lowest error** | **Best fit** | **✔️** | **✔️** | **✔️** |
| KNN | ... | ... | ... | ... | ... |
| SVR | ... | ... | ... | ... | ... |

> 🏆 **Random Forest Regressor** gave the best overall performance on validation data.

The final model was retrained on the full training data and predictions were generated for the test set.

