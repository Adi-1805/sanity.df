# Plan: AI ML-Ready Preprocessing Environment

This document outlines the architecture for a real-world, Agentic Data Science environment built on the OpenEnv spec.

## 1. Task 1 (Easy): Mixed Values & Intelligent Imputation

**Dataset Plan & Columns**
*   **`product_code` (Mixed Values):** Contains values like `"A25"`, `"B-100"`, `"C 50"`. 
    *   *Edge Cases:* Inconsistent delimiters, missing numeric/alphabetical parts.
    *   *Agent Goal:* Split into `product_category` (string) and `product_id` (numeric).
*   **`color` (Categorical Missing):** Contains string categories with `NaN`s.
    *   *Agent Goal:* Mode imputation.
*   **`weight_kg` (Normal Distribution Missing):** Symmetrical, normally distributed continuous data with `NaN`s.
    *   *Agent Goal:* Mean imputation.
*   **`income_usd` (Skewed Missing):** Highly skewed data with massive outliers and `NaN`s.
    *   *Agent Goal:* Median imputation (penalized if Mean is used).
*   **`house_price` & `sqft` (KNN Imputer target):** `house_price` has missing values but is perfectly linearly correlated with `sqft`. 
    *   *Agent Goal:* KNN Imputer or regression imputation.

**Grading Plan**
*   **Method:** MSE / Accuracy comparison against a hidden `Gold` dataset.
*   **Scoring Logic:** +20% for each correctly processed column.
*   **Negative Grading:** -0.1 penalty for dropping any row containing a `NaN` instead of imputing.

## 2. Task 2 (Medium): Advanced Scaling & Transformations

**Dataset Plan & Columns**
*   **`age` (Uniform/Bounded):** Values strictly between 18 and 80.
    *   *Agent Goal:* **MinMax Scaler** (brings values to exactly 0.0 - 1.0).
*   **`sensor_reading` (Normally Distributed):** A perfect bell curve.
    *   *Agent Goal:* **Standardization / Z-Score** (brings mean to 0, std to 1).
*   **`stock_volume` (Significant Outliers):** Mostly normal, but 5% extreme outliers.
    *   *Agent Goal:* **Robust Scaler** (uses median and IQR, ignoring outliers).
*   **`sparse_audio_signal` (Sparse):** Lots of zeros, some peaks.
    *   *Agent Goal:* **MaxAbs Scaler** (preserves sparsity).
*   **`engagement_time` (Right-Skewed):** Exponential decay distribution.
    *   *Agent Goal:* **Log Transformation** (or Box-Cox) before any scaling.

**Grading Plan**
*   **Method:** Statistical distribution checks on the final columns.
*   **Scoring Logic:** +20% for each column matching the expected statistical properties (e.g., MinMax scaled column must have exactly min=0, max=1).

## 3. Task 3 (Hard): Domain-Driven Feature Construction

**Dataset Plan & Columns**
*   *Domain:* E-Commerce / Customer Lifetime Value
*   *Columns Provided:* `user_birthdate`, `account_created_date`, `last_purchase_date`, `total_orders`, `total_spent`, `is_premium`.

**Target Constructed Features:**
1.  **`age_at_signup`:** `account_created_date` - `user_birthdate` (in years).
2.  **`days_since_last_purchase` (Recency):** Current date - `last_purchase_date`.
3.  **`average_order_value` (AOV):** `total_spent` / `total_orders`.
4.  **`customer_lifetime`:** `last_purchase_date` - `account_created_date`.

**Grading Plan**
*   **Method:** Target Column Validation. The grader checks the agent's final dataframe for columns that mathematically match the expected output.
*   **Scoring Logic:** +25% for each semantically correct derived feature.
*   **Negative Grading:** -0.1 for leaving infinite (`inf`) values (e.g., division by zero) or negative lifetimes.

## Strict Normalization (0.01 to 0.99)
For all tasks, the raw score (0.0 to 1.0) is passed through:
`final_score = 0.01 + (raw_score * 0.98)`
This guarantees the score is strictly between 0.0 and 1.0, fulfilling the hackathon rules.
