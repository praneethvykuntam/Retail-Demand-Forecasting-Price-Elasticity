# 🛒 Retail Demand Forecasting & Price Elasticity

This repository contains the code and pipeline for an end-to-end machine learning project focused on **retail demand forecasting** and **price elasticity modeling**.  
The project builds predictive models to forecast product demand and estimate the sensitivity of sales to price changes.  
It is structured for scalability, reproducibility, and integration into enterprise workflows.

---

## 🎯 Main Objective

The primary objective of this project is to forecast product demand at the SKU/store level while also quantifying **price elasticity**.  
This enables retailers to optimize inventory planning, promotions, and pricing strategies.  

- **Demand Forecasting**: Predict short-term and long-term demand to improve replenishment decisions.  
- **Price Elasticity Modeling**: Estimate how changes in price affect demand to support dynamic pricing.  

---

## ⚙️ Project Steps

### 1. Data Preprocessing
- **Data Collection & Cleaning**: Collected transactional sales, pricing, and product metadata. Cleaned the dataset by handling missing values, correcting outliers, and ensuring consistent formatting.  
- **Feature Engineering**:  
  - Lag features for demand (e.g., previous weeks’ sales).  
  - Rolling averages and moving-window statistics for seasonality.  
  - Calendar features such as holidays, weekdays, and month.  
  - Price-related transformations (discount %, relative price vs. category average).  
- **Data Splitting**: Created train, validation, and test sets based on time to mimic a real forecasting setting.  

---

### 2. Exploratory Data Analysis (EDA)
- Performed visualizations of sales trends, seasonality, and distribution across product categories.  
- Analyzed correlation between price, demand, and promotions.  
- Identified clusters of products with similar sales dynamics.  

📊 **Example Visualizations**

**Sales Trend for a Product**  
<img width="1262" height="586" alt="sales_trend" src="https://github.com/user-attachments/assets/3cae38f1-5005-452a-beca-1059ceb70b35" />


**Total Demand Over Time (All Products)**  
<img width="1484" height="579" alt="total_series" src="https://github.com/user-attachments/assets/bb560416-867c-464b-b75a-1aa24e837523" />


---

### 3. Model Development
- Implemented multiple forecasting approaches:  
  - **Baseline models**: Naïve forecast, Moving Average.  
  - **Machine Learning models**: LightGBM, Random Forest Regressor.  
  - **Advanced regressors**: Gradient Boosting with hyperparameter tuning.  
- **Evaluation Metrics**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).  

📊 **Example Outputs**

**Actual vs Predicted Demand**  
<img width="796" height="817" alt="actual_vs_pred" src="https://github.com/user-attachments/assets/0f9f37b9-4165-4e61-92c3-4480f10ef0f2" />


**Error Distribution**  
<img width="1184" height="581" alt="error_hist" src="https://github.com/user-attachments/assets/4bcfca40-baf8-4826-b1dc-2dcf8fa50f0a" />


---

### 4. Price Elasticity Estimation
- Built regression models to estimate demand sensitivity to price changes.  
- Measured elasticity at both **product-level** and **category-level**.  
- Elasticity values provide actionable insight into:  
  - Which products are price-sensitive (elastic).  
  - Which products sustain demand even under price changes (inelastic).  

📊 **Price Elasticity Example**  
<img width="803" height="729" alt="elasticity_scatter" src="https://github.com/user-attachments/assets/78d6c293-9174-43ac-9608-89a6967a0efe" />


---

### 5. Model Deployment (Optional Extension)
- Pipelines designed for modularity and reproducibility.  
- Dependencies managed via `requirements.txt` and virtual environments.  
- Can be extended to run as a scheduled batch job or integrated with APIs for real-time demand forecasting.  

---

## 💡 Business Value and Applicability

This project provides tangible value for retailers:

- **Inventory Optimization**: Accurate forecasts help reduce stockouts and overstocks.  
- **Dynamic Pricing**: Elasticity estimates allow retailers to optimize promotions and price adjustments.  
- **Revenue Growth**: Smarter pricing and better inventory planning improve profitability.  
- **Decision Support**: Managers gain data-driven insights into demand drivers and consumer behavior.  

---

## 📂 Dataset Source

The dataset used is synthetic/representative of real-world retail transactional and pricing data.  
It consists of:  
- Historical sales transactions.  
- Product catalog and metadata.  
- Pricing and promotion history.  
- Calendar/holiday information.  
 

---

