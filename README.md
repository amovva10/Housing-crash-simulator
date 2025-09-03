# Housing Crash Simulator

U.S. housing prices are highly sensitive to economic fluctuations. Events such as rising interest rates or spikes in unemployment can quickly destabilize the market. However, there is a lack of clear and accessible tools for exploring “what-if” housing crash scenarios. To address this gap, we developed a machine learning model that acts as a housing crash probability simulator, enabling users to test how different scenarios might affect the housing market.
 

## Data
The dataset used in this project was sourced from 
[Kaggle – Factors Influencing U.S. House Prices](https://www.kaggle.com/datasets/jyotsnagurjar/factors-influencing-us-house-prices).  

It contains U.S. housing market indicators from 2003–2022, including:
- Building Permits & Construction Costs  
- GDP & Household Income  
- Interest & Mortgage Rates  
- Unemployment & Delinquency Rates  
- Housing Subsidies & Urban Population  
- Home Price Index (Target Variable)


## Machine Learning Models

All variables — including the outcome variable, **Home Price Index (HPI)** — are continuous and measured over time.  

We evaluated two models:  

- **Decision Tree**  
- **Ridge Regression** (**chosen**)  

 Ridge Regression achieved the best performance, with the **lowest MAE** and **highest R²** (see notebooks for details).  
 We also engineered **time-based features** (lagged values of HPI) to capture market trends while avoiding data leakage.

---

## Features

- Forecast **Home Price Index (HPI)** using ML models  
- Simulate *what-if* scenarios:  
  - Interest rates ↑/↓  
  - Mortgage rates ↑/↓  
  - Unemployment ↑/↓  
- Estimate **crash probability** under stress scenarios  
- Explore results via an interactive **Streamlit app**  

---

## Tech Stack
- **Python**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, Tableau Public, Streamlit
- **Version control**: Git & GitHub  
- **ML models**: Linear Regression, Decision Tree, Ridge Regression
- [Presentation](https://docs.google.com/presentation/d/1t2Ebq2lJ2uK7tw7e7eI4wWVfkjm6UCHRCP2EcPVMbMU/edit?usp=sharing)

---
## Streamlit App (Screenshot)
<img width="790" height="1226" alt="image" src="https://github.com/user-attachments/assets/79b2b808-6d1f-40c0-97d5-6eb04888ead5" />


## Authors
- [@amovva10](https://github.com/amovva10)  
- [@Jjc55](https://github.com/Jjc55)  


## License
This project is licensed under the [MIT License](LICENSE).
