# Housing Crash Simulator 

U.S. housing prices are highly sensitive to economic fluctuations. Events such as rising interest rates or spikes in unemployment can quickly destabilize the market. However, there is a lack of clear and accessible tools for exploring “what-if” housing crash scenarios. To address this gap, we developed a machine learning model that acts as a housing crash probability simulator, enabling users to test how different scenarios might affect the housing market.

Given that all variables—including the outcome variable, Home Price Index—are continuous and measured over time, Linear Regression and Random Forest models were identified as the most suitable machine learning approaches for this dataset. Of the two, the Random Forest model outperformed Linear Regression, achieving a lower MAE (#) and a higher R^2 value (#).
 
---

## Features
Planned functionality:
- This app will forecast **Home Price Index (HPI)** with ML models  
- It will simulate **“what-if” scenarios** (e.g., interest rates +2%, unemployment spikes, inventory changes)  
- It will estimate **crash probability** under stress scenarios  
- It will include an interactive **Streamlit app** for exploration  

---

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

---

## Tech Stack
- **Python**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, Tableau Public, Streamlit
- **Version control**: Git & GitHub  
- **ML models**: Linear Regression, Random Forest
- [Presentation](https://docs.google.com/presentation/d/1t2Ebq2lJ2uK7tw7e7eI4wWVfkjm6UCHRCP2EcPVMbMU/edit?usp=sharing)

---

## Authors
- [@amovva10](https://github.com/amovva10)  
- [@Jjc55](https://github.com/Jjc55)  

---

## License
This project is licensed under the [MIT License](LICENSE).
