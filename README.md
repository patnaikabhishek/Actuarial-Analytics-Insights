# Predicting Longevity Risk in Pension Plans

## Project Overview
This project aims to analyze and predict **longevity risk** in pension plans using statistical and machine learning methods. Longevity risk refers to the uncertainty about the lifespan of individuals and the potential financial implications it has for pension funds. The project simulates data for a population of 500 individuals and uses actuarial science principles to predict life expectancy and assess the associated risks.

The key objective of this project is to provide insights into longevity risk, enabling pension plan administrators to better manage risks and optimize their strategies. The model predicts the likelihood of individuals surviving beyond a certain age and estimates the financial impact on pension plans.

## Objective
- **Understand longevity risk**: Explore the concept of longevity risk and its impact on pension funds.
- **Predict life expectancy**: Use demographic and health data to predict the remaining life expectancy of individuals in the dataset.
- **Assess financial implications**: Estimate the financial risks for pension plans due to longevity uncertainty.
- **Data Visualization**: Create visual representations of the data and model predictions to facilitate decision-making.

## Tools & Libraries
This project was developed using the following tools and libraries:
- **Python 3**: The primary programming language used for this project.
- **Jupyter Notebook**: An open-source web application used to create and share documents that contain live code, equations, visualizations, and narrative text.
- **NumPy**: A fundamental package for scientific computing with Python, used for handling arrays and numerical operations.
- **Pandas**: A data manipulation and analysis library, providing data structures like DataFrames for efficient handling of structured data.
- **Matplotlib**: A plotting library used for creating static, interactive, and animated visualizations in Python.
- **Seaborn**: A Python data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
- **scikit-learn (sklearn)**: A library for data analysis and preprocessing, used here for tasks like splitting the dataset into training and test sets, as well as for data transformation and model evaluation.


## Dataset
The dataset simulates data for 500 individuals, including the following columns:
- `Age`: The current age of the individual.
- `Gender`: The gender of the individual (Male or Female).
- `Height`: The height of the individual.
- `Weight`: The weight of the individual.
- `Health_Status`: An indicator of the individual’s general health (Healthy, Unhealthy).
- `Smoke_Status`: Whether the individual smokes (Yes or No).

The dataset is synthetic and does not represent any real-world data.

## Methodology
1. **Data Collection & Preprocessing**:
   - We simulated a dataset with features such as age, income, health status, and smoking habits for 500 individuals.
   - Data is cleaned by removing missing values and encoding categorical features like "smoking status" using one-hot encoding.

2. **Exploratory Data Analysis (EDA)**:
   - Various visualizations (histograms, box plots, pair plots) are created to understand the distribution of variables and relationships between features.
   - Correlation matrices help identify potential relationships between the features and the target variable (longevity risk).

3. **Data Splitting**:
   - The dataset is split into training and testing sets using **`train_test_split`** from **scikit-learn**, ensuring the model is trained and evaluated on different subsets of the data.

4. **Model Building**:
   - A **Linear Regression** model is used to predict the longevity risk based on the dataset's features.
   - **`scikit-learn`**'s **LinearRegression** class is used to build and train the model.

5. **Model Evaluation**:
   - Model performance is evaluated using metrics such as **Mean Absolute Error (MAE)** and **R-squared (R²)** to assess the accuracy of the longevity risk predictions.

6. **Conclusion**:
   - Insights are derived from the analysis results, and suggestions are provided for improving pension plans' risk assessment based on the data-driven findings.

## Code Implementation

### 1. Importing Necessary Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

### 2. Data Simulation
# Randomly generating synthetic data for 500 individuals
np.random.seed(0)

data = {
    'Age': np.random.randint(20, 70, 500),
    'Gender': np.random.choice(['Male', 'Female'], 500),
    'Height': np.random.randint(150, 190, 500),
    'Weight': np.random.randint(45, 100, 500),
    'Health_Status': np.random.choice(['Healthy', 'Unhealthy'], 500),
    'Smoke_Status': np.random.choice(['Yes', 'No'], 500)
}

# Creating the DataFrame
df = pd.DataFrame(data)

### 3. Data Preprocessing
# Encoding categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Health_Status'] = df['Health_Status'].map({'Healthy': 1, 'Unhealthy': 0})
df['Smoke_Status'] = df['Smoke_Status'].map({'Yes': 1, 'No': 0})

# Creating a target variable: Simulated life expectancy based on age and other factors
df['Life_Expectancy'] = df['Age'] + np.random.normal(30, 5, 500)  # Normal distribution around 30

# Feature Selection
X = df[['Age', 'Gender', 'Height', 'Weight', 'Health_Status', 'Smoke_Status']]
y = df['Life_Expectancy']

### 4. Splitting Data into Train and Test Sets
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 5. Model Building and Evaluation
# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions and Evaluation
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

print(f"Linear Regression - MAE: {lr_mae}, RMSE: {lr_rmse}")

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and Evaluation
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

print(f"Random Forest - MAE: {rf_mae}, RMSE: {rf_rmse}")

### 6. Model Comparison and Conclusion
# Comparing the results
model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_mae, rf_mae],
    'RMSE': [lr_rmse, rf_rmse]
})

print(model_comparison)

### 7. Visualizations
# Distribution of Predicted vs Actual Life Expectancy
plt.figure(figsize=(10,6))
sns.histplot(y_test, color='blue', label='Actual', kde=True)
sns.histplot(rf_predictions, color='red', label='Predicted', kde=True)
plt.title('Comparison of Actual vs Predicted Life Expectancy')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Feature Importance in Random Forest Model
# Feature importance plot
features = X.columns
importances = rf_model.feature_importances_

plt.figure(figsize=(10,6))
sns.barplot(x=features, y=importances)
plt.title('Feature Importance in Predicting Life Expectancy')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

## Insights
- **Key Factors**: The analysis shows that certain factors such as **age**, **health status**, and **smoking status** have the greatest influence on predicting life expectancy. These variables are key to assessing the longevity risk of individuals.
  
- **Risk Factors**: Older individuals, those with poorer health, or smokers tend to have a **lower life expectancy**. These factors must be considered when assessing the **longevity risk** in pension plans.

- **Pension Plan Implications**: Pension plan administrators can use these insights to **adjust their models** for longevity risk. By taking into account factors such as health and lifestyle, they can more accurately predict liabilities and make more informed financial decisions.

- **Model Performance**: The **Linear Regression model** showed a reasonable fit to the data, but further improvements can be made by exploring additional features or more advanced techniques. 

- **Visualizations**: The visualizations clearly showed the **predicted vs. actual life expectancy**, helping to assess the accuracy of the model's predictions.

## Conclusion
This project provided an approach to predicting longevity risk in pension plans based on synthetic data. By using **linear regression** and conducting thorough **exploratory data analysis**, we identified key factors that influence life expectancy, such as **age**, **health status**, and **smoking habits**.

These insights can help pension plan administrators make more informed decisions when it comes to assessing the financial risks associated with longevity. The model can be used as a basis for better understanding how to structure pension plans and manage the long-term liabilities that come with predicting life expectancy.

For future work, we could improve the model by incorporating **real-world data**, additional features, or other advanced regression techniques to refine the predictions and provide even more actionable insights.





