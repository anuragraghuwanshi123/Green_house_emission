🌍Greenhouse Gas Emission Analysis & Prediction
This project analyzes and predicts greenhouse gas emissions from commodity and industry sources between 2010 and 2016. It combines multiple Excel sheets, cleans the data, visualizes emission trends, and trains a machine learning model to predict emissions using Random Forest.

Overview
Combines yearly commodity and industry data into a unified dataset
Cleans and standardizes column names
Performs exploratory data analysis (EDA)
Visualizes emissions over time for the top emitting entities
Trains a Random Forest model to predict emissions based on features like year, type, and source
Saves the trained model for future use
Dataset
Format: Excel sheets named YYYY_Detail_Commodity and YYYY_Detail_Industry for each year from 2010 to 2016
Key columns:
Name – Entity name (Commodity or Industry)
Substance – Emission substance
Supply Chain Emission Factors with Margins – Target variable for ML
Year, Source – Metadata
Various Data Quality (DQ) scores
All data is merged and processed into a single DataFrame for uniform analysis.

Technologies Used
Language: Python
Libraries:
pandas, numpy – Data manipulation
matplotlib, seaborn – Data visualization
scikit-learn – Machine learning
joblib – Saving the model
Environment: Jupyter Notebook (Anaconda recommended)
How to Run
Clone the repository:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Make sure the Excel file (green_house_gas_dataset.xlsx) is present in the project folder.

Open and run the notebook:

jupyter notebook Greenhouse_Gas_Analysis.ipynb
Execute the cells in order:

Data loading
Preprocessing
EDA and visualization
Model training and evaluation
Model saving
Output
Summary statistics and trend plots
Machine learning metrics: MSE and R² score
Saved model: emission_model.pkl
Program Information
Program Name: Greenhouse Gas Emission Analysis & Prediction
Initiative: Edunet Foundation – Skills4Future (AICTE)
