🌍 Greenhouse Gas Emission Analysis & Prediction
A machine learning-based project that analyzes and predicts greenhouse gas emissions from commodity and industry sources across the years 2010–2016. This project performs data preprocessing, visualization, and trains a predictive model using Random Forest to estimate emissions based on various attributes.

📌 Overview
This project includes:

📊 Combining yearly Excel sheets (YYYY_Detail_Commodity and YYYY_Detail_Industry) into a unified dataset.

🧹 Cleaning and standardizing column names for consistency.

🔍 Exploratory Data Analysis (EDA) to uncover trends and patterns.

📈 Visualizations to track emission changes over time for top contributors.

🤖 Training a Random Forest model to predict emissions using features like year, entity type, and emission source.

💾 Saving the trained model for future predictions using joblib.

📁 Dataset Format
File: green_house_gas_dataset.xlsx

Sheets: YYYY_Detail_Commodity and YYYY_Detail_Industry (where YYYY = 2010 to 2016)

Key Columns:

Name – Name of the commodity or industry

Substance – Emission substance type

Supply Chain Emission Factors with Margins – Target variable for prediction

Year, Source – Metadata for filtering and modeling

Various Data Quality (DQ) score columns

All sheets are merged into a single DataFrame for streamlined processing.

🧰 Technologies Used
Language: Python

Libraries:

pandas, numpy – Data manipulation

matplotlib, seaborn – Data visualization

scikit-learn – Machine learning

joblib – Model serialization

Environment: Jupyter Notebook (recommended via Anaconda)

🚀 How to Run
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Ensure the Dataset is Available:
Place green_house_gas_dataset.xlsx in the project directory.

Open and Run the Notebook:

bash
Copy
Edit
jupyter notebook Greenhouse_Gas_Analysis.ipynb
Execute Notebook Cells in Order:

Data Loading

Data Preprocessing

EDA and Visualization

Model Training and Evaluation

Model Saving

✅ Output
📈 Trend plots for top emitting commodities and industries

📋 Summary statistics for all years

📊 Model Evaluation:

MSE (Mean Squared Error)

R² Score

💾 Trained model saved as: emission_model.pkl

💡 Program Information
Program Name: Greenhouse Gas Emission Analysis & Prediction

Initiative: Edunet Foundation – Skills4Future (AICTE)

📫 Contact
For any queries or contributions, feel free to open an issue or pull request.
