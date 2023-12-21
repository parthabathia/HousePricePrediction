# Housing Price Prediction

The provided Python code performs a housing price prediction task using the XGBoost regression model. Here is a description of the code:

    Importing Libraries:
        The necessary libraries are imported, including NumPy, Pandas, Matplotlib, Seaborn, StandardScaler from scikit-learn for data preprocessing, train_test_split for splitting the dataset, XGBRegressor for XGBoost regression, and metrics from scikit-learn for evaluating the model.

    Loading the Dataset:
        The code reads a housing dataset from a CSV file named 'housing.csv' using Pandas.

    Exploratory Data Analysis (EDA):
        The first few rows of the dataset are displayed using head() to provide a glimpse of the data.
        The code checks for missing values in the dataset using isnull().sum().
        Descriptive statistics of the dataset are displayed using describe().
        The correlation matrix of the dataset is calculated using the corr() method.

    Correlation Heatmap:
        A heatmap is generated using Seaborn and Matplotlib to visualize the correlation matrix. This helps in understanding the relationships between different features in the dataset.

    Data Preprocessing:
        The target variable 'MEDV' (housing prices) is separated from the features in the dataset, creating input features (X) and target variable (Y).

    Train-Test Split:
        The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

    XGBoost Regression Model:
        An XGBoost regression model is created using XGBRegressor and trained on the training data.

    Model Evaluation on Training Data:
        The model's predictions are generated on the training data, and the R-squared (score_r2) and Mean Absolute Error (score_mae) metrics are calculated and printed.

    Model Evaluation on Testing Data:
        The model's predictions are generated on the testing data, and similar evaluation metrics (R-squared and Mean Absolute Error) are calculated and printed.

    Visualization:

    A scatter plot is created to visualize the predicted prices against the actual prices for the training data.

Overall, the code demonstrates the process of loading, exploring, and preprocessing a housing dataset, training an XGBoost regression model, and evaluating its performance on both training and testing datasets. The visualization provides a qualitative assessment of the model's predictions.
