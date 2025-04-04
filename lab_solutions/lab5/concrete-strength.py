import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"Dataset file '{file_path}' not found.")
        return None

# EDA: Preview and basic statistics
def exploratory_data_analysis(data):
    print("\nFirst 5 rows of the dataset:")
    print(data.head())

    print("\nSummary Statistics:")
    print(data.describe())

    print("\nMissing Values (filling missing values with median):")
    print(data.isnull().sum())
    
    # Fill missing values with the median of the column
    data = data.fillna(data.median())
    print("\nMissing values after filling with median:")
    print(data.isnull().sum())

    return data

# Plot histograms for each feature
def plot_histograms(data):
    data.hist(bins=25, figsize=(12, 8), edgecolor='darkorange')  # Change edge color to 'darkorange'
    plt.suptitle("Feature Distributions", fontsize=16, color='darkblue')  # Change title color to 'darkblue'
    plt.tight_layout()
    plt.show()

# Boxplot to visualize feature distributions and outliers
def plot_boxplots(data):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data, color='lightgreen')  # Change boxplot color to 'lightgreen'
    plt.title("Feature Distributions with Boxplots", fontsize=16, color='darkred')  # Change title color to 'darkred'
    plt.tight_layout()
    plt.show()

# Rename columns for simplicity
def simplify_column_names(data):
    data.columns = [name.split('(')[0].strip() for name in data.columns]
    return data

# Separate features and target variable, then scale the features using MinMaxScaler
def prepare_features_and_target(data):
    X = data.drop('Concrete compressive strength', axis=1)
    y = data['Concrete compressive strength']

    # Min-Max scaling the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Split the dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state=50):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Main workflow
file_path = '../../datasets/concrete_strength/Concrete_Data.xls'
data = load_data(file_path)

if data is not None:
    data = simplify_column_names(data)
    data = exploratory_data_analysis(data)
    plot_histograms(data)
    plot_boxplots(data)  # Added Boxplot visualization

    X_scaled, y = prepare_features_and_target(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    print("\nData preprocessing complete!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display evaluation metrics
print("\nModel Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Visualize Predicted vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')  # Change color to 'purple'
plt.xlabel("Actual Values", color='darkgreen')  # Change label color to 'darkgreen'
plt.ylabel("Predicted Values", color='darkgreen')  # Change label color to 'darkgreen'
plt.title("Predicted vs Actual Concrete Strength", color='navy')  # Change title color to 'navy'
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='darkorange', linestyle='--')  # Change line color to 'darkorange'
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color='teal')  # Change residual plot color to 'teal'
plt.xlabel("Residuals", color='brown')  # Change label color to 'brown'
plt.title("Residual Plot", color='maroon')  # Change title color to 'maroon'
plt.tight_layout()
plt.show()
