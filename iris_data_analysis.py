# Task 1: Load and Explore the Dataset

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("\nFirst five rows of the dataset:")
print(df.head())

# Check data structure
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Group by 'species' and calculate the mean of features
print("\nMean of each feature grouped by species:")
print(df.groupby('species').mean())

# Task 3: Data Visualization

plt.figure(figsize=(10,5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['sepal width (cm)'], label='Sepal Width')
plt.title('Sepal Length and Width over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Measurement (cm)')
plt.legend()
plt.show()

# Bar Chart - Average petal length per species
plt.figure(figsize=(8,5))
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram - Distribution of petal length
plt.figure(figsize=(8,5))
plt.hist(df['petal length (cm)'], bins=20, color='purple', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot - Sepal length vs Petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

