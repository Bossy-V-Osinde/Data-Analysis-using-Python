import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset (or replace with your own dataset)
df = pd.read_csv('iris.csv')  # Replace 'iris.csv' with your dataset

# Display the first few rows to inspect the data
print(df.head())

# Check the data types and missing values
print(df.info())
print(df.isnull().sum())

# Clean the data: Drop rows with missing values (or use fillna() if you prefer to fill them)
df_cleaned = df.dropna()

# Verify the cleaned dataset
print(df_cleaned.info())

# Basic statistics of numerical columns
print(df_cleaned.describe())

# Group by 'variety' and compute the mean of numerical columns for each group
grouped_data = df_cleaned.groupby('variety').mean()
print(grouped_data)

# 1. Line chart for trends (Sepal Length across variety)
plt.figure(figsize=(10, 6))
for variety in df_cleaned['variety'].unique():
    variety_data = df_cleaned[df_cleaned['variety'] == variety]
    plt.plot(variety_data['sepal.length'], label=variety)  # Correct column name 'sepal.length'
plt.title('Sepal Length Trend by variety')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length')
plt.legend(title='Variety')
plt.savefig('sepal_length_trend.png')  # Save the figure
plt.show()

# 2. Bar chart for average sepal length per variety (using matplotlib)
plt.figure(figsize=(10, 6))
avg_sepal_length = df_cleaned.groupby('variety')['sepal.length'].mean()
avg_sepal_length.plot(kind='bar', color='skyblue')
plt.title('Average Sepal Length by variety')
plt.xlabel('Variety')
plt.ylabel('Average Sepal Length')
plt.savefig('average_sepal_length.png')  # Save the figure
plt.show()

# 3. Histogram for Petal Length Distribution (using matplotlib)
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['petal.length'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.savefig('petal_length_distribution.png')  # Save the figure
plt.show()

# 4. Scatter plot for Sepal Length vs Petal Length (using matplotlib)
plt.figure(figsize=(10, 6))
for variety in df_cleaned['variety'].unique():
    variety_data = df_cleaned[df_cleaned['variety'] == variety]
    plt.scatter(variety_data['sepal.length'], variety_data['petal.length'], label=variety)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Variety')
plt.savefig('sepal_vs_petal_length.png')  # Save the figure
plt.show()

# Save the results to a text file
with open('sales_summary.txt', 'w') as f:
    f.write("Basic Statistics:\n")
    f.write(df_cleaned.describe().to_string())
    f.write("\n\nGrouped Data by variety:\n")
    f.write(grouped_data.to_string())
