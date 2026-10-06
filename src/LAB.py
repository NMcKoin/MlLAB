import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
import os


if __name__ == "__main__":
    data_dir = './data'
    out_dir = './output'
    file_name = 'rpgg_db_1903.csv'
    file_path = os.path.join(data_dir, file_name)

    df = pd.read_csv(file_path)
    df.info()
    print(df.isnull().sum())
    print(df.shape)

    #drop rows with missing values
    df2 = df[["num votes", "year", "avg rating"]].dropna()
    input_features = df2[["num votes", "year"]]
    target_feature = df2["avg rating"]

    #separate input and target features, drop rows with missing values.
    input_features = input_features.dropna()
    target_feature = target_feature.loc[input_features.index]
    print(input_features.shape)
    print(target_feature.shape)

    # Create a dummy regressor model and fit it to the data
    model = DummyRegressor(strategy="mean")
    model.fit(input_features, target_feature)

    
    predictions = model.predict([[496, 2004]])
    print("Predictions: ", predictions)
    score = model.score(input_features, target_feature)
    print("Model Score: ", score)

    
    sns.pairplot(df)
    plot_name = 'pairplot.png'
    path = os.path.join(out_dir, plot_name)
    plt.savefig(path)


    df.hist(figsize=(10, 6))
    plot_name = 'histogram.png'
    path = os.path.join(out_dir, plot_name)
    plt.savefig(path)

    
    correlation_matrix = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(8, 6))
    snsheatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plot_name = 'correlation_matrix.png'
    path = os.path.join(out_dir, plot_name)
    plt.savefig(path)


    df["name_length"] = df["names"].str.len()

    corr2 = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(8, 6))
    snsheatmap2 = sns.heatmap(corr2, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix with Name Length')
    plot_name = 'correlation_matrix_with_name_length.png'
    path = os.path.join(out_dir, plot_name)
    plt.savefig(path)