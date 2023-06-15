import os
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Union
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def df_information(df:pd.DataFrame) -> None:
    """This function print the information of the DataFrame

    Args:
        df (pd.DataFrame): DataFrame
    """
    print(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"The dataframe has {df.select_dtypes('object').shape[1]} columns with objects type.")
    print(f"The dataframe has {df.select_dtypes('number').shape[1]} columns with number type.")
    print(f"The dataframe has {df.select_dtypes('bool').shape[1]} columns with bool type.")

def get_number_of_categories(df:pd.DataFrame) -> pd.Series:
    """get number of number of unique categories of every categories feature in dataframe.

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.Series: a series with the categorical features and number of categories.
    """
    return df.select_dtypes('object').nunique(dropna= False).sort_values(ascending= False)

def corr_filter(df:pd.DataFrame, correlation_threshold:float= 0.5) -> pd.Series:
    """filter a pair of features that have higher correlation than threshold within dataframe.

    Args:
        df (pd.DataFrame): DataFrame
        correlation_threshold (float, optional): The threshold values for correlation range. Defaults to 0.5.

    Returns:
        pd.Series: a series of features pair that have higher correlation than threshold within dataframe.
    """
    corr_matrix = df.corr()
    return corr_matrix[((corr_matrix >= correlation_threshold) | (corr_matrix <= -correlation_threshold)) & (corr_matrix != 1)].unstack().sort_values().drop_duplicates().dropna()

def get_missing_value(df:pd.DataFrame, percentage:bool= True, extented:bool= False) -> pd.DataFrame:
    """this function used to identify missing data

    Args:
        df (pd.DataFrame): DataFrame
        percentage (bool, optional): include percentage of missing value or not. Defaults to True.
        extented (bool, optional): include non-missing values or not. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with data_type, number of missing values, percentage of missing values (optional).
    """
    if percentage:
        content = [df.dtypes, df.isnull().sum(), round(df.isnull().mean() * 100, 2)]
        columns_name = {0: 'data_type', 1: 'number_missing', 2: 'percent_missing'}
    else:
        content = [df.dtypes, df.isnull().sum()]
        columns_name = {0: 'data_type', 1: 'number_missing'}
    
    missing_value_df = pd.concat(content, axis=1).rename(columns = columns_name).sort_values(by=['number_missing'], ascending = False)
    
    return missing_value_df if extented else missing_value_df[missing_value_df.number_missing > 0]

def plot_box_dist(feature:str, data:pd.DataFrame, figsize:tuple= (10,10))-> None:
    """plot stacked boxplot and histogram for feature on dataframe

    Args:
        feature (str): Target feature
        data (pd.DataFrame): DataFrame
        figsize (tuple, optional): size of the figure. Defaults to (10,10).
    """
    fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios':[2, 8]}, figsize= figsize)
    sns.boxplot(ax=axes[0], x= feature, data= data)
    sns.distplot(data[feature],ax=axes[1])
    fig.suptitle(f'{feature} Distribution (box-dist)')
    plt.show()

def plot_corr(df:pd.DataFrame, mask:bool= True, figsize:tuple= (10,10))->None:
    """plot correlation matrix of dataframe

    Args:
        df (pd.DataFrame): DataFrame
        mask (bool, optional): Apply masking to upper half or not. Defaults to True.
        figsize (tuple, optional): size of figure. Defaults to (10,10).
    """
    plt.figure(figsize= figsize)
    if mask:
        sns.heatmap(df.corr(), annot= True, mask = np.triu(df.corr()))
    else:
        sns.heatmap(df.corr(), annot= True)
    plt.plot()

def optimal_number_clusters(data_scaled:Union[pd.DataFrame, np.ndarray], n_clusters:int= 10) -> None:
    """Calculates optimal number of clusted based on Elbow Method

    Args:
        data_scaled (Union[pd.DataFrame, np.ndarray]): DataFrame or numpy array
        n_clusters (int, optional): Max number of clusters. Defaults to 10.
    """
    Sum_of_squared_distances = []
    K = range(1, n_clusters) # define the range of clusters we would like to cluster the data into

    for k in K:
        km = KMeans(n_clusters = k)
        km = km.fit(data_scaled)
        Sum_of_squared_distances.append(km.inertia_)

    plt.figure(figsize=(20,6))

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def save_fig(fig_name:str, tight_layout:bool=True, fig_extension:str="png", resolution:int= 300):
    """Save figure as image

    Args:
        fig_name (str): figure name
        tight_layout (bool, optional): plot with tight layout or not. Defaults to True.
        fig_extension (str, optional): the extension of the figure. Defaults to "png".
        resolution (int, optional): DPI resolution. Defaults to 300.
    """
    
    PROJECT_ROOT_DIR = ".."
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "img")
    os.makedirs(IMAGES_PATH, exist_ok= True)
    
    path = os.path.join(IMAGES_PATH, fig_name + "." + fig_extension)
    print("Saving figure", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format= fig_extension, dpi= resolution)

if __name__ == '__main__':
    pass