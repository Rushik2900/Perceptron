import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib 
from matplotlib.colors import ListedColormap
import os
import logging


def prepare_data(df):
    """ It is used to seperate dependent and independent variables

    Args:
        df (pd.DataFrame): It is pandas DataFrame 

    Returns:
        tuple: It returns tuple of dependent and independent variables
    """
    logging.info("preparing the data by segregating independent and dependent variable")
    X=df.drop("y", axis=1)
    y=df["y"]
    return X,y


def save_model(model, filename):
    """ This saves the trained model to filename

    Args:
        model (python object): trained model to
        filename (str): path to save trained model
    """
    logging.info("saving the train model")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True) # only create if model_dir doesn't exists
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    logging.info(f"save the model at {filepath}")


def save_plot(df,file_name, model):
    """
    :param df: It is a DataFrame
    :param file_name: It is a path to save file
    :param model: trained model
    """
    logging
    def _create_base_plot(df):
        logging.info("creating the base plot")
        df.plot(kind="scatter", x="x1", y='x2', c='y', s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure=plt.gcf()   #get current figure
        figure.set_size_inches(10,8)
    
    def _plot_decision_regions(X,y, classifier, resolution=0.02):
        logging.info("plotting the decision regions")
        colors=("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])
        
        X=X.values # As an array
        x1_min, x1_max = X[:,0].min() -1, X[:,0].max() +1
        x2_min, x2_max = X[:,1].min() -1, X[:,1].max() +1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                              np.arange(x2_min, x2_max, resolution))
        logging.info(xx1)
        logging.info(xx1.ravel())
        
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.xlim(xx2.min(), xx2.max())
        
        plt.plot()
    
    X, y = prepare_data(df)
    
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # only create if model_dir doesn't exists
    plotpath = os.path.join(plot_dir, file_name)
    plt.savefig(plotpath)
    logging.info(f"saved plot at {plotpath}")