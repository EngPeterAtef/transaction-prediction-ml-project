# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp  # pip install mlxtend
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import (GridSearchCV, learning_curve,
                                     validation_curve,train_test_split)

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
###############################################
# TOTAL = 359804 #oversampling
# TOTAL = 40196 #undersampling
def get_data(path='../data/train.csv',training_size = None):
    """This function reads the data from the path and returns the features and the target variable

    Args:
        path (str, optional): This the path of the training dataset. Defaults to '../data/train.csv'.
        training_size (float, optional): The percentage of the training data. Defaults to None. if None it will be calculated like the following: 0.7 if len(df) <= 100_000 else 0.5
    Returns:
        tuple: This function returns a tuple of the features and the target variable
    """
    # read first training_size rows from the data
    df = pd.read_csv(path)
    if training_size==None:
        training_size = 0.7 if len(df) <= 100_000 else 0.5
    # drop the id column
    X = df.drop(['target'], axis=1)  
    y = df['target']
    # random indeces for train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=42,shuffle=True,stratify=y)
    return X_train, X_test, y_train, y_test

# Feature Importance

def standardize_features(X: pd.DataFrame)->pd.DataFrame:
    """This function standardizes the features of the dataset

    Args:
        X (DataFrame): This is the features of the dataset

    Returns:
        DataFrame: This function returns the standardized features
    """
    mean_values = X.mean(axis=0)
    std_values = X.std(axis=0)
    X = (X - mean_values) / std_values
    # mean_values = X.mean(axis=0)
    # std_values = X.std(axis=0)
    # print("Mean values of each feature: \n", mean_values)
    # print("Std values of each feature: \n", std_values)
    return X

def get_feature_importance(features, importance):
    """This function returns the feature importance in a dataframe.

    Args:
        features (list): The list of the features' names.
        importance (ndarray): The feature importance values.

    Returns:
        DataFrame: This function returns the feature importance in a dataframe.
    """
    # Add the feature importances into a dataframe
    feature_importance = pd.DataFrame(
        {'feature': features, 'importance': importance})
    feature_importance.sort_values('importance', ascending=False, inplace=True)
    return feature_importance


def get_feature_importance_plot(feature_importance, save=True, modelname='model'):
    """This function plots the feature importance

    Args:
        feature_importance (DataFrame): This is the feature importance dataframe.
        save (bool, optional): Flag to save or not the plot. Defaults to True.
        modelname (str, optional): The model name. Defaults to 'model'.

    Returns:
        matplotlib.pyplot: This function returns the plot of the feature importance
    """
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    # increase the spacing between the title and the plot
    plt.subplots_adjust(top=0.9)
    if save:
        plt.savefig(f'../images/{modelname}/feature_importance.png', dpi=300, bbox_inches='tight')
    return plt

# Learning Curves


def get_learning_curve_plot(estimator, X, y, cv=5, scoring='f1_weighted', modelname='model', save=True):
    """This function plots the learning curve

    Args:
        estimator (the return of fit): This is the model after training
        X (DataFrame): This is the features of the dataset
        modelname (str, optional): The model name. Defaults to 'model'.
        y (_type_): _description_
        cv (int, optional): _description_. Defaults to 5.
        scoring (str, optional): _description_. Defaults to 'f1_weighted'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.

    Returns:
        matplotlib.pyplot:: This function returns the plot of the learning curve
    """
    # It uses cross-validation with cv folds
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            train_sizes=np.linspace(
                                                                .1, 1.0, 5),
                                                            scoring='f1_weighted', shuffle=True, random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title(f'Learning Curves for {modelname}')
    plt.xlabel('Training Set Size')
    plt.ylabel(f'1 - {scoring}')
    plt.grid()

    plt.fill_between(train_sizes, 1-(train_scores_mean - train_scores_std),
                     1-(train_scores_mean + train_scores_std), alpha=0.1, color='r')
    plt.fill_between(train_sizes, 1-(test_scores_mean - test_scores_std),
                     1-(test_scores_mean + test_scores_std), alpha=0.1, color='g')

    plt.plot(train_sizes, 1-train_scores_mean, 'o-', color='r', label='Ein')
    plt.plot(train_sizes, 1-test_scores_mean, 'o-', color='g', label='Eval')
    max_val = np.mean(1-(test_scores_mean + test_scores_std))+ np.mean(1-(train_scores_mean + train_scores_std))
    
    plt.ylim(0.0, max_val*2)
    plt.legend(loc='best')
    if save:
        plt.savefig(f'../images/{modelname}/learning_curve.png', dpi=300, bbox_inches='tight')
    return plt

# Partial Dependence Plot
def get_partial_dependencies_plot(estimator, X, modelname='model', save=True):
    """This function plots the partial dependencies

    Args:
        estimator (the return of fit): This is the model after training
        X (DataFrame): This is the features of the dataset
        modelname (str, optional): The model name. Defaults to 'model'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.

    Returns:
        matplotlib.pyplot: This function returns the plot of the partial dependencies
    """
    # Create a grid layout for the subplots
    ncols = 3  # Number of columns in the grid
    nrows = (X.shape[1] + ncols - 1) // ncols  # Number of rows calculated based on number of features
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6 * nrows))

    # Flatten the axes if there's only one row
    if nrows == 1:
        axes = axes.reshape(1, -1)

    target_class = 0  # specify the target class
    for i, ax in enumerate(axes.flatten()):
        if i < X.shape[1]:
            PartialDependenceDisplay.from_estimator(
                estimator, X, [X.columns[i]], ax=ax, target=target_class)
            ax.set_title(f'Partial dependence of {X.columns[i]}')
        else:
            ax.axis('off')  # Hide extra subplots if there are no more features

    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.suptitle(f'Partial Dependence Plots for {modelname}')
    # title at the top of the image
    fig.tight_layout(h_pad=0,w_pad=0)
    if save:
        plt.savefig(f'../images/{modelname}/partial_dependencies.png', dpi=300, bbox_inches='tight')
    return plt

# Grid Search


def get_grid_search(estimator, param_grid, X, y, scoring, cv=10):
    """This function performs grid search

    Args:
        estimator (the return of fit): This is the model after training
        param_grid (dict): This is the hyperparameters grid
        X (DataFrame): This is the features of the dataset
        y (DataFrame): This is the target variable
        scoring (str|list): This is the scoring metric or list of scoring metrics
        cv (int, optional): The number of flods used to estimate the model parameters. Defaults to 10.

    Returns:
        GridSearchCV: This function returns the grid search object
    """
    # Grid Search
    grid_search = GridSearchCV(
        estimator, param_grid, cv=cv, scoring=scoring, return_train_score=True)
    grid_search.fit(X, y)
    return grid_search


def plot_hyper_param_heat_maps(param_grid, grid_search, modelname='model', save=True):
    """This function plots the hyperparameter heat maps

    Args:
        param_grid (dict): This is the hyperparameters grid
        grid_search (GridSearchCV): This is the grid search object
        modelname (str, optional): The model name. Defaults to 'model'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.

    Returns:
        matplotlib.pyplot: This function returns the plot of the hyperparameter heat maps
    """
    # Create dataframe of validation accuracy for each hyperparameter combination
    results = pd.DataFrame(grid_search.cv_results_)[
        ['params', 'mean_test_score']]
    for param in param_grid:
        results[param] = results['params'].apply(lambda x: x[param])
    results.drop(columns=['params'], inplace=True)
    vis_num = len(param_grid.keys())
    fig, axes = plt.subplots(1, vis_num, figsize=(vis_num*5, 5))
    # Loop through all combinations of hyperparameters and plot heatmap of validation accuracy
    index = 0
    for i, param1 in enumerate(param_grid.keys()):
        for j, param2 in enumerate(list(param_grid.keys())[i+1:]):
            heatmap_data = results.pivot_table(
                index=param1, columns=param2, values='mean_test_score')
            heatmap_data.index = heatmap_data.index.astype(str)

            # Plot heatmap of validation accuracy for each hyperparameter combination
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=axes[index])
            axes[index].set_title(f"{param1} and {param2}")
            index += 1
    # set title for the figure
    fig.suptitle(f'Hyperparameter Heat Maps for {modelname}')
    if save:
        plt.savefig(f'../images/{modelname}/hyper_param_heat_maps.png', dpi=300, bbox_inches='tight')
    return plt
            


# Train-Validation Curve
def plot_hyper_param_train_validation_curve(estimator, param_grid, X, y, cv=10, scoring='f1_weighted', modelname='model', save=True):
    """This function plots the hyperparameter train-validation curve

    Args:
        estimator (the return of fit): This is the model after training
        param_grid (dict): This is the hyperparameters grid
        X (DataFrame): This is the features of the dataset
        y (DataFrame): This is the target variable
        cv (int, optional): The number of flods used to estimate the model parameters. Defaults to 10.
        scoring (str|list): This is the scoring metric or list of scoring metrics
        modelname (str, optional): The model name. Defaults to 'model'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.
    """
    # iterate over the parameters and get the key and value pairs
    for param, value in param_grid.items():
        # Calculate training and validation scores for different values of max_depth
        train_scores, valid_scores = validation_curve(estimator, X, y,
                                                      param_name=param, param_range=value,
                                                      cv=cv, scoring=scoring)
        # Calculate the mean and standard deviation of the training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)

        # Plot the bias-variance tradeoff

        plt.plot(value, train_mean, label='Training score', color='blue')
        plt.fill_between(value, train_mean - train_std,
                         train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(value, valid_mean, label='Cross-validation score', color='red')
        plt.fill_between(value, valid_mean - valid_std,
                         valid_mean + valid_std, alpha=0.2, color='red')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel(scoring)
        plt.title(f'Bias-Variance Tradeoff for {param} using {modelname}')
        if save:
            plt.savefig(f'../images/{modelname}/hyper_param_train_val_{param}_{value}.png', dpi=300, bbox_inches='tight')
        plt.show()


# Bias-Variance Analysis
def get_bias_variance(estimator, X_train, y_train, X_test, y_test):
    """This function performs the bias-variance analysis

    Args:
        estimator (the return of fit): This is the model after training
        X (DataFrame): This is the features of the training dataset
        y (DataFrame): This is the target variable of the training dataset
        X_test (DataFrame): This is the features of the testing dataset
        y_test (DataFrame): This is the target variable of the testing dataset

    Returns:
        tuple: This function returns the mean squared error, bias and variance
    """
    # convert X, y, X_test, y_test to numpy arrays
    XX = X_train.to_numpy()
    yy = y_train.to_numpy()
    XX_test = X_test.to_numpy()
    yy_test = y_test.to_numpy()

    # perform the bias-variance analysis
    mse, bias, var = bias_variance_decomp(
        estimator, X_train=XX, y_train=yy, X_test=XX_test, y_test=yy_test, loss='mse', random_seed=42)
    return mse, bias, var
