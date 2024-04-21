# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp  # pip install mlxtend
from sklearn.inspection import PartialDependenceDisplay#,plot_partial_dependence
from sklearn.model_selection import (GridSearchCV, learning_curve,
                                     validation_curve)
from sklearn.preprocessing import StandardScaler

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
###############################################


def get_train_data(path='../data/train.csv'):
    """This function reads the data from the path and returns the features and the target variable

    Args:
        path (str, optional): This the path of the training dataset. Defaults to '../data/train.csv'.

    Returns:
        tuple: This function returns a tuple of the features and the target variable
    """
    # read the data
    df = pd.read_csv(path)
    # drop the id column
    X = df.drop(['ID_code', 'target'], axis=1)  
    y = df['target']
    return X, y

def get_test_data(path='../data/test.csv', scaleNumericalFeatures=False):
    """This function reads the data from the path and returns the features and the target variable

    Args:
        path (str, optional): The path of the testing . Defaults to '../data/test.csv'.
        scaleNumericalFeatures (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # read the data
    df = pd.read_csv(path)
    # drop the id column
    X = df.drop(['ID_code'], axis=1)  
    # scale the numerical features
    if scaleNumericalFeatures:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X

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
    mean_values = X.mean(axis=0)
    std_values = X.std(axis=0)
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
        estimator (the return fo fit): This is the model after training
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
    plt.ylim(0.0, 1.1)
    plt.grid()

    plt.fill_between(train_sizes, 1-(train_scores_mean - train_scores_std),
                     1-(train_scores_mean + train_scores_std), alpha=0.1, color='r')
    plt.fill_between(train_sizes, 1-(test_scores_mean - test_scores_std),
                     1-(test_scores_mean + test_scores_std), alpha=0.1, color='g')

    plt.plot(train_sizes, 1-train_scores_mean, 'o-', color='r', label='Ein')
    plt.plot(train_sizes, 1-test_scores_mean, 'o-', color='g', label='Eval')

    plt.legend(loc='best')
    if save:
        plt.savefig(f'../images/{modelname}/learning_curve.png', dpi=300, bbox_inches='tight')
    return plt

# Partial Dependence Plot


# def get_partial_dependencies_plot(estimator, X, modelname='model', save=True):
#     """This function plots the partial dependencies

#     Args:
#         estimator (the return fo fit): This is the model after training
#         X (DataFrame): This is the features of the dataset
#         modelname (str, optional): The model name. Defaults to 'model'.
#         save (bool, optional): Flag to save the plot or not. Defaults to True.

#     Returns:
#         matplotlib.pyplot: This function returns the plot of the partial dependencies
#     """
#     fig, ax = plt.subplots(figsize=(15, 6), )
#     target_class = 0  # specify the target class
#     PartialDependenceDisplay.from_estimator(
#         estimator, X, X.columns, ax=ax, target=target_class)

#     # Increase spacing between subplots
#     fig.subplots_adjust(wspace=0.3, hspace=0.3)

#     fig.suptitle(f'Partial Dependence Plots for {modelname}')
#     fig.tight_layout()
#     if save:
#         plt.savefig(f'../images/{modelname}/partial_dependencies.png', dpi=300, bbox_inches='tight')
#     return plt

# def get_partial_dependencies_plot(estimator, X, modelname='model', save=True):
#     """This function plots the partial dependencies

#     Args:
#         estimator (the return fo fit): This is the model after training
#         X (DataFrame): This is the features of the dataset
#         modelname (str, optional): The model name. Defaults to 'model'.
#         save (bool, optional): Flag to save the plot or not. Defaults to True.

#     Returns:
#         matplotlib.pyplot: This function returns the plot of the partial dependencies
#     """
#     fig, ax = plt.subplots(figsize=(15, 6), )
#     target_class = 0  # specify the target class
#     plot_partial_dependence(estimator, X, features=range(X.shape[1]), ax=ax, target=target_class)

#     # Increase spacing between subplots
#     fig.subplots_adjust(wspace=0.3, hspace=0.3)

#     fig.suptitle(f'Partial Dependence Plots for {modelname}')
#     fig.tight_layout()
#     if save:
#         plt.savefig(f'../images/{modelname}/partial_dependencies.png', dpi=300, bbox_inches='tight')
#     return plt
def get_partial_dependencies_plot(estimator, X, modelname='model', save=True):
    """This function plots the partial dependencies

    Args:
        estimator (the return fo fit): This is the model after training
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
    fig.tight_layout(h_pad=2,w_pad=2)
    if save:
        plt.savefig(f'../images/{modelname}/partial_dependencies.png', dpi=300, bbox_inches='tight')
    return plt

# Grid Search


def get_grid_search(estimator, param_grid, X, y, scoring, cv=10):
    """This function performs grid search

    Args:
        estimator (the return fo fit): This is the model after training
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
        estimator (the return fo fit): This is the model after training
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
        estimator (the return fo fit): This is the model after training
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

def _get_decision_contours(estimator, f1, f2, y, ax):
    """This function plots the decision boundary

    Args:
        estimator (the return fo fit): This is the model after training
        f1 (_type_): _description_
        f2 (_type_): _description_
        y (_type_): _description_
        ax (_type_): _description_

    Returns:
        _type_: _description_
    """
    # plot the decision boundary
    x_min, x_max = f1.min() - 0.5, f1.max() + 0.5
    y_min, y_max = f2.min() - 0.1, f2.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    contours = ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(f1, f2, c=y, s=20, edgecolor='k')

    return contours
def get_decision_regions(estimator, X, f1, f2, y, C=[0.001, 0.1, 10], modelname='model', save=True):
    """This function plots the decision regions

    Args:
        estimator (the return fo fit): This is the model after training
        X (DataFrame): This is the features of the dataset
        f2 (_type_): _description_
        y (_type_): _description_
        C (list, optional): _description_. Defaults to [0.001, 0.1, 10].
        modelname (str, optional): The model name. Defaults to 'model'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.

    Returns:
        _type_: _description_
    """
    # reset style to default
    # plt.style.use('default')

    # create a decision boundary plot at 3 different C values
    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    formatter = plt.FuncFormatter(lambda val, loc: ['0','1'][val])
    plt.title(f'Decision Regions using {modelname}')
    for i, c in enumerate(C):
        estimator.set_params(C=c)
        estimator.fit(X, y)
        contours = _get_decision_contours(estimator, f1, f2, y, axes[i])
        axes[i].set_title('C = ' + str(c), size=15)
        plt.colorbar(contours, ticks =[0, 1],format=formatter)
    if save:
        plt.savefig(f'../images/{modelname}/decision_regions_for_{modelname}.png', dpi=300, bbox_inches='tight')
    return plt
