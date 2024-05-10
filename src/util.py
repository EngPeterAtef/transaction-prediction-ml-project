import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp  # pip install mlxtend
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import (GridSearchCV, learning_curve, validation_curve,train_test_split)
# The above code is a Python script that imports various libraries and modules for data visualization,
# machine learning, and model evaluation. Here is a breakdown of the code:

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
# The above code is using the `seaborn` library in Python to set the plotting style to 'darkgrid' with
# a light gray background color for the axes and a slightly darker gray color for the grid lines. It
# also sets the color palette to 'deep' and stores the color palette in the variable `sns_c`.
###############################################

def get_data(path='../data/train.csv',training_size = None):
    """
    Function: get_data

    The function `get_data` reads data from a specified path, splits it into features and target
    variable, and returns training and testing datasets based on the specified training size or a
    default value.
    
    :param path: The `path` parameter in the `get_data` function is the path to the training dataset. By
    default, it is set to `'../data/train.csv'`. This is the location where the function will look for
    the training data unless a different path is specified when calling the function, defaults to
    ../data/train.csv (optional)
    :param training_size: The `training_size` parameter in the `get_data` function is used to specify
    the percentage of the training data to be used. If this parameter is not provided (i.e., it is set
    to `None`), the function calculates the training size based on the length of the dataset. If
    :return: The function `get_data` returns a tuple containing the training features (`X_train`),
    testing features (`X_test`), training target variable (`y_train`), and testing target variable
    (`y_test`).

Args:
    path (str, optional): This parameter specifies the path to the training dataset. By default, it is set 
    to `'../data/train.csv'`. This is the location where the function will look for the training data unless 
    a different path is specified when calling the function. (optional)
    training_size (float, optional): This parameter is used to specify the percentage of the training data to 
    be used. If this parameter is not provided (i.e., it is set to `None`), the function calculates the training 
    size based on the length of the dataset. (optional)

Returns:
    tuple: This function returns a tuple containing the training features (`X_train`), testing features (`X_test`), 
    training target variable (`y_train`), and testing target variable (`y_test`).

Notes:
    - The function reads the training dataset from the specified path and assigns it to the DataFrame `df`.
    - If the `training_size` parameter is not provided, it is calculated based on the length of the dataset.
    - The target variable is extracted from the DataFrame `df` and stored in the variable `y`, while the features 
      are obtained by dropping the target variable from `df` and stored in the DataFrame `X`.
    - The function then splits the data into training and testing sets using `train_test_split` from scikit-learn.
    - The default split ratio is 70% for training and 30% for testing if the length of the dataset is less than or 
      equal to 100,000, otherwise, the split ratio is 50% for training and 50% for testing.

    """
    # read first training_size rows from the data
    df = pd.read_csv(path)

    # If training_size is not provided, calculate it based on the length of the dataset
    if training_size == None:
        training_size = 0.7 if len(df) <= 100_000 else 0.5

    # Drop the target column to get features (X) and assign the target column to y
    X = df.drop(['target'], axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=100, shuffle=True, stratify=y)
    
    # Return the training and testing datasets
    return X_train, X_test, y_train, y_test


def standardize_features(X: pd.DataFrame)->pd.DataFrame:
    """
    Function: standardize_features
    This function standardizes the features of a dataset by subtracting the mean and dividing by the
    standard deviation.
    
    Description:
    
    :param X: X is a pandas DataFrame containing the features of a dataset. The function
    `standardize_features` takes this DataFrame as input and standardizes the features by subtracting
    the mean values and dividing by the standard deviation values of each feature. The function then
    returns the standardized DataFrame
    :type X: pd.DataFrame
    :return: The function `standardize_features` returns the standardized features of the dataset after
    applying the standardization process.
Args:
    X (DataFrame): This parameter represents the features of the dataset. The function `standardize_features` takes 
    this DataFrame as input and standardizes the features by subtracting the mean values and dividing by the standard 
    deviation values of each feature.

Returns:
    DataFrame: This function returns the standardized features.

Notes:
    - The function computes the mean and standard deviation of each feature in the DataFrame `X`.
    - It then standardizes the features by subtracting the mean and dividing by the standard deviation.
    - The standardized features are returned as a DataFrame.

Example:
    # Importing necessary libraries
    import pandas as pd

    # Create a DataFrame with features
    features_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    # Standardize features
    standardized_features_df = standardize_features(X=features_df)

    """
    # Compute the mean values of each feature
    mean_values = X.mean(axis=0)

    # Compute the standard deviation values of each feature
    std_values = X.std(axis=0)

    # Standardize the features by subtracting the mean and dividing by the standard deviation
    X = (X - mean_values) / std_values

    # Return the standardized features
    return X


def get_feature_importance(features, importance):
    """
    Function: get_feature_importance

Description:
    The function `get_feature_importance` takes a list of features and their importance values, and
    returns them in a DataFrame sorted by importance.
    
    :param features: Features is a list containing the names of the features used in a machine learning
    model. For example, features could be ['age', 'gender', 'income', 'education'] representing
    different attributes of individuals in a dataset
    :param importance: The `importance` parameter should be a NumPy ndarray containing the feature
    importance values for each feature in the `features` list. You can pass this ndarray to the
    `get_feature_importance` function along with the list of feature names to create a DataFrame showing
    the feature importance values sorted in descending
    :return: The function `get_feature_importance` returns the feature importance in a DataFrame, where
    each row contains a feature name and its corresponding importance value. The DataFrame is sorted in
    descending order based on the importance values.



Args:
    features (list): This parameter represents a list containing the names of the features used in a machine learning model.
    importance (ndarray): This parameter should be a NumPy ndarray containing the feature importance values for each feature in the `features` list.

Returns:
    DataFrame: This function returns the feature importance in a DataFrame, where each row contains a feature name and its corresponding importance value. The DataFrame is sorted in descending order based on the importance values.

Notes:
    - The function creates a DataFrame containing the feature names and their corresponding importance values.
    - It sorts the DataFrame in descending order based on the importance values.

Example:
    # Importing necessary libraries
    import numpy as np

    # Define features and their importance values
    features_list = ['age', 'gender', 'income', 'education']
    importance_values = np.array([0.5, 0.3, 0.8, 0.7])

    # Get feature importance DataFrame
    feature_importance_df = get_feature_importance(features=features_list, importance=importance_values)

    """
    # Add the feature importances into a DataFrame
    feature_importance = pd.DataFrame({'feature': features, 'importance': importance})

    # Sort the DataFrame in descending order based on the importance values
    feature_importance.sort_values('importance', ascending=False, inplace=True)

    # Return the feature importance DataFrame
    return feature_importance



def get_feature_importance_plot(feature_importance, save=True, modelname='model'):
    """Function: get_feature_importance_plot

Description:
    This function plots the feature importance to visualize the importance of each feature in a model. Feature importance quantifies the contribution of each feature to the model's performance.

Args:
    feature_importance (DataFrame): DataFrame containing feature names and their corresponding importance scores.
    save (bool, optional): Flag to save the plot. Defaults to True.
    modelname (str, optional): Name of the model. Defaults to 'model'.

Returns:
    matplotlib.pyplot: Plot of the feature importance.

Notes:
    - Feature importance plots help identify which features are most influential in predicting the target variable.
    - The function creates a bar plot with feature names on the x-axis and their importance scores on the y-axis.
    - If 'save' is set to True, the plot is saved as a PNG image in the '../images/{modelname}' directory.

Example:
    # Importing necessary libraries
    import pandas as pd

    # Create a DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({'feature': ['feature1', 'feature2', 'feature3'],'importance': [0.2, 0.5, 0.3]})

    # Plot feature importance
    get_feature_importance_plot(feature_importance=feature_importance_df, save=True, modelname='RandomForest')

    """
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')

    # Increase the spacing between the title and the plot
    plt.subplots_adjust(top=0.9)

    # Save the plot if 'save' is True
    if save:
        plt.savefig(f'../images/{modelname}/feature_importance.png', dpi=300, bbox_inches='tight')

    # Return the plot
    return plt


def get_learning_curve_plot(estimator, X, y, cv=5, scoring='f1_weighted', modelname='model', save=True):
    """
    Function: get_learning_curve_plot

    Description:
        This function plots the learning curve to visualize the performance of a model as a function of training set size. It demonstrates how the model's training and validation scores evolve with an increasing number of training examples.

    Args:
        estimator (object): The model after training. It is expected to have a 'fit' method.
        X (DataFrame or array-like): Features of the dataset.
        y (DataFrame or array-like): Target variable of the dataset.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        scoring (str, optional): Scoring metric used to evaluate the model's performance. Defaults to 'f1_weighted'.
        modelname (str, optional): Name of the model. Defaults to 'model'.
        save (bool, optional): Flag to save the plot. Defaults to True.

    Returns:
        matplotlib.pyplot: Plot of the learning curve.

    Notes:
        - Learning curves provide insights into a model's bias and variance, as well as whether the model would benefit from additional data.
        - The function uses cross-validation to compute the learning curve.
        - It computes the mean and standard deviation of training and testing scores across different training set sizes.
        - The learning curve is plotted with shaded areas representing the standard deviation from the mean scores.
        - The y-axis represents the error (1 - scoring) to align with the convention that lower values indicate better performance.
        - If 'save' is set to True, the plot is saved as a PNG image in the '../images/{modelname}' directory.

    Example:
        # Importing necessary libraries
        from sklearn.svm import SVC

        # Instantiate a Support Vector Classifier
        svc_classifier = SVC()

        # Plot learning curve
        get_learning_curve_plot(estimator=svc_classifier, X=X_train, y=y_train, modelname='SVC', save=True)

    """
    # Compute learning curve using cross-validation with cv folds
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,train_sizes=np.linspace(.1, 1.0, 5),scoring='f1_weighted', shuffle=True, random_state=42)

    # Compute mean and standard deviation of training and testing scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.figure(figsize=(8, 6))
    plt.title(f'Learning Curves for {modelname}')
    plt.xlabel('Training Set Size')
    plt.ylabel(f'1 - {scoring}')
    plt.grid()

    # Draw shaded area for the standard deviation (distance from the mean)
    plt.fill_between(train_sizes, 1-(train_scores_mean - train_scores_std),1-(train_scores_mean + train_scores_std), alpha=0.1, color='r')
    plt.fill_between(train_sizes, 1-(test_scores_mean - test_scores_std),1-(test_scores_mean + test_scores_std), alpha=0.1, color='g')

    # Plot the mean training and testing scores
    plt.plot(train_sizes, 1-train_scores_mean, 'o-', color='r', label='Ein')
    plt.plot(train_sizes, 1-test_scores_mean, 'o-', color='g', label='Eval')

    # Get the maximum value of the plot
    max_val = np.mean(1-(test_scores_mean - test_scores_std))+ np.mean(1-(train_scores_mean - train_scores_std))

    # Set the y-axis limit
    plt.ylim(0.0, max_val*2)
    plt.legend(loc='best')

    # Save the plot if 'save' is True
    if save:
        plt.savefig(f'../images/{modelname}/learning_curve.png', dpi=300, bbox_inches='tight')

    # Return the plot
    return plt

def get_partial_dependencies_plot(estimator, X, modelname='model', save=True):
    """
    Function: get_partial_dependencies_plot

    Description:
        This function plots the partial dependencies to visualize the relationship between the target variable and each individual feature in the dataset. Partial dependence plots show how the predicted outcome (target) changes as a function of a single feature, while accounting for the average effects of all other features.

    Args:
        estimator (the return of fit): This is the model after training
        X (DataFrame): This is the features of the dataset
        modelname (str, optional): The model name. Defaults to 'model'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.

    Returns:
        matplotlib.pyplot: Plot of the partial dependencies.

    Notes:
        - Partial dependence plots provide insights into the relationship between individual features and the target variable in a machine learning model.
        - The function generates partial dependence plots for each feature in the dataset.
        - It utilizes the PartialDependenceDisplay class from scikit-learn to create the plots.
        - The 'target_class' variable specifies the target class for classification models (default is set to 0).
        - The resulting plots visualize the effect of each feature on the predicted outcome while holding other features constant.
        - If 'save' is set to True, the plot is saved as a PNG image in the '../images/{modelname}' directory.
    Example:
        # Importing necessary libraries
        from sklearn.ensemble import RandomForestRegressor

        # Instantiate a Random Forest regressor
        rf_regressor = RandomForestRegressor()

        # Plot partial dependencies
        get_partial_dependencies_plot(estimator=rf_regressor, X=X_train, modelname='RandomForest', save=True)
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
            # Generate partial dependence plot for the current feature
            PartialDependenceDisplay.from_estimator(
                estimator, X, [X.columns[i]], ax=ax, target=target_class)
            ax.set_title(f'Partial dependence of {X.columns[i]}')
        else:
            ax.axis('off')  # Hide extra subplots if there are no more features

    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # Set title for the entire figure
    fig.suptitle(f'Partial Dependence Plots for {modelname}')

    # Adjust layout to avoid overlapping titles
    fig.tight_layout(h_pad=0, w_pad=0)

    # Save the plot if 'save' is True
    if save:
        plt.savefig(f'../images/{modelname}/partial_dependencies.png', dpi=300, bbox_inches='tight')

    # Return the plot
    return plt



def get_grid_search(estimator, param_grid, X, y, scoring, cv=10):
    """
    Function: get_grid_search

    Description:
        This function performs grid search to find the best hyperparameters for a given estimator using cross-validated evaluation.

   Args:
        estimator (the return of fit): This is the model after training
        param_grid (dict): This is the hyperparameters grid
        X (DataFrame): This is the features of the dataset
        y (DataFrame): This is the target variable
        scoring (str|list): This is the scoring metric or list of scoring metrics
        cv (int, optional): The number of flods used to estimate the model parameters. Defaults to 10.

    Returns:
        GridSearchCV: Grid search object containing the results of the hyperparameter tuning process.

    Notes:
        - Grid search is performed to exhaustively search through a specified hyperparameter grid and find the combination of hyperparameters that maximizes the model's performance.
        - The 'GridSearchCV' class from scikit-learn is used to perform grid search with cross-validation.
        - The 'scoring' parameter specifies the metric(s) used to evaluate the model's performance during cross-validation.
        - The 'return_train_score' parameter is set to True to return the training scores along with the validation scores.
        - The grid search object is fitted to the dataset (X, y) to find the best hyperparameters.
        - The function returns the grid search object containing the results of the hyperparameter tuning process.

    Example:
        # Importing necessary libraries
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier

        # Define hyperparameters grid
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 6, 9]}

        # Instantiate a Random Forest classifier
        rf_clf = RandomForestClassifier()

        # Perform grid search
        grid_search = get_grid_search(estimator=rf_clf, param_grid=param_grid, X=X_train, y=y_train, scoring='accuracy', cv=5)
    """
    # Grid Search
    # Initialize GridSearchCV object with the provided estimator, hyperparameter grid, cross-validation folds,
    # scoring metric, and option to return training scores
    grid_search = GridSearchCV(
        estimator, param_grid, cv=cv, scoring=scoring, return_train_score=True)
    
    # Fit the grid search object to the dataset to find the best hyperparameters
    grid_search.fit(X, y)
    
    # Return the grid search object containing the results of the hyperparameter tuning process
    return grid_search


def plot_hyper_param_heat_maps(param_grid, grid_search, modelname='model', save=True):
    """
    Function: plot_hyper_param_heat_maps

    Description:
        This function plots the hyperparameter heat maps to visualize the performance of a model across different combinations of hyperparameters. It utilizes the results obtained from a grid search conducted on the hyperparameter grid.

    Args:
        param_grid (dict): Dictionary specifying the hyperparameters and their ranges.
        grid_search (GridSearchCV): Grid search object containing the results of the hyperparameter tuning process.
        modelname (str, optional): Name of the model. Defaults to 'model'.
        save (bool, optional): Flag to save the plot. Defaults to True.

    Returns:
        matplotlib.pyplot: The plot of the hyperparameter heat maps.

    Notes:
        - The function creates a DataFrame ('results') from the 'cv_results_' attribute of the grid search object, containing the mean test scores for each hyperparameter combination.
        - It then generates heat maps for all possible combinations of hyperparameters using seaborn's heatmap function.
        - Each heat map represents the performance (mean test score) of the model for a specific pair of hyperparameters.
        - The color intensity in the heat maps indicates the performance level, with warmer colors representing higher scores.
        - The function returns the matplotlib.pyplot object representing the plot of hyperparameter heat maps.
        - If 'save' is set to True, the plot is saved as a PNG image in the '../images/{modelname}' directory.

    Example:
        # Importing necessary libraries
        from sklearn.model_selection import GridSearchCV
        import seaborn as sns

        # Define hyperparameters grid
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.001, 0.01]}

        # Instantiate a grid search object
        grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5)

        # Fit the grid search to the data

        # Plot hyperparameter heat maps
        plot_hyper_param_heat_maps(param_grid=param_grid, grid_search=grid_search, modelname='SVM', save=True)
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
            

def plot_hyper_param_train_validation_curve(estimator, param_grid, X, y, cv=10, scoring='f1_weighted', modelname='model', save=True):
    """
    Function: plot_hyper_param_train_validation_curve

    Description:
        This function plots the hyperparameter train-validation curve to visualize the bias-variance tradeoff for different hyperparameter values of a given estimator.

    Args:
        estimator (the return of fit): This is the model after training
        param_grid (dict): This is the hyperparameters grid
        X (DataFrame): This is the features of the dataset
        y (DataFrame): This is the target variable
        cv (int, optional): The number of flods used to estimate the model parameters. Defaults to 10.
        scoring (str|list): This is the scoring metric or list of scoring metrics
        modelname (str, optional): The model name. Defaults to 'model'.
        save (bool, optional): Flag to save the plot or not. Defaults to True.

    Returns:
        None

    Notes:
        - The function iterates over each hyperparameter specified in 'param_grid' and plots the training and cross-validation scores across the range of values for that hyperparameter.
        - The 'validation_curve' function from scikit-learn is used to calculate the training and validation scores.
        - The bias-variance tradeoff is visualized by plotting the mean training and validation scores, along with the corresponding standard deviations.
        - Each plot represents the bias-variance tradeoff for a specific hyperparameter, with the hyperparameter values on the x-axis and the scoring metric on the y-axis.
        - If 'save' is set to True, the plots are saved as PNG images in the '../images/{modelname}' directory, with filenames indicating the hyperparameter and its range of values.

    Example:
        # Importing necessary libraries
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import validation_curve
        import numpy as np

        # Generating synthetic data
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

        # Define hyperparameters and their ranges
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 6, 9]}

        # Instantiate and train a Random Forest classifier
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train, y_train)

        # Plotting hyperparameter train-validation curves
        plot_hyper_param_train_validation_curve(estimator=rf_clf, param_grid=param_grid, X=X, y=y, cv=5, scoring='f1_macro', modelname='RandomForest', save=True)

    """
    # iterate over the parameters and get the key and value pairs
    for param, value in param_grid.items():
        # Calculate training and validation scores for different values of max_depth
        train_scores, valid_scores = validation_curve(estimator, X, y,param_name=param, param_range=value,cv=cv, scoring=scoring)
        # Calculate the mean and standard deviation of the training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)

        # Plot the bias-variance tradeoff

        plt.plot(value, train_mean, label='Training score', color='blue')
        plt.fill_between(value, train_mean - train_std,train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(value, valid_mean, label='Cross-validation score', color='red')
        plt.fill_between(value, valid_mean - valid_std,valid_mean + valid_std, alpha=0.2, color='red')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel(scoring)
        plt.title(f'Bias-Variance Tradeoff for {param} using {modelname}')
        if save:
            plt.savefig(f'../images/{modelname}/hyper_param_train_val_{param}_{value}.png', dpi=300, bbox_inches='tight')
        plt.show()


def get_bias_variance(estimator, X_train, y_train, X_test, y_test):
    """
    Function: get_bias_variance
    Description:
        This function performs the bias-variance analysis for a given estimator, which is a model trained on a dataset. It assesses the model's performance by decomposing the mean squared error into bias and variance components.

        Args:
            estimator (the return of fit): This is the model after training
            X (DataFrame): This is the features of the training dataset
            y (DataFrame): This is the target variable of the training dataset
            X_test (DataFrame): This is the features of the testing dataset
            y_test (DataFrame): This is the target variable of the testing dataset

    Returns:
        tuple: A tuple containing three elements: mean squared error (mse), bias, and variance.

    Notes:
        - Ensure that the 'estimator' object has been trained (fit) before passing it to this function.
        - The input datasets (X_train, y_train, X_test, y_test) can be either DataFrame or array-like objects.
        - The returned mse, bias, and variance provide insights into the performance and generalization of the model.
        - The bias-variance decomposition is computed based on the provided estimator and datasets, using the 'bias_variance_decomp' function from the mlxtend library.
        - The random seed for reproducibility is fixed to 42.

    Example:
        # Importing necessary libraries
        from mlxtend.evaluate import bias_variance_decomp
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        # Generating synthetic data
        X, y = make_regression(n_samples=100, n_features=1, noise=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instantiating and training a linear regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Calculating bias, variance, and mean squared error using the function
        mse, bias, var = get_bias_variance(lr, X_train, y_train, X_test, y_test)
        print("Mean Squared Error:", mse)
        print("Bias:", bias)
        print("Variance:", var)

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
