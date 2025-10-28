import pandas as pd
import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

from wine_local import X_train

# Load the dataset
wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# Separate dataset into X and y
X = wine_df.drop("target", axis=1)
y = wine_df["target"]

# Split the dataset into train and test
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Create an classifier object
rf = RandomForestClassifier(random_state=47)

# Define the hyperparameters
param_grid = {
    "n_estimators": [10, 50, 100, 150, 200, 250, 300],
    "max_depth": [None, 5, 10, 15, 20, 25, 30]
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Start MLFlow
mlflow.set_experiment("Wine_Classification-RF")
with mlflow.start_run(run_name="wine_hypertune"):
    grid_search.fit(X_train, y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Find the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # Calculate precision, recall, and f1 score
    y_pred = grid_search.predict(X_test)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    # Store the training and testing data
    train_data = X_train.copy()
    train_data["target"] = y_train
    train_data = mlflow.data.from_pandas(train_data)
    test_data = X_test.copy()
    test_data["target"] = y_test
    test_data = mlflow.data.from_pandas(test_data)

    # Now set logging/tracking
    mlflow.log_param("best_params", best_params)
    mlflow.log_metrics({"accuracy": best_score, "precision": precision, "recall": recall, "f1": f1})
    mlflow.log_input(train_data, context="training")
    mlflow.log_input(test_data, context="testing")
    mlflow.sklearn.log_model(grid_search.best_estimator_, "rf_wine_tuned_model")
    mlflow.set_tag("Author", "Jeet")
    mlflow.log_artifact(__file__)

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

