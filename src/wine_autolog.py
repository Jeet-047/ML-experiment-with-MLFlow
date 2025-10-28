import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the wine dataset
wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# Separate X and y
X = wine_df.drop('target', axis=1)
y = wine_df['target']

# Split the dataset into train, test and split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=47)

# Define the RandomForest parameters
n_estimators = 100
max_depth = None

# Start the mlflow
mlflow.set_experiment("Wine_Classification-RF")
mlflow.autolog()
with mlflow.start_run(run_name="use_auto-log"):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=47)
    rf.fit(X_train, y_train)
    # calculate y-predict
    y_pred = rf.predict(X_test)

    # calculate evaluation scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    # create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png") # save the confusion matrix

    # Now start logging using mlflow
    # ## log metrics
    # mlflow.log_metric("accuracy", accuracy) # use for tracking one parameter at a time
    # mlflow.log_metrics({"precision": precision, "recall": recall, "f1": f1})
    # ## log parameters
    # mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
    # mlflow.log_param("test_size", test_size)
    ## log artifacts
    mlflow.log_artifacts("confusion_matrix.png")
    mlflow.log_artifacts(__file__)
    ## log the model
    # # Log model with signature
    # example_input = pd.DataFrame(X_train[:2], columns=X_train.columns).astype(float)
    # mlflow.sklearn.log_model(rf, name="rf_wine_model", input_example=example_input)

    # Now add tags
    mlflow.set_tags({"Author": "Jeet", "Task": "Wine Quality Classification"})
    
    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F1", f1)
