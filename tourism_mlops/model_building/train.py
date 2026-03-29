# for data manipulation
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-production-experiment")

api = HfApi()

# Load train/test data from Hugging Face
Xtrain_path = "hf://datasets/satyabrd123/Wellness-Tourism-Dataset/Xtrain.csv"
Xtest_path = "hf://datasets/satyabrd123/Wellness-Tourism-Dataset/Xtest.csv"
ytrain_path = "hf://datasets/satyabrd123/Wellness-Tourism-Dataset/ytrain.csv"
ytest_path = "hf://datasets/satyabrd123/Wellness-Tourism-Dataset/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print(f"Train set size: {len(Xtrain)}")
print(f"Test set size: {len(Xtest)}")

# Calculate class weight
class_weight_ratio = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define models to compare
models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=class_weight_ratio, random_state=42)
}

# Define parameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}

best_models = {}
best_scores = {}

# Start MLflow run
with mlflow.start_run():
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=3, 
            scoring='f1', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(Xtrain, ytrain)
        
        # Log all parameter combinations
        results = grid_search.cv_results_
        for i in range(len(results['params'])):
            param_set = results['params'][i]
            mean_score = results['mean_test_score'][i]
            std_score = results['std_test_score'][i]
            
            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_type", model_name)
                mlflow.log_params(param_set)
                mlflow.log_metric("mean_test_score", mean_score)
                mlflow.log_metric("std_test_score", std_score)
        
        # Store best model
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        best_scores[model_name] = grid_search.best_score_
        
        # Make predictions
        y_pred_train = best_model.predict(Xtrain)
        y_pred_test = best_model.predict(Xtest)
        
        # Calculate metrics
        train_metrics = {
            f"{model_name}_train_accuracy": accuracy_score(ytrain, y_pred_train),
            f"{model_name}_train_precision": precision_score(ytrain, y_pred_train),
            f"{model_name}_train_recall": recall_score(ytrain, y_pred_train),
            f"{model_name}_train_f1": f1_score(ytrain, y_pred_train)
        }
        
        test_metrics = {
            f"{model_name}_test_accuracy": accuracy_score(ytest, y_pred_test),
            f"{model_name}_test_precision": precision_score(ytest, y_pred_test),
            f"{model_name}_test_recall": recall_score(ytest, y_pred_test),
            f"{model_name}_test_f1": f1_score(ytest, y_pred_test)
        }
        
        # Log metrics for this model
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_type", model_name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({**train_metrics, **test_metrics})
        
        print(f"\n{model_name} Best Parameters:", grid_search.best_params_)
        print(f"{model_name} Test F1 Score:", test_metrics[f"{model_name}_test_f1"])
    
    # Select best model overall
    best_model_name = max(best_scores, key=best_scores.get)
    best_model_overall = best_models[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Best CV F1 Score: {best_scores[best_model_name]:.4f}")
    print(f"{'='*60}")
    
    # Save the best model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model_overall, model_path)
    
    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_param("best_model_type", best_model_name)
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "satyabrd123/tourism-prediction-model"
    repo_type = "model"

    # Check if the model repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repository '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_model_v1.joblib",
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {repo_id}")
