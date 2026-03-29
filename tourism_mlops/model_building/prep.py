# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/satyabrd123/Wellness-Tourism-Dataset/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {tourism_dataset.shape}")

# Define the target variable for the classification task
target = 'ProdTaken'

# Handle missing values
print("\nHandling missing values...")
# For numerical columns, fill with median
numerical_cols = tourism_dataset.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if tourism_dataset[col].isnull().sum() > 0:
        tourism_dataset[col].fillna(tourism_dataset[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = tourism_dataset.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if tourism_dataset[col].isnull().sum() > 0:
        tourism_dataset[col].fillna(tourism_dataset[col].mode()[0], inplace=True)

print("Missing values handled.")

# Remove unnecessary columns
columns_to_drop = ['CustomerID']
if 'Unnamed: 0' in tourism_dataset.columns:
    columns_to_drop.append('Unnamed: 0')

tourism_dataset = tourism_dataset.drop(columns=columns_to_drop, errors='ignore')
print(f"Columns after dropping: {list(tourism_dataset.columns)}")

# Label encoding for categorical features
print("\nEncoding categorical features...")
label_encoders = {}
categorical_features = tourism_dataset.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    tourism_dataset[col] = le.fit_transform(tourism_dataset[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} unique values")

print("Encoding complete.")

# Define predictor matrix (X) and target (y)
X = tourism_dataset.drop(target, axis=1)
y = tourism_dataset[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # Maintain class distribution in both sets
)

print(f"\nTrain set size: {len(Xtrain)}")
print(f"Test set size: {len(Xtest)}")
print(f"Target distribution in train: {ytrain.value_counts().to_dict()}")

# Save datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="satyabrd123/Wellness-Tourism-Dataset",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face")

print("\nData preparation complete!")
