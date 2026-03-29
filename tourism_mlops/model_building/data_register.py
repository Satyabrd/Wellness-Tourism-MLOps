from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "satyabrd123/Wellness-Tourism-Dataset"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repository '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_mlops/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
