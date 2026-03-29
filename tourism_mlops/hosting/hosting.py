from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_mlops/deployment",     # the local folder containing your files
    repo_id="satyabrd123/Wellness-Tourism-Predictor",          # the target repo (space name)
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
