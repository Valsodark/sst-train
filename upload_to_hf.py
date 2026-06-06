from huggingface_hub import upload_folder

# Uploads the current folder to the Space, excluding the 40 GB data/ folder,
# the virtual environment, and other junk. data_subset/ IS included.
upload_folder(
    repo_id="Trimuerto/stta_app",
    repo_type="space",
    folder_path=".",
    ignore_patterns=[
        "data/*",          # the full 40 GB dataset (data_subset/ is kept)
        ".venv/*",
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        "*.7z",
        "*.npy",
        "upload_to_hf.py",
    ],
    commit_message="Deploy app + 30-day data subset",
)
print("UPLOAD COMPLETE")
