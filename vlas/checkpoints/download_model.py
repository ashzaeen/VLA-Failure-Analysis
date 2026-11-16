from huggingface_hub import snapshot_download
import argparse
import os


def download_model(repo_id, target_dir=None):
    if target_dir is None:
        target_dir = os.getcwd()

    target_dir = os.path.abspath(target_dir)

    # Extract model name after last slash
    # e.g. "someuser/model-name" -> "model-name"
    model_name = repo_id.split("/")[-1]

    # Final download path
    target_path = os.path.join(target_dir, model_name)

    print(f"Downloading model '{repo_id}' into folder '{target_path}'...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_path,
        local_dir_use_symlinks=False
    )

    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model into a target folder."
    )

    parser.add_argument(
        "repo_id",
        type=str,
        help="Model repo ID, e.g. 'bert-base-uncased' or 'user/model-name'"
    )

    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Target directory to save the model. Default: current working directory."
    )

    args = parser.parse_args()

    download_model(args.repo_id, args.target_dir)