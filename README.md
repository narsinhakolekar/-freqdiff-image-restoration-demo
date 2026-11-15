# FreqDiff — Interactive Image Restoration Research Demo

A Streamlit research interface for the FreqDiff conditional diffusion image-restoration project. It supports uploaded-image inference with the frequency-aware and spatial-only V2 models, a controlled degradation demo, PSNR/SSIM when a clean reference is available, and the stored V2 held-out benchmark analysis.

## Deployment target

This repository is configured for Streamlit Community Cloud using CPU-only PyTorch. Choose **Python 3.12** in the deployment's Advanced settings. PyTorch CPU wheels are installed from the official PyTorch CPU package index; GPU acceleration is not assumed.

## Files required in the repository

```text
.
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── diffusion_model.py
│       ├── spatial_conditional_unet.py
│       └── diffusion_process.py
├── checkpoints/v2_corrected/
│   ├── frequency_best.pth
│   └── spatial_best.pth
└── results/v2_corrected/
    ├── frequency_training_history.json
    ├── spatial_training_history.json
    └── evaluation/
        ├── summary.csv
        ├── per_image_metrics.csv
        ├── paired_bootstrap.csv
        └── evaluation_protocol.json
```

The two best V2 checkpoints and listed result files are created in your local FreqDiff project. The `.gitignore` is deliberately configured to exclude the DIV2K dataset, other checkpoints, intermediate outputs, credentials, and local environments while allowing only the inference checkpoints and benchmark files listed above.

## Important: what not to upload

Do **not** commit `data/`, DIV2K image files, raw datasets, training runs, optimizer backups, `.env` files, or `.streamlit/secrets.toml`. The live app only needs the model source modules, the two best V2 checkpoint files, and the stored benchmark/history files. The dataset itself is not required for uploading a new image and running inference.

The included `requirements.txt` is for the cloud deployment only. Do not use it to replace your local CUDA-enabled PyTorch installation. Continue using `requirements_streamlit.txt` in your local CUDA environment.

## Local validation before pushing

From the FreqDiff project root:

```cmd
conda activate freqdiff
python -m py_compile app.py src\models\diffusion_model.py src\models\spatial_conditional_unet.py src\models\diffusion_process.py
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
streamlit run app.py
```

Verify both V2 checkpoint files and all V2 history/evaluation CSV/JSON files exist before pushing. If a checkpoint file exceeds GitHub's 100 MB per-file limit, do not force it into Git; use Git LFS or external model hosting instead.

## Push to GitHub (after creating an empty repository)

Run these from your local FreqDiff project root. First inspect the staged file list; never proceed if `data/`, extra checkpoints, or secret files appear.

```cmd
git init -b main
git add app.py requirements.txt README.md .gitignore .streamlit\config.toml
git add src\__init__.py src\models\__init__.py src\models\diffusion_model.py src\models\spatial_conditional_unet.py src\models\diffusion_process.py
git add checkpoints\v2_corrected\frequency_best.pth checkpoints\v2_corrected\spatial_best.pth
git add results\v2_corrected\frequency_training_history.json results\v2_corrected\spatial_training_history.json
git add results\v2_corrected\evaluation\summary.csv results\v2_corrected\evaluation\per_image_metrics.csv results\v2_corrected\evaluation\paired_bootstrap.csv results\v2_corrected\evaluation\evaluation_protocol.json
git status --short
```

After checking that the staged list contains only the intended files:

```cmd
git commit -m "Add FreqDiff Streamlit research demo"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the project already has a Git repository or an `origin` remote, do not repeat `git init` or `git remote add`; inspect `git status` and `git remote -v` first.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ using GitHub.
2. Choose **Create app** → **Yup, I have an app**.
3. Select the repository, `main` branch, and `app.py` as the entrypoint.
4. Open Advanced settings and select Python 3.12.
5. Deploy. The app will receive a `*.streamlit.app` URL.

For private repositories, first grant Streamlit Community Cloud access to the private repository through its linked-account settings. Public repositories are simpler to connect. Repository admins can deploy apps from the repository.

## Runtime expectation

Streamlit Community Cloud resources are shared and limited. CPU inference will be slower than the local RTX 3050 run. Keep upload dimensions modest and use fewer DDIM steps for public interactive demos. The benchmark charts remain stored experiment results and do not require running the full held-out evaluation during app startup.

## Scientific interpretation

The current V2 held-out benchmark uses 80 source images and 720 deterministic 128×128 patches, with patch metrics aggregated within each source image. It uses synthetic Gaussian blur/noise. The spatial-only V2 model outperformed the frequency-aware V2 model on both PSNR and SSIM in this experiment. Five training epochs were compute-constrained; do not claim convergence or real-world validation.
