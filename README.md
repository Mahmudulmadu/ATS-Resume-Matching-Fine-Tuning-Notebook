# ATS Resume Embedding Match Score Fine-Tuning

This project fine-tunes a sentence embedding model for ATS-style resume to job-description matching.

The main notebook is [ATS_RESUME_EMBEDDING_MATCH_SCORE_FINE_TUNE.ipynb](ATS_RESUME_EMBEDDING_MATCH_SCORE_FINE_TUNE.ipynb).

## Dataset

This project uses the following dataset from Hugging Face:

- Dataset: `0xnbk/resume-ats-score-v1-en`
- Link: https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en

## Project Goal

The goal of this project is to train a better semantic similarity model for matching:

- `resume_text`
- `job_description`

The original dataset stores both texts together in a single `text` field separated by `SEP`.

Example:

`resume text SEP job description`

In this notebook, that combined field is split into:

- `resume_text`
- `job_description`

This allows the model to learn resume-job alignment more effectively.

## What This Notebook Does

The notebook covers the full workflow:

1. Load the dataset from Google Drive in Google Colab.
2. Inspect and validate the raw data.
3. Split the combined `text` column using `SEP`.
4. Clean and normalize the dataset.
5. Perform EDA with visualizations.
6. Compare multiple embedding backbones.
7. Fine-tune the best model for ATS similarity scoring.
8. Evaluate the fine-tuned model on validation and test data.
9. Save the trained model, predictions, and metadata.
10. Create a reusable inference function for new resume-job pairs.

## Candidate Embedding Models

The notebook compares these sentence embedding backbones before fine-tuning:

- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-base-v2`

The best backbone is selected automatically based on validation metrics.

## Training Objective

The project uses:

- `SentenceTransformer`
- `CosineSimilarityLoss`

Each training example contains:

- text 1: resume
- text 2: job description
- label: normalized ATS score from `0` to `1`

The model is trained so that semantically aligned resume-job pairs produce higher cosine similarity.

## Evaluation Metrics

The notebook evaluates both the base model and the fine-tuned model using:

- MAE
- RMSE
- Pearson correlation
- Spearman correlation

It also includes:

- predicted vs actual scatter plot
- error distribution plots
- prediction-by-label plots
- manual business-style test cases

## Outputs

After training, the notebook saves:

- fine-tuned model directory
- `test_predictions.csv`
- `metrics.json`

Typical save path in Colab:

`/content/drive/MyDrive/ATS_resume_matching_models/`

## Inference

The notebook defines a reusable function:

`score_resume_against_jd(resume_text, job_description)`

This function returns a similarity score between `0` and `1` for a new resume and job-description pair.

## Requirements

This notebook is designed for:

- Google Colab
- T4 GPU recommended

Main package:

- `sentence-transformers`

Install command used in the notebook:

```python
!pip -q install -U sentence-transformers accelerate
```

## How To Run
- Open the notebook in Google Colab.
- Mount Google Drive.
- Load the train and validation CSV files.
- Run the notebook from top to bottom.
- Review the evaluation metrics and plots.
- Use the saved model for inference on new resume-job pairs.
## Notes
The raw dataset may contain both resume and job description in the same text field.
This project explicitly separates them before training.
Fine-tuning is performed only after comparing strong baseline embedding backbones.
The notebook is written as both a training workflow and a learning guide.
## Dataset Credit
Dataset source:

https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en
