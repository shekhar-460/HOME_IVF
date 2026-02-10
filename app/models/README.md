# Models Directory

This directory stores downloaded ML models for local use.

## Medgemma Model

The medgemma-4b-it model is stored here when downloaded using the download script.

### Download the Model

To download the medgemma model to this directory:

```bash
python scripts/download_medgemma.py
```

This will:
- Download the model from HuggingFace
- Save it to `app/models/medgemma-4b-it/`
- Make it available for offline use

### Model Structure

After downloading, the directory structure will be:
```
app/models/
├── medgemma-4b-it/
│   ├── config.json
│   ├── generation_config.json
│   ├── model files...
│   └── tokenizer files...
└── README.md
```

### Usage

The application will automatically:
1. Check for local model in `app/models/medgemma-4b-it/` if `USE_LOCAL_MEDGEMMA=True`
2. Fall back to HuggingFace download if local model not found
3. Use the model for generating responses when FAQs don't have good matches

### Configuration

Set these in your `.env` file or environment:

```env
USE_MEDGEMMA=True
USE_LOCAL_MEDGEMMA=True
MEDGEMMA_MODEL_PATH=app/models/medgemma-4b-it
```

### Notes

- Model size: ~8GB (downloads automatically)
- First download may take 10-30 minutes depending on internet speed
- Model files are gitignored (too large for git)
- Model is cached locally after first download
