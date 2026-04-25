# Data

Persistent data assets for the NWP server.

## Files

| File | Purpose |
|---|---|
| `seed_corpus.txt` | Seed phrases pre-loaded into every new n-gram model on first startup. One sentence per line; lines starting with `#` are comments. |
| `lstm_weights_placeholder.json` | Placeholder that documents the expected LSTM weights file format. Replace with a real export from `NWP_Training_Notebook.ipynb`. |

## Runtime data (generated at runtime, not tracked in version control)

| Path | Purpose |
|---|---|
| `trainer/` | SQLite database (`trainer.sqlite3`), per-model hybrid state (`hybrid_<id>.json`), LSTM weights (`lstm_<id>.json`), personalization data (`pers_<id>.json`), and device ID (`device_id.txt`). |

## Loading external LSTM weights

1. Train the model using `NWP_Training_Notebook.ipynb`.
2. Export weights using the notebook's export cell (produces a `.json` file).
3. Either:
   - Upload via the **DL Tuning** tab in the web UI (drag-and-drop or file picker), or
   - `POST /local/model/upload_weights` with `model_id` form field and the `.json` file.
