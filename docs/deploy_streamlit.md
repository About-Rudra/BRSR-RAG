# Getting a live link (Streamlit Community Cloud)

The hosted app runs `streamlit_app.py` from GitHub. It cannot see your laptop's `vector_db/`
folder, and `data/`/`vector_db/` are not committed, so the index lives in a Qdrant Cloud
collection that the app reads from.

## 1. Put the index in Qdrant Cloud (one time, and again after re-ingesting)

1. At <https://cloud.qdrant.io>, resume your free cluster (free clusters are suspended when they
   are unused), or create a new free cluster. Copy its URL (ending in `:6333`) and an API key.
2. In `.env`:
   ```
   QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333
   QDRANT_API_KEY=<key>
   ```
3. Build the local index if you have not already (`python main.py --ingest`), then copy it:
   ```
   python -m scripts.copy_index_to_cloud            # add --replace to overwrite an existing copy
   ```
   This copies the stored vectors, so nothing is re-embedded.

## 2. Push the code

```
git push origin main
```

## 3. Create the app

1. Go to <https://share.streamlit.io>, sign in with GitHub, and choose **Create app → Deploy a public app from GitHub**.
2. Repository `About-Rudra/BRSR-RAG`, branch `main`, main file `streamlit_app.py`.
3. **Advanced settings**: Python 3.12. Paste the secrets below:
   ```toml
   LLM_PROVIDER = "gemini"
   LLM_MODEL = "gemini-3.6-flash"
   GOOGLE_API_KEY = "<your key>"
   LLM_RPM = "4"          # free tier allows 5 requests/min per model; raise it on a paid key
   QDRANT_MODE = "cloud"
   QDRANT_URL = "https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "<key>"
   APP_PASSWORD = "<optional: anyone with the link needs this>"
   ```
4. Deploy. The first build takes a few minutes (torch + the embedding model).
   The link looks like `https://<name>.streamlit.app`.

On the Gemini free tier one agent question makes roughly 10-20 LLM calls, so at 4 calls/min it
takes a few minutes; the app spaces the calls out instead of failing. A paid key or
`LLM_PROVIDER = "groq"` (higher free limits) makes it much faster.

`APP_PASSWORD` is optional. Without it, anyone with the link can run queries on your API quota.

## Running the same UI locally or in Colab

```
streamlit run streamlit_app.py
```
Locally it uses the embedded `vector_db/` (`QDRANT_MODE=local`) and your `.env`.
