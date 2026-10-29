# turbopuffer playground: security questionnaire assistant

A hobby Streamlit app for exploring turbopuffer: hybrid search over per-customer answer libraries, questionnaire auto-fill with Claude, real HashiCorp Vault docs, and hundreds of tenant namespaces. All company names and answers in the sample data are fictional.

## Deploy on Streamlit Community Cloud (free)

1. Create a new GitHub repository (private is fine) and upload `app.py`, `requirements.txt` and this `README.md`.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app** → **Deploy a public app from GitHub**.
3. Pick the repository, branch `main`, main file `app.py`. Choose an app URL.
4. Open **Advanced settings**, pick Python 3.12, and paste the secrets below.
5. Click **Deploy**. The first build takes two or three minutes.

```toml
TURBOPUFFER_API_KEY = "tpuf_..."
APP_PASSWORD = "choose-a-password"
# Optional:
# ANTHROPIC_API_KEY = "sk-ant-..."   # Claude-drafted answers in Auto-fill
# GITHUB_TOKEN = "github_pat_..."    # only if GitHub rate-limits the Vault docs loader
```

To change secrets later: your app → **⋮** → **Settings** → **Secrets**. The app restarts on its own.

## Tips

- Free apps sleep after a period without visitors, and waking takes about 30 seconds.
- Data stays in turbopuffer between visits (namespaces start with `tpdemo-`), so you only need **Load sample libraries** once.
- **Reset everything** on the first tab deletes everything the app created.

## Run locally

```bash
pip install -r requirements.txt
export TURBOPUFFER_API_KEY=tpuf_...
streamlit run app.py
```
