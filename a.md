Since you are using `uv` and a `pyproject.toml`, your README should reflect the modern Python workflow. `uv` replaces both `pip` and `venv` management, making the setup much cleaner.

### 📝 What to write in your README.md

Replace the old `pip` instructions with these. It’s best to provide the "one-step" command and the "manual" sync option:

#### Option 1: The "One-Command" (Recommended)

This is the cleanest way. `uv` will automatically create a virtual environment, install dependencies from your `pyproject.toml`, and launch the app in one go.

---

### 🙈 What about `.gitignore`?

When using `uv`, you have a few specific files and folders you should ignore to keep your repository clean. Here is a standard `.gitignore` block for your project:

```text
# uv / Python environments
.venv/
__pycache__/
*.py[cod]

# Streamlit
.streamlit/config.toml
! .streamlit/  # Keep the folder, but ignore specific local configs if needed

# OS specific
.DS_Store
Thumbs.db

# Data (Optional)
# Ignore local CSVs if they contain sensitive data or are very large
*.csv
```

### 💡 A Note on `uv.lock`

**Do not ignore `uv.lock`.** Unlike a `requirements.txt`, the `uv.lock` file ensures that every person who clones your repo gets the **exact** same versions of every sub-dependency. You should commit both `pyproject.toml` and `uv.lock` to Git.

### Pro-Tip for your README

Since your app has that cool "dark sci-fi" CSS, I'd highly recommend adding a **"Prerequisites"** section mentioning that `uv` is required, as it's significantly faster than standard `pip`.

Would you like me to help you draft a "Features" section for the Markdown that highlights the Naive Bayes logic you implemented?
