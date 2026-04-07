# 🧠 Naive Bayes Predictor

A Streamlit app that lets you upload any CSV, train a Naive Bayes classifier, and get instant predictions via interactive checkboxes/radio buttons.

## Features

- 📂 Upload any CSV with Yes/No or categorical columns
- ⚙️ Choose which columns are features and which is the target
- 🤖 Auto-selects BernoulliNB (binary) or CategoricalNB based on data
- 🎛️ Interactive radio buttons / dropdowns for each feature
- 📊 Shows accuracy, confusion matrix, classification report
- 🔮 Instant prediction with probability bars
- 🔍 Naive Bayes reasoning explanation

### 🚀 Quick Start

Ensure you have [uv](https://github.com/astral-sh/uv) installed, then run:

```bash
uv run streamlit run app.py
```

#### Option 2: The "Step-by-Step"

If you want users to have a persistent virtual environment they can see:

### 🛠️ Setup & Run

1. **Install dependencies and create virtual env:**

   ```bash
   uv sync

   ```

2. **Run the application:**
   ```bash
   uv run streamlit run app.py
   ```

## Example CSV Format

```
Cough,Fever,Headache,Flu
Yes,Yes,Yes,Yes
Yes,No,Yes,Yes
No,Yes,No,No
No,No,No,No
```

The **last column** is usually the target (what you want to predict), but you can choose any column in the app.
