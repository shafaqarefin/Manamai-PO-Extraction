# 📦 Purchase Order PDF Extraction

A Python project for extracting and normalizing structured data from purchase-order PDFs.

---

## 🚀 Features
- Parse PDF tables using **Camelot** (`flavor="stream"`).
- Normalize delivery dates to `YYYY-MM-DD`.
- Extract fields such as **Country**, **Planning Markets**, **Invoice Average Price**, etc.
- Output clean, analysis-ready Excel/CSV data.

---

## 🛠 Tech Stack
- **Python 3.11+**
- **uv** – a modern Python package manager (similar to `npm` for JavaScript).

Key libraries (all declared in `pyproject.toml`):
- `camelot-py`
- `pandas`
- `openpyxl`
- `python-dateutil`
- any other project-specific packages you added.

---

## ⚙️ Full Setup Instructions

Follow these steps exactly to get the project running from scratch.

### 1. Install uv
`uv` is like `npm` for Python: it manages virtual environments and installs dependencies from `pyproject.toml`.

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```
**Windows Inside Powershell with admin pass**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2.Install Packages
**In your current directory clone project**
```bash
git clone https://github.com/shafaqarefin/pdfExtractionandExcel.git
```
**Run this command to install dependencies**
```bash
uv pip install -r pyproject.toml
```
**Run command**
```bash
uv run -m src.main
```
**Run command**
*Check newly created output folder which contains all excel files saved using the first id of each pdf in data folder*





