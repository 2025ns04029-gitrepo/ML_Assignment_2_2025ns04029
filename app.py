from pathlib import Path
from model import train_models

if __name__ == "__main__":
    train_models.load_ml_imports()
    train_models.run_streamlit_app(Path(__file__).resolve().parent)
