import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======== Konfigurasi Path ========
DATA_PATH = "dataset/aug_train.csv"
MODEL_PATH = "recruitment_model.joblib"  # Simpan langsung di root folder

def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Hapus kolom ID karena tidak digunakan
    df.drop("enrollee_id", axis=1, inplace=True)

    # Isi missing value
    df['education_level'].fillna("Unknown", inplace=True)
    df['major_discipline'].fillna("Unknown", inplace=True)
    df['experience'].fillna("0", inplace=True)
    df['company_type'].fillna("Unknown", inplace=True)
    df['company_size'].fillna("Unknown", inplace=True)
    df['last_new_job'].fillna("0", inplace=True)

    # Label encoding semua kolom kategori
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Pisahkan fitur dan target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Tampilkan fitur yang digunakan (debug)
    print("✅ Fitur yang digunakan untuk training:")
    print(X.columns.tolist())

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model_with_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print(f"✅ Model terbaik: {grid_search.best_params_}")

    # Simpan model terbaik ke root folder
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    print(f"📦 Model terbaik berhasil disimpan di: {MODEL_PATH}")

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Akurasi pada data uji: {acc:.4f}")

if __name__ == "__main__":
    print("🚀 Memuat dan memproses data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("🔧 Melatih model dengan GridSearchCV...")
    model = train_model_with_grid_search(X_train, y_train)

    print("📊 Evaluasi model...")
    evaluate_model(model, X_test, y_test)
    
