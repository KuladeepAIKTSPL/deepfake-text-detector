import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ✅ Ensure data file exists
DATA_PATH = "data/ai_human_text.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found! Please add dataset file.")

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)

# ✅ Expect columns: 'text' and 'label'
# label: 0 = human, 1 = AI-generated
X = df['text']
y = df['label']

# ✅ Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# ✅ Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("Model trained successfully ✅")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Files saved: model.pkl and vectorizer.pkl")
