import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 🧾 Sample training data (replace this with your dataset later)
data = {
    "text": [
        "Breaking: AI-generated fake article spreads misinformation",
        "Scientists discover new treatment for diabetes",
        "AI writes completely fake political news",
        "Government releases real economic report",
        "ChatGPT-generated fake blog post goes viral",
        "Local students win national science competition",
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Fake, 0 = Real
}

df = pd.DataFrame(data)

# 🧩 Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ✨ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🤖 Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 📊 Evaluate (optional)
y_pred = model.predict(X_test_tfidf)
print("Model trained successfully ✅")
print(classification_report(y_test, y_pred))

# 💾 Save model and vectorizer together
joblib.dump((model, vectorizer), "model.pkl")
print("Saved trained model as model.pkl 🎯")
