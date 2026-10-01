# Support Ticket Classification
# Complete NLP classification project.
# Install:
# pip install pandas numpy scikit-learn matplotlib openpyxl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("support_tickets.csv")

# -----------------------------
# Category classification
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["Ticket_Text"], df["Category"],
    test_size=0.20, random_state=42, stratify=df["Category"]
)

category_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=2000))
])

category_model.fit(X_train, y_train)
pred = category_model.predict(X_test)

print("CATEGORY ACCURACY:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))

# -----------------------------
# Priority classification
# -----------------------------
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    df["Ticket_Text"], df["Priority"],
    test_size=0.20, random_state=42, stratify=df["Priority"]
)

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

priority_model.fit(Xp_train, yp_train)
priority_pred = priority_model.predict(Xp_test)

print("PRIORITY ACCURACY:", round(accuracy_score(yp_test, priority_pred), 4))
print(classification_report(yp_test, priority_pred))

# Predict a new ticket
new_ticket = ["Urgent! My account was hacked and I cannot log in."]
print("Predicted category:", category_model.predict(new_ticket)[0])
print("Predicted priority:", priority_model.predict(new_ticket)[0])
