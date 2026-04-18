import pandas as pd
import numpy as np

import re
import string

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_PATH = "good_rows_unique.csv"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head()


df_task = df[["Description", "TimeOfDay"]].copy()   #Keep only needed columns

# to drop rows with missing values in Description or TimeOfDay
before = len(df_task)
df_task = df_task.dropna(subset=["Description", "TimeOfDay"])
after = len(df_task)

print(f"Dropped {before - after} rows due to missing Description/TimeOfDay.")
print("Remaining:", len(df_task))

df_task.head()


def clean_label(x: str):
    x = str(x).strip().lower()

    allowed = {
        "morning": "Morning",
        "afternoon": "Afternoon",
        "evening": "Evening"
    }

    return allowed.get(x, None)

df_task["TimeOfDay"] = df_task["TimeOfDay"].apply(clean_label)

before = len(df_task)
df_task = df_task.dropna(subset=["TimeOfDay"])
after = len(df_task)

print(f"Removed {before - after} rows with invalid TimeOfDay labels")
print("Remaining labels:", df_task["TimeOfDay"].unique())



def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation (,)
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

df_task["Description_clean"] = df_task["Description"].apply(clean_text)
df_task[["Description", "Description_clean", "TimeOfDay"]].head()

#save cleaned dataset (required submission)
CLEAN_PATH = "cleaned_timeofday_dataset.csv"
df_task.to_csv(CLEAN_PATH, index=False)
print("Saved cleaned dataset to:", CLEAN_PATH)

counts = df_task["TimeOfDay"].value_counts()

plt.figure()
counts.plot(kind="bar")
plt.title("Class Distribution: TimeOfDay")
plt.xlabel("TimeOfDay")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

#Split the data
X = df_task["Description_clean"]
y = df_task["TimeOfDay"]

#split Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=7,
    stratify=y
)

#split Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=7,
    stratify=y_temp
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot()
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

    return acc, y_pred

knn_results = []

for k in [1, 3]:
    knn_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine"))
    ])

    knn_pipe.fit(X_train, y_train)
    acc, _ = evaluate_model(f"kNN Baseline (k={k})", knn_pipe, X_test, y_test)
    knn_results.append((f"kNN k={k}", acc))

#Logistic Regression
lr_results = []

for C in [0.01, 0.1, 1, 10]: #C is hyperparameter that is inverse of lambda that is the penalty of parameter that do regularization
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(C=C, max_iter=5000))
    ])

    lr_pipe.fit(X_train, y_train)
    val_pred = lr_pipe.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    lr_results.append((C, val_acc))

best_C_lr = max(lr_results, key=lambda x: x[1])[0]
print("Best LR C:", best_C_lr)

best_lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
    ("lr", LogisticRegression(C=best_C_lr, max_iter=5000))
])

best_lr.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
lr_test_acc, lr_pred = evaluate_model("Logistic Regression (best)", best_lr, X_test, y_test)


#SVM Grid Search (4 values)
svm_results = []
#C here is hyperparameter which controls the balance between maximizing the margin and minimizing classification errors
for C in [0.01,0.1, 1, 10]:
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("svm", LinearSVC(C=C,max_iter=20000))
    ])

    svm_pipe.fit(X_train, y_train)
    val_pred = svm_pipe.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    svm_results.append((C, val_acc))


#Evaluate best SVM on test
best_C_svm = max(svm_results, key=lambda x: x[1])[0]
print("Best SVM C:", best_C_svm)

best_svm = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
    ("svm", LinearSVC(C=best_C_svm))
])

best_svm.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
svm_test_acc, svm_pred = evaluate_model("Linear SVM (best)", best_svm, X_test, y_test)



#Compare all models in one table
summary = pd.DataFrame({
    "Model": [r[0] for r in knn_results] + ["Logistic Regression (best)", "Linear SVM (best)"],
    "Test Accuracy": [r[1] for r in knn_results] + [lr_test_acc, svm_test_acc]
}).sort_values("Test Accuracy", ascending=False)


#Find best model automatically
best_model_name = summary.iloc[0]["Model"]
print("Best model based on test accuracy:", best_model_name)


#Collect misclassified examples (for best model)
# Decide which trained model is best
if "kNN k=1" in best_model_name:
    final_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("knn", KNeighborsClassifier(n_neighbors=1, metric="cosine"))
    ]).fit(X_train, y_train)
elif "kNN k=3" in best_model_name:
    final_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("knn", KNeighborsClassifier(n_neighbors=3, metric="cosine"))
    ]).fit(X_train, y_train)
elif "Logistic" in best_model_name:
    final_model = best_lr
else:
    final_model = best_svm

y_pred_final = final_model.predict(X_test)

errors = pd.DataFrame({
    "Description": X_test.values,
    "True": y_test.values,
    "Pred": y_pred_final
})

errors = errors[errors["True"] != errors["Pred"]]
print("Number of errors:", len(errors))
errors.head(10)


#Error patterns (which classes get confused)
conf_pairs = errors.groupby(["True", "Pred"]).size().sort_values(ascending=False)
print(conf_pairs)

#Show examples for the top confusion type
if len(conf_pairs) > 0:
    top_pair = conf_pairs.index[0]
    t, p = top_pair
    print("Most common confusion:", (t, "->", p))
    sample = errors[(errors["True"] == t) & (errors["Pred"] == p)].head(10)
    print(sample)
else:
    print("No errors found (perfect accuracy).")
