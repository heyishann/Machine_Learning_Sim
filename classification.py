import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def classify_models(df):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        classifiers = {
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=50),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
        }

        results = []

        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": round(accuracy*100,2) })

        return {"results": results, "message": "Classification completed!"}

    except Exception as e:
        return {"error": str(e)}

def classify_ensemble(df,selected_models,voting_type):
    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        available_classifiers = {
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(n_estimators=100),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "Logistic Regression": LogisticRegression(max_iter=50),
                    "KNN": KNeighborsClassifier(),
                    "Support Vector Machine": SVC(probability=True)
                }

        estimators = [(name, available_classifiers[name]) for name in selected_models]
        ensemble_model = VotingClassifier(estimators=estimators, voting=voting_type)
        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        results = [{"Model": "Voting Classifier", "Type": voting_type,"Accuracy": round(accuracy*100,2)}]

        return {"results": results, "message": "Ensemble Classification completed!"}

    except Exception as e:
        return {"error": str(e)}
