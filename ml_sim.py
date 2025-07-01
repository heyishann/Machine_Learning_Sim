import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier

def read_file(path):
    """Reads a CSV/TSV/space-separated file intelligently."""
    sep = ","
    df = pd.read_csv(path)
    
    if df.shape[1] == 1:  # Try different separators
        for s in ["\t", " "]:
            sep = s
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                break
    
    try:
        float(df.columns[0])  # Check if first column is numeric
        df = pd.read_csv(path, sep=sep, header=None)
    except ValueError:
        pass
    
    return df

def upload_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV and Text Files", "*.csv;*.txt")])
    if file_path:
        global dataset
        dataset = read_file(file_path)
        messagebox.showinfo("Success", "Dataset uploaded successfully!")
        data_name = file_path.split("/")[-1]

      
        # Dataset Section
        dataset_label = tk.Label(frame, text=f"Dataset: {data_name},\n Shape = {dataset.shape}", font=("Arial", 12, "bold"))
        dataset_label.grid(row=3, column=0, pady=10, sticky="w")

        for widget in dataset_frame.winfo_children():  # Clear previous results
            widget.destroy()
        
        if dataset.shape[1]>8:
            headers = dataset.columns[:8]
            for col, header in enumerate(headers):
                label = tk.Label(dataset_frame, text=header, font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=5, pady=5)
                label.grid(row=0, column=col, sticky="news")
            
            for row, result in enumerate(np.array(dataset.head())[:,:8], start=1):
                for col, value in enumerate(result):
                    label = tk.Label(dataset_frame, text=round(value,3), font=("Arial", 10), borderwidth=1, relief="solid", padx=5, pady=5)
                    label.grid(row=row, column=col, sticky="news")
        else:
            s = dataset.shape[1]
            headers = dataset.columns[:s]
            for col, header in enumerate(headers):
                label = tk.Label(dataset_frame, text=header, font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=5, pady=5)
                label.grid(row=0, column=col, sticky="news")
            
            for row, result in enumerate(np.array(dataset.head())[:,:s], start=1):
                for col, value in enumerate(result):
                    label = tk.Label(dataset_frame, text=round(value,3), font=("Arial", 10), borderwidth=1, relief="solid", padx=5, pady=5)
                    label.grid(row=row, column=col, sticky="news")


def show_results_table(results):
    
    """Function to dynamically create table cells in Tkinter"""
    for widget in result_frame.winfo_children():  # Clear previous results
        widget.destroy()
    result_label = tk.Label(frame, text="Results:", font=("Arial", 12, "bold"))
    result_label.grid(row=3, column=1, pady=10,padx=10, sticky="w")
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-score"]
    for col, header in enumerate(headers):
        label = tk.Label(result_frame, text=header, font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=5, pady=5)
        label.grid(row=8, column=col, sticky="news")
    
    for row, result in enumerate(results, start=1):
        for col, value in enumerate(result):
            label = tk.Label(result_frame, text=value, font=("Arial", 10), borderwidth=1, relief="solid", padx=5, pady=5)
            label.grid(row=row+8, column=col, sticky="news")

def run_normal_classifier():
    if dataset is None:
        messagebox.showerror("Error", "Please upload a dataset first.")
        return
    
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append([name, f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"])
    
    show_results_table(results)

# Function to dynamically update UI for parameters
def update_parameters():
    for widget in param_frame.winfo_children():
        widget.destroy()

    ensemble_type = ensemble_var.get()

    if ensemble_type == "Voting":
        tk.Label(param_frame, text="Select Classifiers:").pack()
        global voting_classifiers, voting_type_var
        voting_classifiers = {
            "Logistic Regression": tk.BooleanVar(),
            "Decision Tree": tk.BooleanVar(),
            "SVM": tk.BooleanVar(),
            "KNN": tk.BooleanVar()
        }
        for clf, var in voting_classifiers.items():
            tk.Checkbutton(param_frame, text=clf, variable=var).pack(anchor="w")

        tk.Label(param_frame, text="Voting Type:").pack()
        voting_type_var = tk.StringVar(value="soft")
        ttk.Combobox(param_frame, textvariable=voting_type_var, values=["hard", "soft"]).pack()

    elif ensemble_type == "Bagging":
        tk.Label(param_frame, text="Number of Estimators:").pack()
        global n_estimators_bagging
        n_estimators_bagging = tk.Entry(param_frame)
        n_estimators_bagging.insert(0, "10")
        n_estimators_bagging.pack()

        tk.Label(param_frame, text="Classifier:").pack()
        global bagging_classifier_var
        bagging_classifier_var = tk.StringVar(value="Decision Tree")
        ttk.Combobox(param_frame, textvariable=bagging_classifier_var, values=["Decision Tree", "Logistic Regression", "SVM", "KNN"]).pack()

    elif ensemble_type == "Boosting":
        tk.Label(param_frame, text="Select Boosting Technique:").pack()
        global boosting_technique_var
        boosting_technique_var = tk.StringVar(value="AdaBoost")
        ttk.Combobox(param_frame, textvariable=boosting_technique_var, values=[
            "AdaBoost",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
            "CatBoost"
        ]).pack()

        tk.Label(param_frame, text="Number of Estimators:").pack()
        global n_estimators_boost, learning_rate_boost
        n_estimators_boost = tk.Entry(param_frame)
        n_estimators_boost.insert(0, "50")
        n_estimators_boost.pack()

        tk.Label(param_frame, text="Learning Rate:").pack()
        learning_rate_boost = tk.Entry(param_frame)
        learning_rate_boost.insert(0, "1.0")
        learning_rate_boost.pack()

    elif ensemble_type == "Stacking":
        tk.Label(param_frame, text="Select Initial Estimator:").pack()
        
        global initial_estimator_var
        initial_estimator_var = tk.StringVar(value="Logistic Regression")
        
        estimators = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes"]
        
        for clf in estimators:
            tk.Radiobutton(param_frame, text=clf, variable=initial_estimator_var, value=clf).pack(anchor="w")
        
        tk.Label(param_frame, text="Final Estimator:").pack()
        global final_estimator_var
        final_estimator_var = tk.StringVar(value="SVM")
        ttk.Combobox(param_frame, textvariable=final_estimator_var, values=["SVM", "Logistic Regression", "KNN"]).pack()

# Function to run classification
def run_ensemble_classifier():
    if dataset is None:
        messagebox.showerror("Error", "Please upload a dataset first.")
        return

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ensemble_type = ensemble_var.get()
    model = None
    
    if ensemble_type == "Voting":
        selected_classifiers = []
        if voting_classifiers["Logistic Regression"].get():
            selected_classifiers.append(('lr', LogisticRegression()))
        if voting_classifiers["Decision Tree"].get():
            selected_classifiers.append(('dt', DecisionTreeClassifier()))
        if voting_classifiers["SVM"].get():
            selected_classifiers.append(('svc', SVC(probability=True)))
        if voting_classifiers["KNN"].get():
            selected_classifiers.append(('knn', KNeighborsClassifier()))
        
        voting_type = voting_type_var.get()
        model = VotingClassifier(estimators=selected_classifiers, voting=voting_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')


    elif ensemble_type == "Bagging":
        n = int(n_estimators_bagging.get())
        
        base_estimator = None
        if bagging_classifier_var.get() == "Decision Tree":
            base_estimator = DecisionTreeClassifier()
        elif bagging_classifier_var.get() == "Logistic Regression":
            base_estimator = LogisticRegression()
        elif bagging_classifier_var.get() == "SVM":
            base_estimator = SVC()
        elif bagging_classifier_var.get() == "KNN":
            base_estimator = KNeighborsClassifier()
        
        # Create Bagging model
        model = BaggingClassifier(estimator=base_estimator, n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')


    elif ensemble_type == "Boosting":
        boosting_technique = boosting_technique_var.get()
        n_estimators = int(n_estimators_boost.get())
        learning_rate = float(learning_rate_boost.get())
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        if boosting_technique == "AdaBoost":
            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        elif boosting_technique == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        elif boosting_technique == "XGBoost":
            model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss')
        elif boosting_technique == "LightGBM":
            model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        elif boosting_technique == "CatBoost":
            model = CatBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, verbose=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

    # Display results
    # show_results_table([[f'Boosting ({boosting_technique})', f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]])


    elif ensemble_type == "Stacking":
        selected_estimators = []
        
        selected = initial_estimator_var.get()
        
        if selected == "Logistic Regression":
            selected_estimators.append(('lr', LogisticRegression()))
        elif selected == "Decision Tree":
            selected_estimators.append(('dt', DecisionTreeClassifier()))
        elif selected == "KNN":
            selected_estimators.append(('knn', KNeighborsClassifier()))
        elif selected == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB
            selected_estimators.append(('nb', GaussianNB()))
        
        final_estimator_choice = final_estimator_var.get()
        if final_estimator_choice == "SVM":
            final_estimator = SVC()
        elif final_estimator_choice == "Logistic Regression":
            final_estimator = LogisticRegression(max_iter=1000)
        else:
            final_estimator = KNeighborsClassifier()
        
        model = StackingClassifier(estimators=selected_estimators, final_estimator=final_estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    
    show_results_table([[f'Ensemble: {ensemble_type}', f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]])



# UI Setup

root = tk.Tk()
root.title("ML Simulator")



root.geometry("1200x800")  # Adjusted window size
root.configure(bg="lightblue")


# Main Frame (Now Using Grid Instead of Pack)
frame = tk.Frame(root, padx=10, pady=10, bg='lightblue')
frame.grid(row=0, column=0, padx=20, pady=20, sticky="nw")




# Title
title_label = tk.Label(frame, text="Machine Learning Simulator", font=("Arial", 60, "bold"))
title_label.grid(row=0, column=0, padx= 50, columnspan=20, pady=10)

# Upload Dataset Button
upload_btn = tk.Button(frame, text="Upload Dataset", command=upload_dataset, width=30)
upload_btn.grid(row=1, column=0, pady=10, padx=10, sticky="w")

# Run Normal Classifier Button
normal_classify_btn = tk.Button(frame, text="Run Normal Classifier", command=run_normal_classifier, width=30)
normal_classify_btn.grid(row=1, column=1, pady=10, sticky="w")


# Ensemble Selection
ensemble_label = tk.Label(frame, text="Select Ensemble Technique:", font=("Arial", 12,"bold"))
ensemble_label.grid(row=1, column=4, pady=5, sticky="w")

ensemble_var = tk.StringVar()
ensemble_dropdown = ttk.Combobox(frame, textvariable=ensemble_var, values=["Voting", "Bagging", "Boosting", "Stacking"])
ensemble_dropdown.bind("<<ComboboxSelected>>", lambda e: update_parameters())
ensemble_dropdown.grid(row=1, column=5, pady=5, sticky="w")

# Parameter Frame (Inside Frame)
param_frame = tk.Frame(frame, padx=10, pady=10, borderwidth=2, relief="groove")
param_frame.grid(row=2, column=5, rowspan=4, pady=5, sticky="w")

# Run Ensemble Classifier Button
ensemble_classify_btn = tk.Button(frame, text="Run Classification", command=run_ensemble_classifier)
ensemble_classify_btn.grid(row=1, column=7, pady=10, sticky="w")



dataset_frame = tk.Frame(frame)
dataset_frame.grid(row=4, column=0, pady=5, sticky="w")
# dataset_frame.pack_propagate(False)

# Results Section
# result_label = tk.Label(frame, text="Results:", font=("Arial", 12, "bold"))
# result_label.grid(row=6, column=1, pady=10, sticky="w")

result_frame = tk.Frame(frame)
result_frame.grid(row=4, column=1, columnspan=5, pady=5, padx= 10,sticky="w")

root.mainloop()