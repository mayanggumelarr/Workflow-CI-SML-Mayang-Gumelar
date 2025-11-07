import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(40) # jaga konsistensi model

    # baca data
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)),"train_data_liver.csv")
    df = pd.read_csv(file_path)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('Target', axis=1),
        df['Target'],
        test_size=0.2,
        random_state=42
    )

    # contoh input
    input_example = X_train[0:5]

    # definisikan parameter berdasarkan hasil tuning
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 257
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 13

    # tracking MLflow
    with mlflow.start_run():
        # latih model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        predicted_liver = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        model.fit(X_train,y_train)
        # log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
