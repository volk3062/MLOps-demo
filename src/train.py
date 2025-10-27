import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os
from dotenv import load_dotenv

# load_dotenv()

# This is the crucial line for Docker. It gets the URI from docker-compose.yml
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_uri)

experiment_name = "Customer_Churn_Experiment"
mlflow.set_experiment(experiment_name)

# Data path is relative to the WORKDIR in the container
data = pd.read_csv("data/processed/cleaned.csv")
X = data.drop("Churn", axis=1)
print("Training features:", list(X.columns))
y = data["Churn"]

# Save feature metadata inside the container's file system
os.makedirs("models", exist_ok=True)
with open("models/feature_metadata.json", "w") as f:
    json.dump(list(X.columns), f)
print("✅ Feature metadata saved to models/feature_metadata.json")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model trained with accuracy: {acc:.3f}")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Initialize client to talk to the server
    client = MlflowClient()
    try:
        client.create_registered_model("ChurnModel")
    except Exception:
        pass

    client.create_model_version("ChurnModel", model_uri, run_id)
    print(f"✅ Model registered with run_id: {run_id}")





















# import mlflow
# import mlflow.sklearn
# from mlflow.tracking import MlflowClient
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_experiment("Customer_Churn_Experiment")

# data = pd.read_csv("data/processed/cleaned.csv")
# X = data.drop("Churn", axis=1)
# print("Training features:", list(X.columns))
# y = data["Churn"]

# models_dir = "/home/himashakti/Documents/Shrushanth/MLOps/MLOps-demo/models"

# os.makedirs(models_dir, exist_ok=True)
# with open(f"{models_dir}/feature_metadata.json", "w") as f:
#     json.dump(list(X.columns), f)

# print("✅ Feature metadata saved to models/feature_metadata.json")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# with mlflow.start_run() as run:
#     model = RandomForestClassifier(n_estimators=100, max_depth=8)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     acc = accuracy_score(y_test, preds)

#     mlflow.log_param("n_estimators", 100)
#     mlflow.log_param("max_depth", 8)
#     mlflow.log_metric("accuracy", acc)
#     mlflow.sklearn.log_model(model, "model")

#     print(f"✅ Model trained with accuracy: {acc:.3f}")

#     # ✅ Capture the run ID *inside* the context
#     run_id = run.info.run_id
#     model_uri = f"runs:/{run_id}/model"

#     # Register the model automatically
#     client = MlflowClient()
#     try:
#         client.create_registered_model("ChurnModel")
#     except Exception:
#         pass  # Model already exists

#     client.create_model_version("ChurnModel", model_uri, run_id)
#     print(f"✅ Model registered as versioned model with run_id: {run_id}")
