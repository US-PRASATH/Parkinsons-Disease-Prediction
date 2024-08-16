from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train, model_type='svm'):
    """Trains the model based on the model_type argument."""
    if model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Saves the trained model to a file."""
    joblib.dump(model, file_path)
