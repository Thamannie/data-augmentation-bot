import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import os

class MLProcessor:
    def __init__(self):
        self.pipeline = None
        self.model_path = '/app/model.pkl'
        
    def train(self, data_path):
        """Train model with SMOTE resampling"""
        try:
            df = pd.read_csv(data_path)
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Create pipeline
            self.pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100))
            ])
            
            self.pipeline.fit(X, y)
            joblib.dump(self.pipeline, self.model_path)
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def predict(self, inputs):
        """Make predictions"""
        if not self.pipeline:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
            else:
                raise RuntimeError("Model not trained")
                
        # Properly indented return statement
        return self.pipeline.predict(pd.DataFrame(inputs))