import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClimateMLModels:
    """
    Machine learning models for climate data analysis.
    Includes Temperature prediction, Region clustering, Anomaly detection

    Attributes: 
        prediction_model: the prediction model from sklearn
        clustering_model: the clustering model from sklearn (kmeans)
        anomaly_model: the anomaly model from sklearn (isolation forest)
        prediction_features: features of the prediction model
        prediction_target: target of the model
    Methods:
        train_prediction_model(self, data, target, features, test_size, model_type): Train a model to predict climate parameters.
        predict_future(self, future_data, years_ahead): Predict future climate parameter values.
        cluster_regions(self, data, features, n_clusters): Cluster regions with similar climate patterns.
        detect_anomalies(self, data, features, contamination):  Detect anomalies in climate data.
        _get_feature_importance(self, model, features): Get feature importance from the model if available.
    """
    
    def __init__(self):
        """Initialize the ML models module."""
        self.prediction_model = None
        self.clustering_model = None
        self.anomaly_model = None
        self.prediction_features = []
        self.prediction_target = ""
        
    def train_prediction_model(self, 
                              data: pd.DataFrame, 
                              target: str, 
                              features: Optional[List[str]] = None,
                              test_size: float = 0.2,
                              model_type: str = 'random_forest') -> Dict[str, float]:
        """
        Train a model to predict climate parameters.
        
        Args:
            data (DataFrame): DataFrame containing climate data
            target (str): Target variable to predict
            
        Kwargs:
            features (List[str]): List of feature columns to use (if None, use all numeric columns except target)
            test_size (float): Proportion of data to use for testing
            model_type (str): Type of model to use ('linear' or 'random_forest')

        Returns:
            Dictionary with model performance metrics
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Default to all numeric features if not specified
        if features is None:
            features = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col != target]
        
        # Check that all features exist in the data
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns {missing_features} not found in data")
        
        logger.info(f"Training prediction model for {target} using {len(features)} features")
        
        # Prepare data
        X = data[features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if model_type == 'linear':
            model = LinearRegression()
        else:  # default to random forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and features
        self.prediction_model = model
        self.prediction_features = features
        self.prediction_target = target
        
        logger.info(f"Model trained. MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'model_type': model_type,
            'feature_importance': self._get_feature_importance(model, features)
        }
    
    def predict_future(self, 
                      future_data: pd.DataFrame, 
                      years_ahead: int = 5) -> pd.DataFrame:
        """
        Predict future climate parameter values.
        
        Args:
            future_data (DataFrame): DataFrame with future feature values

        Kwargs:
            years_ahead (int): Number of years ahead to predict
            
        Returns:
            DataFrame with predicted values
        """
        if self.prediction_model is None:
            raise ValueError("Prediction model not trained. Call train_prediction_model() first.")
        
        # Check that all required features are in the future data
        missing_features = [f for f in self.prediction_features if f not in future_data.columns]
        if missing_features:
            raise ValueError(f"Future data missing required features: {missing_features}")
        
        logger.info(f"Predicting {self.prediction_target} {years_ahead} years ahead")
        
        # Make predictions
        X_future = future_data[self.prediction_features]
        predictions = self.prediction_model.predict(X_future)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'year': future_data['year'] if 'year' in future_data.columns else range(years_ahead),
            f'predicted_{self.prediction_target}': predictions
        })
        
        return result
    
    def cluster_regions(self, 
                       data: pd.DataFrame, 
                       features: List[str], 
                       n_clusters: int = 3) -> pd.DataFrame:
        """
        Cluster regions with similar climate patterns.
        
        Args:
            data (DataFrame): DataFrame containing climate data
            features (List[str]): List of feature columns to use for clustering
        Kwargs:
            n_clusters (int): Number of clusters to create
            
        Returns:
            DataFrame with original data and cluster assignments
        """
        # Check that all features exist in the data
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns {missing_features} not found in data")
        
        logger.info(f"Clustering regions using {len(features)} features into {n_clusters} clusters")
        
        # Train clustering model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data[features])
        
        # Store model
        self.clustering_model = kmeans
        
        # Add cluster assignments to data
        result = data.copy()
        result['cluster'] = clusters
        
        logger.info(f"Clustering complete. Cluster distribution: {np.bincount(clusters)}")
        
        return result
    
    def detect_anomalies(self, 
                        data: pd.DataFrame, 
                        features: List[str], 
                        contamination: float = 0.05) -> pd.DataFrame:
        """
        Detect anomalies in climate data.
        
        Args:
            data (DataFrame): DataFrame containing climate data
            feature (List[str]): List of feature columns to use for anomaly detection
        Kwargs:
            contamination (float): Expected proportion of outliers in the data
            
        Returns:
            DataFrame with original data and anomaly flags
        """
        # Check that all features exist in the data
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns {missing_features} not found in data")
        
        logger.info(f"Detecting anomalies using {len(features)} features")
        
        # Train anomaly detection model
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(data[features])
        
        # Store model
        self.anomaly_model = iso_forest
        
        # Add anomaly flags to data (1 for normal, -1 for anomaly)
        result = data.copy()
        result['is_anomaly'] = anomalies == -1
        
        anomaly_count = np.sum(result['is_anomaly'])
        logger.info(f"Anomaly detection complete. Found {anomaly_count} anomalies ({anomaly_count/len(data)*100:.2f}%)")
        
        return result
    
    def _get_feature_importance(self, model, features: List[str]) -> Dict[str, float]:
        """Get feature importance from the model if available."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(features, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(features, model.coef_))
        else:
            return {}
