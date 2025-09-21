import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

class DrugConsumptionAnalyzer:
    def __init__(self):
        self.data = None
        self.features = None
        self.targets = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load data from UCI repository and preprocess it"""
        print("Loading Drug Consumption dataset from UCI repository...")
        
        # Fetch dataset from UCI repository
        drug_consumption = fetch_ucirepo(id=373)
        
        # Get features and targets
        X = drug_consumption.data.features
        y = drug_consumption.data.targets
        
        # Combine for easier processing
        self.data = pd.concat([X, y], axis=1)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {X.shape[1]}")
        print(f"Targets: {y.shape[1]}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return self.data
    
    def analyze_features(self):
        """Comprehensive feature analysis"""
        print("\n" + "="*60)
        print("1. FEATURE ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\nDataset Info:")
        print(f"- Total records: {len(self.data)}")
        print(f"- Total features: {len(self.feature_names)}")
        print(f"- Missing values: {self.data.isnull().sum().sum()}")
        
        # Feature statistics
        features_df = self.data[self.feature_names]
        print("\nFeature Statistics:")
        print(features_df.describe())
        
        # Feature correlations
        plt.figure(figsize=(12, 10))
        correlation_matrix = features_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Feature distributions
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            axes[i].hist(features_df[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return features_df.describe()
    
    def prepare_binary_classification_data(self, target_drug='cannabis'):
        """Prepare data for binary classification"""
        print(f"\nPreparing binary classification data for {target_drug}...")
        
        # Get features
        self.features = self.data[self.feature_names].copy()
        
        # Get target drug data and convert to binary
        target_column = self.data[target_drug].copy()
        
        # Convert drug usage categories to binary
        # CL0, CL1, CL2 -> 0 (No recent use)
        # CL3, CL4, CL5, CL6 -> 1 (Recent use)
        binary_mapping = {
            'CL0': 0,  # Never Used
            'CL1': 0,  # Used over a Decade Ago
            'CL2': 0,  # Used in Last Decade
            'CL3': 1,  # Used in Last Year
            'CL4': 1,  # Used in Last Month
            'CL5': 1,  # Used in Last Week
            'CL6': 1   # Used in Last Day
        }
        
        self.targets = target_column.map(binary_mapping)
        
        print(f"Target distribution:")
        print(f"No recent use (0): {(self.targets == 0).sum()} ({(self.targets == 0).mean():.2%})")
        print(f"Recent use (1): {(self.targets == 1).sum()} ({(self.targets == 1).mean():.2%})")
        
        return self.features, self.targets
    
    def develop_classifier_models(self):
        """Develop and compare multiple classifier models"""
        print("\n" + "="*60)
        print("2. CLASSIFIER MODEL DEVELOPMENT")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=0.2, random_state=42, stratify=self.targets
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for later use
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  AUC Score: {auc_score:.4f}")
        
        # Deep Learning Model
        print("\nTraining Deep Neural Network...")
        
        # Build neural network
        nn_model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train neural network
        history = nn_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Neural network predictions
        nn_pred_proba = nn_model.predict(X_test_scaled, verbose=0).flatten()
        nn_pred = (nn_pred_proba > 0.5).astype(int)
        
        nn_accuracy = accuracy_score(y_test, nn_pred)
        nn_auc = roc_auc_score(y_test, nn_pred_proba)
        
        results['Deep Neural Network'] = {
            'model': nn_model,
            'cv_mean': None,
            'cv_std': None,
            'test_accuracy': nn_accuracy,
            'auc_score': nn_auc,
            'predictions': nn_pred,
            'probabilities': nn_pred_proba,
            'history': history
        }
        
        print(f"  Test Accuracy: {nn_accuracy:.4f}")
        print(f"  AUC Score: {nn_auc:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {results[best_model_name]['test_accuracy']:.4f})")
        
        return results
    
    def performance_analysis(self):
        """Comprehensive performance analysis"""
        print("\n" + "="*60)
        print("3. PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Model comparison
        model_comparison = pd.DataFrame({
            'Model': list(self.models.keys()),
            'CV_Mean': [self.models[model]['cv_mean'] for model in self.models.keys()],
            'CV_Std': [self.models[model]['cv_std'] for model in self.models.keys()],
            'Test_Accuracy': [self.models[model]['test_accuracy'] for model in self.models.keys()],
            'AUC_Score': [self.models[model]['auc_score'] for model in self.models.keys()]
        })
        
        print("\nModel Performance Comparison:")
        print(model_comparison.to_string(index=False, float_format='%.4f'))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(model_comparison['Model'], model_comparison['Test_Accuracy'], color='skyblue')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # AUC comparison
        axes[0, 1].bar(model_comparison['Model'], model_comparison['AUC_Score'], color='lightcoral')
        axes[0, 1].set_title('AUC Score Comparison')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC Curves
        for model_name in self.models.keys():
            if model_name != 'Deep Neural Network':  # Skip DNN for CV plotting
                fpr, tpr, _ = roc_curve(self.y_test, self.models[model_name]['probabilities'])
                auc_score = self.models[model_name]['auc_score']
                axes[1, 0].plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Add DNN to ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, self.models['Deep Neural Network']['probabilities'])
        auc_score = self.models['Deep Neural Network']['auc_score']
        axes[1, 0].plot(fpr, tpr, label=f'Deep Neural Network (AUC = {auc_score:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()
        
        # Best model confusion matrix
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['test_accuracy'])
        best_predictions = self.models[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report - {best_model_name}:")
        print(classification_report(self.y_test, best_predictions))
        
        # Feature importance (for tree-based models)
        if 'Random Forest' in self.models or 'Gradient Boosting' in self.models:
            self._plot_feature_importance()
        
        return model_comparison
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        plt.figure(figsize=(12, 8))
        
        for i, (model_name, model_data) in enumerate(self.models.items()):
            if hasattr(model_data['model'], 'feature_importances_'):
                importances = model_data['model'].feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.subplot(1, 2, i//2 + 1)
                plt.bar(range(len(importances)), importances[indices], color='lightgreen')
                plt.title(f'Feature Importance - {model_name}')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def predict_user_data(self):
        """Interactive prediction for user input"""
        print("\n" + "="*60)
        print("4. USER PREDICTION INTERFACE")
        print("="*60)
        
        print("\nPlease provide your data for drug consumption prediction.")
        print("Feature descriptions:")
        
        feature_descriptions = {
            'age': 'Age (continuous value, typically -0.95 to 2.59)',
            'gender': 'Gender (continuous value, typically around -0.48 to 0.48)',
            'education': 'Education level (continuous value, typically -2.43 to 1.98)',
            'country': 'Country (continuous value, typically -0.57 to 0.21)',
            'ethnicity': 'Ethnicity (continuous value, typically -1.10 to 1.90)',
            'nscore': 'Neuroticism score (continuous value, typically -2.55 to 2.90)',
            'escore': 'Extraversion score (continuous value, typically -2.44 to 3.27)',
            'oscore': 'Openness score (continuous value, typically -2.85 to 2.90)',
            'ascore': 'Agreeableness score (continuous value, typically -2.07 to 2.23)',
            'cscore': 'Conscientiousness score (continuous value, typically -2.41 to 2.99)',
            'impulsive': 'Impulsiveness score (continuous value, typically -2.55 to 2.44)',
            'ss': 'Sensation seeking score (continuous value, typically -2.07 to 2.32)'
        }
        
        user_data = []
        
        for feature in self.feature_names:
            description = feature_descriptions.get(feature, f'{feature} (continuous value)')
            while True:
                try:
                    value = float(input(f"Enter {description}: "))
                    user_data.append(value)
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        # Scale user data
        user_data_scaled = self.scaler.transform([user_data])
        
        # Get best model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['test_accuracy'])
        best_model = self.models[best_model_name]['model']
        
        # Make prediction
        if best_model_name == 'Deep Neural Network':
            prediction_proba = best_model.predict(user_data_scaled, verbose=0)[0][0]
            prediction = int(prediction_proba > 0.5)
        else:
            prediction = best_model.predict(user_data_scaled)[0]
            prediction_proba = best_model.predict_proba(user_data_scaled)[0][1]
        
        print(f"\nPrediction Results using {best_model_name}:")
        print(f"Predicted class: {prediction} ({'Recent drug use' if prediction == 1 else 'No recent drug use'})")
        print(f"Probability of recent drug use: {prediction_proba:.4f}")
        print(f"Confidence: {max(prediction_proba, 1-prediction_proba):.4f}")
        
        return prediction, prediction_proba

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DrugConsumptionAnalyzer()
    
    # Load and preprocess data
    data = analyzer.load_and_preprocess_data()
    
    # Analyze features
    feature_stats = analyzer.analyze_features()
    
    # Prepare binary classification data (using cannabis as example)
    features, targets = analyzer.prepare_binary_classification_data('cannabis')
    
    # Develop classifier models
    models = analyzer.develop_classifier_models()
    
    # Performance analysis
    performance = analyzer.performance_analysis()
    
    # User prediction interface
    print("\nReady for user predictions!")
    print("You can now call analyzer.predict_user_data() to make predictions.")















