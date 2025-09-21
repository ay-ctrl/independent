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

# Set matplotlib to non-interactive backend
plt.ioff()
plt.switch_backend('Agg')

def main():
    print("="*60)
    print("DRUG CONSUMPTION CLASSIFICATION ANALYSIS")
    print("="*60)
    
    # 1. Load Data
    print("\n1. Loading Drug Consumption dataset from UCI repository...")
    
    try:
        drug_consumption = fetch_ucirepo(id=373)
        X = drug_consumption.data.features
        y = drug_consumption.data.targets
        data = pd.concat([X, y], axis=1)
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {data.shape}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Targets: {y.shape[1]}")
        print(f"  - Missing values: {data.isnull().sum().sum()}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # 2. Feature Analysis
    print("\n" + "="*60)
    print("2. FEATURE ANALYSIS")
    print("="*60)
    
    feature_names = X.columns.tolist()
    print(f"\nFeatures ({len(feature_names)}):")
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nFeature Statistics:")
    feature_stats = X.describe()
    print(feature_stats)
    
    # Save feature correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature correlation heatmap saved as 'feature_correlation.png'")
    
    # 3. Data Preparation for Cannabis Classification
    print("\n" + "="*60)
    print("3. BINARY CLASSIFICATION PREPARATION")
    print("="*60)
    
    target_drug = 'cannabis'
    print(f"Target drug: {target_drug}")
    
    # Prepare features
    features = X.copy()
    
    # Convert cannabis usage to binary
    binary_mapping = {
        'CL0': 0,  # Never Used
        'CL1': 0,  # Used over a Decade Ago
        'CL2': 0,  # Used in Last Decade
        'CL3': 1,  # Used in Last Year
        'CL4': 1,  # Used in Last Month
        'CL5': 1,  # Used in Last Week
        'CL6': 1   # Used in Last Day
    }
    
    targets = y[target_drug].map(binary_mapping)
    
    print(f"\nTarget distribution:")
    print(f"  No recent use (0): {(targets == 0).sum():4d} ({(targets == 0).mean():.1%})")
    print(f"  Recent use (1):    {(targets == 1).sum():4d} ({(targets == 1).mean():.1%})")
    
    # 4. Model Development
    print("\n" + "="*60)
    print("4. CLASSIFIER MODEL DEVELOPMENT")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
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
        
        print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"    Test Accuracy: {accuracy:.4f}")
        print(f"    AUC Score: {auc_score:.4f}")
    
    # Deep Learning Model
    print(f"\n  Training Deep Neural Network...")
    
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
    
    print(f"    Test Accuracy: {nn_accuracy:.4f}")
    print(f"    AUC Score: {nn_auc:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    
    print(f"\n✓ Best Model: {best_model_name} (Accuracy: {results[best_model_name]['test_accuracy']:.4f})")
    
    # 5. Performance Analysis
    print("\n" + "="*60)
    print("5. PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Model comparison table
    print(f"\n{'Model':<20} {'CV Mean':<10} {'CV Std':<10} {'Test Acc':<10} {'AUC':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        cv_mean = f"{result['cv_mean']:.4f}" if result['cv_mean'] else "N/A"
        cv_std = f"{result['cv_std']:.4f}" if result['cv_std'] else "N/A"
        print(f"{name:<20} {cv_mean:<10} {cv_std:<10} {result['test_accuracy']:<10.4f} {result['auc_score']:<10.4f}")
    
    # Performance visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in model_names]
    aucs = [results[name]['auc_score'] for name in model_names]
    
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # AUC comparison
    axes[0, 1].bar(model_names, aucs, color='lightcoral')
    axes[0, 1].set_title('AUC Score Comparison')
    axes[0, 1].set_ylabel('AUC Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # ROC Curves
    for name in results.keys():
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        auc_score = results[name]['auc_score']
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves')
    axes[1, 0].legend()
    
    # Best model confusion matrix
    best_predictions = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Performance analysis plots saved as 'model_performance.png'")
    
    # Detailed classification report
    print(f"\nClassification Report - {best_model_name}:")
    print(classification_report(y_test, best_predictions))
    
    # Feature importance for tree-based models
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nFeature Importance (Random Forest):")
        for i in range(len(feature_names)):
            print(f"  {i+1:2d}. {feature_names[indices[i]]:<15} {importances[indices[i]]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices], color='lightgreen')
        plt.title('Feature Importance - Random Forest')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved as 'feature_importance.png'")
    
    # 6. User Prediction Function
    print("\n" + "="*60)
    print("6. USER PREDICTION SETUP")
    print("="*60)
    
    def predict_user_input():
        """Function to predict drug consumption for user input"""
        
        print("\nEnter your personality and demographic data:")
        print("(Enter numerical values as described)")
        
        feature_descriptions = {
            'age': 'Age (typical range: -0.95 to 2.59)',
            'gender': 'Gender (typical range: -0.48 to 0.48)',
            'education': 'Education level (typical range: -2.43 to 1.98)',
            'country': 'Country (typical range: -0.57 to 0.21)',
            'ethnicity': 'Ethnicity (typical range: -1.10 to 1.90)',
            'nscore': 'Neuroticism score (typical range: -2.55 to 2.90)',
            'escore': 'Extraversion score (typical range: -2.44 to 3.27)',
            'oscore': 'Openness score (typical range: -2.85 to 2.90)',
            'ascore': 'Agreeableness score (typical range: -2.07 to 2.23)',
            'cscore': 'Conscientiousness score (typical range: -2.41 to 2.99)',
            'impulsive': 'Impulsiveness score (typical range: -2.55 to 2.44)',
            'ss': 'Sensation seeking score (typical range: -2.07 to 2.32)'
        }
        
        user_data = []
        
        for feature in feature_names:
            description = feature_descriptions.get(feature, f'{feature} (continuous value)')
            while True:
                try:
                    value = float(input(f"Enter {description}: "))
                    user_data.append(value)
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        # Scale user data
        user_data_scaled = scaler.transform([user_data])
        
        # Get best model and make prediction
        best_model = results[best_model_name]['model']
        
        if best_model_name == 'Deep Neural Network':
            prediction_proba = best_model.predict(user_data_scaled, verbose=0)[0][0]
            prediction = int(prediction_proba > 0.5)
        else:
            prediction = best_model.predict(user_data_scaled)[0]
            prediction_proba = best_model.predict_proba(user_data_scaled)[0][1]
        
        print(f"\n" + "="*50)
        print(f"PREDICTION RESULTS")
        print(f"="*50)
        print(f"Model used: {best_model_name}")
        print(f"Predicted class: {prediction} ({'Recent cannabis use' if prediction == 1 else 'No recent cannabis use'})")
        print(f"Probability of recent use: {prediction_proba:.4f}")
        print(f"Confidence: {max(prediction_proba, 1-prediction_proba):.4f}")
        
        return prediction, prediction_proba
    
    # Store the prediction function globally
    global user_predict
    user_predict = predict_user_input
    
    print("\n✓ Analysis complete!")
    print("\nGenerated files:")
    print("  - feature_correlation.png")
    print("  - model_performance.png") 
    print("  - feature_importance.png")
    
    print(f"\nTo make predictions for new users, call:")
    print(f"  user_predict()")
    
    return results, scaler, feature_names

if __name__ == "__main__":
    results, scaler, feature_names = main()