import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# All available drug targets in the dataset
DRUG_TARGETS = [
    'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 
    'crack', 'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 
    'mushrooms', 'nicotine', 'semer', 'vsa'
]

DRUG_NAMES = {
    'alcohol': 'Alcohol',
    'amphet': 'Amphetamines', 
    'amyl': 'Amyl Nitrite',
    'benzos': 'Benzodiazepines',
    'caff': 'Caffeine',
    'cannabis': 'Cannabis',
    'choc': 'Chocolate',
    'coke': 'Cocaine',
    'crack': 'Crack Cocaine',
    'ecstasy': 'Ecstasy',
    'heroin': 'Heroin',
    'ketamine': 'Ketamine',
    'legalh': 'Legal Highs',
    'lsd': 'LSD',
    'meth': 'Methamphetamine',
    'mushrooms': 'Magic Mushrooms',
    'nicotine': 'Nicotine',
    'semer': 'Semeron',
    'vsa': 'VSA (Volatile Substance Abuse)'
}

def load_and_train_all_models():
    """Load dataset and train models for all drug targets"""
    print("Loading and training models for all drug targets...")
    print("This may take a few minutes...")
    
    # Load data
    drug_consumption = fetch_ucirepo(id=373)
    X = drug_consumption.data.features
    y = drug_consumption.data.targets
    
    # Binary mapping for all drugs
    binary_mapping = {
        'CL0': 0, 'CL1': 0, 'CL2': 0,  # No recent use
        'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1  # Recent use
    }
    
    features = X.copy()
    models = {}
    scalers = {}
    
    # Train a model for each drug
    for i, drug in enumerate(DRUG_TARGETS, 1):
        print(f"  [{i:2d}/19] Training model for {DRUG_NAMES[drug]}...")
        
        # Convert drug usage to binary
        targets = y[drug].map(binary_mapping)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # Scale features for this drug
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model (using Logistic Regression for consistency)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        models[drug] = model
        scalers[drug] = scaler
    
    print("✓ All models trained successfully!")
    return models, scalers, features.columns.tolist()

def convert_to_normalized_values(responses):
    """Convert user-friendly responses to normalized values expected by the model"""
    
    # Mapping dictionaries based on UCI dataset documentation
    age_mapping = {
        '18-24': -0.95197, '25-34': -0.07854, '35-44': 0.49788,
        '45-54': 1.09449, '55-64': 1.82213, '65+': 2.59171
    }
    
    gender_mapping = {
        'Female': 0.48246, 'Male': -0.48246
    }
    
    education_mapping = {
        'Left school before 16': -2.43591,
        'Left school at 16': -1.73790,
        'Left school at 17': -1.43719,
        'Left school at 18': -1.22751,
        'Some college/university, no degree': -0.61113,
        'Professional certificate/diploma': -0.05921,
        'University degree': 0.45468,
        'Masters degree': 1.16365,
        'Doctorate degree': 1.98437
    }
    
    country_mapping = {
        'Australia': -0.09765, 'Canada': 0.24923, 'New Zealand': -0.46841,
        'Other': -0.28519, 'Republic of Ireland': 0.21128, 
        'UK': 0.96082, 'USA': -0.57009
    }
    
    ethnicity_mapping = {
        'Asian': -0.50212, 'Black': -1.10702, 'Mixed-Black/Asian': 1.90725,
        'Mixed-White/Asian': 0.12600, 'Mixed-White/Black': -0.22166,
        'Other': 0.11440, 'White': -0.31685
    }
    
    # Convert categorical responses to normalized values
    normalized = []
    normalized.append(age_mapping[responses['age']])
    normalized.append(gender_mapping[responses['gender']])
    normalized.append(education_mapping[responses['education']])
    normalized.append(country_mapping[responses['country']])
    normalized.append(ethnicity_mapping[responses['ethnicity']])
    
    # Personality scores are already in normalized form
    normalized.extend([
        responses['nscore'], responses['escore'], responses['oscore'],
        responses['ascore'], responses['cscore'], responses['impulsive'], responses['ss']
    ])
    
    return normalized

def predict_all_drugs_web(responses):
    """Predict all drugs for web form responses"""
    models, scalers, feature_names = load_and_train_all_models()
    user_data = convert_to_normalized_values(responses)
    
    predictions = {}
    high_risk_drugs = []
    
    for drug in DRUG_TARGETS:
        user_data_scaled = scalers[drug].transform([user_data])
        model = models[drug]
        prediction = model.predict(user_data_scaled)[0]
        prediction_proba = model.predict_proba(user_data_scaled)[0][1]

        if prediction_proba > 0.7:
            risk_level = "Very High"
            if prediction_proba > 0.8:
                high_risk_drugs.append(DRUG_NAMES[drug])
        elif prediction_proba > 0.5:
            risk_level = "High" 
        elif prediction_proba > 0.3:
            risk_level = "Moderate"
        elif prediction_proba > 0.1:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        predictions[drug] = {
            'probability': prediction_proba,
            'prediction': prediction,
            'risk_level': risk_level
        }
    
    return predictions, responses, high_risk_drugs