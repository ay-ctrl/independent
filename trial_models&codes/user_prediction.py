import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_trained_model():
    """Load the dataset and train the best model (Logistic Regression)"""
    print("Loading and training the model...")
    
    # Load data
    drug_consumption = fetch_ucirepo(id=373)
    X = drug_consumption.data.features
    y = drug_consumption.data.targets
    
    # Prepare cannabis binary classification
    binary_mapping = {
        'CL0': 0, 'CL1': 0, 'CL2': 0,  # No recent use
        'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1  # Recent use
    }
    
    features = X.copy()
    targets = y['cannabis'].map(binary_mapping)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train best model (Logistic Regression)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    print("✓ Model trained successfully!")
    return model, scaler, features.columns.tolist()

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

def predict_drug_consumption():
    """Interactive function to predict drug consumption for user input"""
    
    print("\\n" + "="*70)
    print("CANNABIS CONSUMPTION PREDICTION - USER-FRIENDLY INTERFACE")
    print("="*70)
    
    # Load trained model
    model, scaler, feature_names = load_trained_model()
    
    print("\\nPlease provide your personal information for cannabis consumption prediction.")
    print("Choose from the available options for each question.")
    print("-" * 70)
    
    responses = {}
    
    # 1. Age
    print("\\n1. AGE GROUP:")
    age_options = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    for i, option in enumerate(age_options, 1):
        print(f"   {i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (1-6): "))
            if 1 <= choice <= 6:
                responses['age'] = age_options[choice-1]
                break
            print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")
    
    # 2. Gender
    print("\\n2. GENDER:")
    gender_options = ['Female', 'Male']
    for i, option in enumerate(gender_options, 1):
        print(f"   {i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (1-2): "))
            if 1 <= choice <= 2:
                responses['gender'] = gender_options[choice-1]
                break
            print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
    
    # 3. Education
    print("\\n3. EDUCATION LEVEL:")
    education_options = [
        'Left school before 16', 'Left school at 16', 'Left school at 17',
        'Left school at 18', 'Some college/university, no degree',
        'Professional certificate/diploma', 'University degree',
        'Masters degree', 'Doctorate degree'
    ]
    for i, option in enumerate(education_options, 1):
        print(f"   {i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (1-9): "))
            if 1 <= choice <= 9:
                responses['education'] = education_options[choice-1]
                break
            print("Please enter a number between 1 and 9.")
        except ValueError:
            print("Please enter a valid number.")
    
    # 4. Country
    print("\\n4. COUNTRY OF RESIDENCE:")
    country_options = ['Australia', 'Canada', 'New Zealand', 'Other', 
                      'Republic of Ireland', 'UK', 'USA']
    for i, option in enumerate(country_options, 1):
        print(f"   {i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (1-7): "))
            if 1 <= choice <= 7:
                responses['country'] = country_options[choice-1]
                break
            print("Please enter a number between 1 and 7.")
        except ValueError:
            print("Please enter a valid number.")
    
    # 5. Ethnicity
    print("\\n5. ETHNICITY:")
    ethnicity_options = ['Asian', 'Black', 'Mixed-Black/Asian', 'Mixed-White/Asian',
                        'Mixed-White/Black', 'Other', 'White']
    for i, option in enumerate(ethnicity_options, 1):
        print(f"   {i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (1-7): "))
            if 1 <= choice <= 7:
                responses['ethnicity'] = ethnicity_options[choice-1]
                break
            print("Please enter a number between 1 and 7.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Personality Scores (NEO-FFI-R and other measures)
    print("\\n" + "="*70)
    print("PERSONALITY ASSESSMENT")
    print("="*70)
    print("For the following personality traits, please rate yourself on a scale")
    print("from 1 (very low) to 7 (very high). We'll convert to standardized scores.\\n")
    
    personality_traits = {
        'nscore': ('Neuroticism (anxiety, emotional instability)', -2.5, 2.9),
        'escore': ('Extraversion (sociability, assertiveness)', -2.4, 3.3),
        'oscore': ('Openness (creativity, intellectual curiosity)', -2.9, 2.9),
        'ascore': ('Agreeableness (cooperation, trust)', -2.1, 2.2),
        'cscore': ('Conscientiousness (organization, self-discipline)', -2.4, 3.0)
    }
    
    for trait, (description, min_val, max_val) in personality_traits.items():
        print(f"{description.upper()}:")
        print(f"Range: {min_val:.1f} to {max_val:.1f} (standardized)")
        while True:
            try:
                rating = int(input("Rate yourself 1-7 (1=very low, 7=very high): "))
                if 1 <= rating <= 7:
                    # Convert 1-7 scale to approximate standardized range
                    normalized_score = min_val + (rating - 1) * (max_val - min_val) / 6
                    responses[trait] = round(normalized_score, 2)
                    break
                print("Please enter a number between 1 and 7.")
            except ValueError:
                print("Please enter a valid number.")
        print(f"→ Standardized score: {responses[trait]:.2f}\\n")
    
    # Impulsiveness and Sensation Seeking
    print("IMPULSIVENESS (tendency to act without thinking):")
    print("Range: -2.6 to 2.9 (standardized)")
    while True:
        try:
            rating = int(input("Rate yourself 1-7 (1=very low, 7=very high): "))
            if 1 <= rating <= 7:
                responses['impulsive'] = round(-2.6 + (rating - 1) * 5.5 / 6, 2)
                break
            print("Please enter a number between 1 and 7.")
        except ValueError:
            print("Please enter a valid number.")
    print(f"→ Standardized score: {responses['impulsive']:.2f}\\n")
    
    print("SENSATION SEEKING (desire for novel, exciting experiences):")
    print("Range: -2.1 to 1.9 (standardized)")
    while True:
        try:
            rating = int(input("Rate yourself 1-7 (1=very low, 7=very high): "))
            if 1 <= rating <= 7:
                responses['ss'] = round(-2.1 + (rating - 1) * 4.0 / 6, 2)
                break
            print("Please enter a number between 1 and 7.")
        except ValueError:
            print("Please enter a valid number.")
    print(f"→ Standardized score: {responses['ss']:.2f}\\n")
    
    # Convert to normalized values
    user_data = convert_to_normalized_values(responses)
    
    # Scale user data
    user_data_scaled = scaler.transform([user_data])
    
    # Make prediction
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]
    
    # Display results
    print("\\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Model used: Logistic Regression (82.76% accuracy)")
    
    # Show user profile summary
    print(f"\\nYOUR PROFILE SUMMARY:")
    print(f"Age: {responses['age']}")
    print(f"Gender: {responses['gender']}")
    print(f"Education: {responses['education']}")
    print(f"Country: {responses['country']}")
    print(f"Ethnicity: {responses['ethnicity']}")
    print(f"\\nPersonality Scores (standardized):")
    print(f"  Neuroticism: {responses['nscore']:.2f}")
    print(f"  Extraversion: {responses['escore']:.2f}")
    print(f"  Openness: {responses['oscore']:.2f}")
    print(f"  Agreeableness: {responses['ascore']:.2f}")
    print(f"  Conscientiousness: {responses['cscore']:.2f}")
    print(f"  Impulsiveness: {responses['impulsive']:.2f}")
    print(f"  Sensation Seeking: {responses['ss']:.2f}")
    
    print(f"\\n" + "-"*70)
    print(f"PREDICTION: {prediction}")
    
    if prediction == 1:
        print("→ LIKELY TO USE CANNABIS RECENTLY")
        print("  (Used in last year/month/week/day)")
    else:
        print("→ UNLIKELY TO USE CANNABIS RECENTLY") 
        print("  (Never used or used over a decade ago)")
    
    print(f"\\nProbability of recent cannabis use: {prediction_proba:.4f} ({prediction_proba*100:.1f}%)")
    print(f"Confidence level: {max(prediction_proba, 1-prediction_proba):.4f} ({max(prediction_proba, 1-prediction_proba)*100:.1f}%)")
    
    # Risk interpretation
    if prediction_proba > 0.8:
        risk_level = "Very High"
        interpretation = "Strong likelihood of recent cannabis use"
    elif prediction_proba > 0.6:
        risk_level = "High"
        interpretation = "High likelihood of recent cannabis use"
    elif prediction_proba > 0.4:
        risk_level = "Moderate"
        interpretation = "Moderate risk - could go either way"
    elif prediction_proba > 0.2:
        risk_level = "Low"
        interpretation = "Low likelihood of recent cannabis use"
    else:
        risk_level = "Very Low"
        interpretation = "Very low likelihood of recent cannabis use"
    
    print(f"\\nRisk Level: {risk_level}")
    print(f"Interpretation: {interpretation}")
    print("\\n" + "="*70)
    
    return prediction, prediction_proba, responses

def show_example_prediction():
    """Show an example prediction with sample data"""
    print("\\n" + "="*70)
    print("EXAMPLE PREDICTION - MODERATE RISK PROFILE")
    print("="*70)
    
    # Example responses (moderate risk profile)
    example_responses = {
        'age': '25-34',
        'gender': 'Male',
        'education': 'University degree',
        'country': 'UK',
        'ethnicity': 'White',
        'nscore': 0.2,   # Moderate neuroticism
        'escore': 1.5,   # High extraversion
        'oscore': 1.0,   # High openness
        'ascore': 0.0,   # Moderate agreeableness
        'cscore': -0.5,  # Lower conscientiousness
        'impulsive': 0.8, # Moderate-high impulsiveness
        'ss': 1.2        # High sensation seeking
    }
    
    print("Example profile:")
    print(f"Age: {example_responses['age']}")
    print(f"Gender: {example_responses['gender']}")
    print(f"Education: {example_responses['education']}")
    print(f"Country: {example_responses['country']}")
    print(f"Ethnicity: {example_responses['ethnicity']}")
    print(f"\\nPersonality Scores (standardized):")
    print(f"  Neuroticism: {example_responses['nscore']:.2f}")
    print(f"  Extraversion: {example_responses['escore']:.2f}")
    print(f"  Openness: {example_responses['oscore']:.2f}")
    print(f"  Agreeableness: {example_responses['ascore']:.2f}")
    print(f"  Conscientiousness: {example_responses['cscore']:.2f}")
    print(f"  Impulsiveness: {example_responses['impulsive']:.2f}")
    print(f"  Sensation Seeking: {example_responses['ss']:.2f}")
    
    # Load model and predict
    model, scaler, _ = load_trained_model()
    example_data = convert_to_normalized_values(example_responses)
    example_scaled = scaler.transform([example_data])
    
    prediction = model.predict(example_scaled)[0]
    prediction_proba = model.predict_proba(example_scaled)[0][1]
    
    print(f"\\n" + "-"*70)
    print(f"Prediction: {prediction} ({'Recent use likely' if prediction == 1 else 'Recent use unlikely'})")
    print(f"Probability: {prediction_proba:.4f} ({prediction_proba*100:.1f}%)")
    
    if prediction_proba > 0.6:
        risk_level = "High"
    elif prediction_proba > 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    print(f"Risk Level: {risk_level}")
    print("\\n" + "="*70)

if __name__ == "__main__":
    print("DRUG CONSUMPTION PREDICTION SYSTEM")
    print("Based on personality traits and demographics")
    print("Model Accuracy: 82.76% (Logistic Regression)")
    print("\\nNOTE: This is for research/educational purposes only.")
    print("Results should not be used for clinical decisions.")
    
    while True:
        print("\\n" + "="*50)
        print("OPTIONS:")
        print("1. Make a prediction with your data")
        print("2. Show example prediction")
        print("3. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            try:
                predict_drug_consumption()
            except KeyboardInterrupt:
                print("\\nPrediction cancelled.")
            except Exception as e:
                print(f"\\nError: {e}")
                
        elif choice == '2':
            show_example_prediction()
            
        elif choice == '3':
            print("\\nGoodbye!")
            break
            
        else:
            print("\\nInvalid choice. Please enter 1, 2, or 3.")