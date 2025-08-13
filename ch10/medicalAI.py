# Install required packages
# !pip install transformers torch datasets pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Medical LLM Analysis - Understanding AGI in Healthcare")
print("=" * 60)

# PART 1: Initialize Medical LLM and Create Dataset
print("Initializing Medical Language Model...")

print("Initializing lightweight medical analysis models...")

try:
    # Sentiment analysis as a proxy for medical text analysis
    medical_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True,
        device=-1  
    )
    print("Medical text classifier initialized!")
except Exception as e:
    print(f"Classifier initialization failed: {e}")
    medical_classifier = None

print("Using rule-based medical NLP")

def construct_narrative_corpus(n_patients=500):
    """
    Create synthetic medical dataset with text descriptions

    """
    np.random.seed(42)
    
    # Patient demographics
    ages = np.random.normal(45, 15, n_patients).astype(int)
    ages = np.clip(ages, 18, 90)
    
    genders = np.random.choice(['Male', 'Female'], n_patients)
    
    # Symptoms and conditions 
    symptoms_list = [
        "chest pain", "shortness of breath", "fatigue", "dizziness",
        "irregular heartbeat", "high blood pressure", "leg swelling",
        "nausea", "sweating", "back pain", "joint pain", "headaches"
    ]
    
    medical_history = [
        "diabetes", "hypertension", "high cholesterol", "obesity",
        "smoking history", "family history of heart disease", 
        "previous heart attack", "stroke history", "kidney disease"
    ]
    
    # Generate patient descriptions
    patient_descriptions = []
    risk_labels = []
    
    for i in range(n_patients):
        age = ages[i]
        gender = genders[i]
        
        # Select random symptoms and history
        num_symptoms = np.random.poisson(2) + 1
        patient_symptoms = np.random.choice(symptoms_list, min(num_symptoms, 5), replace=False)
        
        num_conditions = np.random.poisson(1)
        patient_history = np.random.choice(medical_history, min(num_conditions, 3), replace=False)
        
        # Create patient description
        description = f"Patient: {age}-year-old {gender}. "
        
        if len(patient_symptoms) > 0:
            description += f"Presenting symptoms: {', '.join(patient_symptoms)}. "
        
        if len(patient_history) > 0:
            description += f"Medical history: {', '.join(patient_history)}. "
        
        # Vitals 
        bp_sys = np.random.normal(130, 20)
        cholesterol = np.random.normal(200, 40)
        description += f"Blood pressure: {bp_sys:.0f}/80, Cholesterol: {cholesterol:.0f}."
        
        patient_descriptions.append(description)
        
        # Determine risk based on age, symptoms, and history
        risk_score = 0
        risk_score += (age > 60) * 2
        risk_score += ("chest pain" in patient_symptoms) * 3
        risk_score += ("shortness of breath" in patient_symptoms) * 2
        risk_score += ("diabetes" in patient_history) * 2
        risk_score += ("hypertension" in patient_history) * 2
        risk_score += ("smoking history" in patient_history) * 3
        risk_score += (bp_sys > 140) * 1
        risk_score += (cholesterol > 240) * 1
        
        risk_labels.append("High Risk" if risk_score > 4 else "Low Risk")
    
    df = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'patient_description': patient_descriptions,
        'risk_category': risk_labels
    })
    
    return df

# Create medical dataset
print("\nConstructing narrative corpus with patient descriptions...")
medical_data = construct_narrative_corpus(500)
print(f"Narrative corpus created with {len(medical_data)} patients")

print("\nSample patient descriptions:")
for i in range(3):
    print(f"\nPatient {i+1} ({medical_data.iloc[i]['risk_category']}):")
    print(medical_data.iloc[i]['patient_description'])


# ========== put this code block in a new cell ==========
# LLM-Based Medical Text Analysis

print("\nLLM MEDICAL TEXT ANALYSIS")
print("=" * 35)

def synthesize_clinical_narratives(text_descriptions, sample_size=10):
    """
    Use lightweight NLP to synthesize clinical narrative descriptions
    """
    results = []
    
    # Analyze sample of descriptions
    sample_texts = text_descriptions[:sample_size]
    
    print(f"Synthesizing {len(sample_texts)} patient narratives...")
    
    for i, description in enumerate(sample_texts):
        try:
            # Rule-based analysis 
            medical_keywords = identify_symptom_patterns(description)
            risk_indicators = aggregate_vulnerability_markers(description)
            
            # Medical urgency
            urgency_score = compute_intervention_priority(description)
            
            llm_analysis = None
            if medical_classifier is not None:
                try:
                    llm_analysis = medical_classifier(description[:512])  
                except:
                    llm_analysis = None
            
            result = {
                'patient_id': i,
                'description': description,
                'llm_analysis': llm_analysis,
                'medical_keywords': medical_keywords,
                'risk_indicators': risk_indicators,
                'urgency_score': urgency_score
            }
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 3 == 0:
                print(f"  Processed {i + 1}/{len(sample_texts)} patients...")
            
        except Exception as e:
            print(f"Error analyzing patient {i}: {e}")
            continue
    
    return results

def compute_intervention_priority(text):
    """Calculate medical intervention priority based on symptom urgency"""
    urgent_keywords = [
        'chest pain', 'heart attack', 'stroke', 'emergency', 
        'severe', 'acute', 'critical', 'urgent'
    ]
    
    moderate_keywords = [
        'shortness of breath', 'irregular heartbeat', 
        'high blood pressure', 'diabetes'
    ]
    
    text_lower = text.lower()
    score = 0
    
    for keyword in urgent_keywords:
        if keyword in text_lower:
            score += 3
    
    for keyword in moderate_keywords:
        if keyword in text_lower:
            score += 1
    
    return min(score, 10)  

def identify_symptom_patterns(text):
    """Identify medical symptom patterns from patient narrative"""
    medical_terms = [
        'chest pain', 'shortness of breath', 'fatigue', 'dizziness',
        'diabetes', 'hypertension', 'cholesterol', 'smoking', 'heart disease'
    ]
    
    found_terms = []
    text_lower = text.lower()
    for term in medical_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms

def aggregate_vulnerability_markers(text):
    """Aggregate cardiovascular vulnerability markers in narrative"""
    high_risk_terms = [
        'chest pain', 'heart attack', 'diabetes', 'smoking', 
        'high cholesterol', 'hypertension', 'obesity'
    ]
    
    count = 0
    text_lower = text.lower()
    for term in high_risk_terms:
        if term in text_lower:
            count += 1
    
    return count

# Perform clinical narrative 
narrative_results = synthesize_clinical_narratives(medical_data['patient_description'].tolist())

print("\nClinical Narrative Synthesis Results:")
print("=" * 40)

for result in narrative_results[:5]:  
    print(f"\nPatient {result['patient_id'] + 1}:")
    print(f"Narrative: {result['description'][:100]}...")
    print(f"Symptom Patterns: {result['medical_keywords']}")
    print(f"Vulnerability Markers: {result['risk_indicators']}")
    print(f"Intervention Priority: {result['urgency_score']}/10")
    
    if result['llm_analysis']:
        # Extract confidence 
        try:
            # Handle different pipeline output formats
            if isinstance(result['llm_analysis'], list) and len(result['llm_analysis']) > 0:
                if isinstance(result['llm_analysis'][0], dict):
                    sentiment_score = result['llm_analysis'][0]['score']
                    sentiment_label = result['llm_analysis'][0]['label']
                elif isinstance(result['llm_analysis'][0], list) and len(result['llm_analysis'][0]) > 0:
                    sentiment_score = result['llm_analysis'][0][0]['score']
                    sentiment_label = result['llm_analysis'][0][0]['label']
                else:
                    sentiment_score = 0.5
                    sentiment_label = "NEUTRAL"
            else:
                sentiment_score = 0.5
                sentiment_label = "NEUTRAL"
            
            print(f"Narrative Analysis: {sentiment_label} (confidence: {sentiment_score:.3f})")
        except (KeyError, IndexError, TypeError) as e:
            print(f"Narrative Analysis: Available (format: {type(result['llm_analysis'])})")
    
    print("-" * 40)

# ========== put this code block in a new cell ==========

# LLM-Powered Risk Assessment and Insights

print("\nCOLLECTIVE REASONING AND HOLISTIC ASSESSMENT")
print("=" * 48)

def derive_holistic_assessment(patient_description, age, gender):
    """
    Derive holistic medical assessment using collective reasoning
    """
    # Risk factor analysis
    risk_factors = []
    description_lower = patient_description.lower()
    
    # Age-based risk
    if age > 65:
        risk_factors.append("Advanced age (>65)")
    elif age > 50:
        risk_factors.append("Moderate age risk (50-65)")
    
    # Symptom-based risk
    if "chest pain" in description_lower:
        risk_factors.append("Chest pain symptoms")
    if "shortness of breath" in description_lower:
        risk_factors.append("Respiratory symptoms")
    if "diabetes" in description_lower:
        risk_factors.append("Diabetes comorbidity")
    if "smoking" in description_lower:
        risk_factors.append("Smoking history")
    if "hypertension" in description_lower or "high blood pressure" in description_lower:
        risk_factors.append("Hypertension")
    
    # Generate recommendations
    recommendations = synthesize_care_protocols(risk_factors, age)
    
    return {
        'risk_factors': risk_factors,
        'risk_score': len(risk_factors),
        'recommendations': recommendations
    }

def synthesize_care_protocols(risk_factors, age):
    """Synthesize personalized care protocols from collective wisdom"""
    recommendations = []
    
    if "Advanced age (>65)" in risk_factors or "Moderate age risk (50-65)" in risk_factors:
        recommendations.append("Regular cardiovascular screening recommended")
    
    if "Chest pain symptoms" in risk_factors:
        recommendations.append("Immediate cardiac evaluation needed")
    
    if "Diabetes comorbidity" in risk_factors:
        recommendations.append("Blood glucose monitoring and management")
    
    if "Smoking history" in risk_factors:
        recommendations.append("Smoking cessation program strongly advised")
    
    if "Hypertension" in risk_factors:
        recommendations.append("Blood pressure monitoring and management")
    
    if len(risk_factors) > 3:
        recommendations.append("Comprehensive cardiac risk assessment recommended")
    
    # General recommendations
    recommendations.extend([
        "Regular exercise as tolerated",
        "Heart-healthy diet",
        "Stress management techniques"
    ])
    
    return recommendations

print("Generating holistic medical assessments through collective reasoning...")

sample_patients = medical_data.head(5)
for idx, patient in sample_patients.iterrows():
    print(f"\n{'='*50}")
    print(f"PATIENT {idx + 1} - HOLISTIC COLLECTIVE ASSESSMENT")
    print(f"{'='*50}")
    
    insights = derive_holistic_assessment(
        patient['patient_description'], 
        patient['age'], 
        patient['gender']
    )
    
    print(f"Age: {patient['age']}, Gender: {patient['gender']}")
    print(f"Actual Risk Category: {patient['risk_category']}")
    print(f"\nPatient Narrative:")
    print(patient['patient_description'])
    
    print(f"\nCollective Risk Analysis:")
    print(f"Risk Factors Identified: {insights['risk_score']}")
    for factor in insights['risk_factors']:
        print(f"  - {factor}")
    
    print(f"\nSynthesized Care Protocols:")
    for i, rec in enumerate(insights['recommendations'][:5], 1):
        print(f"  {i}. {rec}")

# ========== put this code block in a new cell ==========
# LLM vs Traditional ML Comparison

print("\nLLM vs TRADITIONAL ML COMPARISON")
print("=" * 40)

# Traditional ML 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Prepare data 
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_text = vectorizer.fit_transform(medical_data['patient_description'])
y = medical_data['risk_category']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Split for training and test
descriptions_train, descriptions_test, _, _ = train_test_split(
    medical_data['patient_description'], y, test_size=0.2, random_state=42
)

traditional_model = RandomForestClassifier(n_estimators=50, random_state=42)
traditional_model.fit(X_train, y_train)

# Evaluate model
traditional_predictions = traditional_model.predict(X_test)
traditional_accuracy = accuracy_score(y_test, traditional_predictions)

print(f"Traditional ML Accuracy: {traditional_accuracy:.2%}")

# Collective reasoning predictions 
def forecast_health_trajectory(description):
    insights = derive_holistic_assessment(description, 50, "Unknown")  
    risk_score = insights['risk_score']
    return "High Risk" if risk_score > 2 else "Low Risk"

# Evaluate collective reasoning 
test_descriptions = descriptions_test.tolist()
test_actual = y_test.tolist()

collective_predictions = [forecast_health_trajectory(desc) for desc in test_descriptions]
collective_accuracy = accuracy_score(test_actual, collective_predictions)

print(f"Collective Reasoning Accuracy: {collective_accuracy:.2%}")

print(f"\nComparison Results:")
print(f"Traditional ML: {traditional_accuracy:.2%}")
print(f"Collective Reasoning: {collective_accuracy:.2%}")

# ========== put this code block in a new cell ==========
# Advanced LLM Medical Reasoning

print("\nCOLLECTIVE REASONING MEDICAL INTELLIGENCE")
print("=" * 45)

def execute_collective_reasoning():
    """
    Execute collective reasoning for complex medical decision-making
    """
    complex_case = """
    Patient: 58-year-old Male. Presenting symptoms: chest pain, shortness of breath, 
    fatigue, irregular heartbeat. Medical history: diabetes, hypertension, smoking history, 
    family history of heart disease. Blood pressure: 165/95, Cholesterol: 280. 
    Recent stress test showed abnormalities.
    """
    
    print("Complex Medical Case:")
    print(complex_case)
    
    # Simulate collective reasoning 
    print("\nCollective Medical Reasoning Process:")
    print("1. SYMPTOM PATTERN SYNTHESIS:")
    print("   - Chest pain + shortness of breath -> Possible cardiac event")
    print("   - Irregular heartbeat -> Arrhythmia concern")
    print("   - Fatigue -> Could indicate reduced cardiac output")
    
    print("\n2. VULNERABILITY ASSESSMENT:")
    print("   - Age 58 + Male -> Moderate demographic risk")
    print("   - Diabetes + Hypertension -> Major cardiac risk factors")
    print("   - Smoking history -> Significant vascular damage risk")
    print("   - Family history -> Genetic predisposition")
    
    print("\n3. CLINICAL DATA INTEGRATION:")
    print("   - BP 165/95 -> Stage 2 hypertension")
    print("   - Cholesterol 280 -> High cardiovascular risk")
    print("   - Abnormal stress test -> Confirms cardiac concern")
    
    print("\n4. COLLECTIVE SYNTHESIS & RECOMMENDATIONS:")
    print("   HIGH PRIORITY: Multiple acute cardiac risk factors")
    print("   IMMEDIATE ACTIONS:")
    print("     - Emergency cardiac evaluation")
    print("     - ECG and cardiac enzymes")
    print("     - Consider cardiac catheterization")
    print("     - Initiate dual antiplatelet therapy if indicated")
    
    print("\n5. HOLISTIC CARE PROTOCOLS:")
    print("     - Aggressive blood pressure control")
    print("     - Diabetes management optimization")
    print("     - Statin therapy for cholesterol")
    print("     - Smoking cessation program")
    print("     - Cardiac rehabilitation")

execute_collective_reasoning()





