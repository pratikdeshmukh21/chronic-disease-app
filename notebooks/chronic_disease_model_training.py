
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== Chronic Disease Prediction Model Training ===\n")

# Load the combined dataset
print("Loading dataset...")
df = pd.read_csv('../csv/chronic_diseases_combined.csv')
print(f"Dataset loaded successfully! Shape: {df.shape}")
print(f"Diseases: {df['Disease'].value_counts().to_dict()}")

# Data preprocessing
print("\n=== Data Preprocessing ===")

# Separate features and target
target = df['Disease']
features_df = df.drop(['Disease', 'PatientID'], axis=1)

# Define preprocessing for different column types
numeric_features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'Cholesterol', 'BloodSugar']
categorical_features = ['Gender', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'FamilyHistory']
text_features = ['Symptoms', 'Lifestyle']

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")
print(f"Text features: {text_features}")

# Create preprocessing pipelines
# For numeric features: standardization
numeric_transformer = StandardScaler()

# For categorical features: one-hot encoding
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# Combine categorical and numeric preprocessing
preprocessor_basic = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Process the basic features
X_basic = preprocessor_basic.fit_transform(features_df)
feature_names = (numeric_features + 
                list(preprocessor_basic.named_transformers_['cat'].get_feature_names_out(categorical_features)))

# Create DataFrame with processed basic features
X_basic_df = pd.DataFrame(X_basic, columns=feature_names)

print(f"\nBasic features shape after preprocessing: {X_basic_df.shape}")

# Text feature processing using TF-IDF
print("\n=== Text Feature Processing ===")

# Combine symptoms and lifestyle text
combined_text = features_df['Symptoms'] + ' ' + features_df['Lifestyle']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=100,  # Limit to top 100 features
    stop_words='english',
    ngram_range=(1, 2),  # Include both unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)

X_tfidf = tfidf_vectorizer.fit_transform(combined_text)
tfidf_feature_names = [f"tfidf_{feature}" for feature in tfidf_vectorizer.get_feature_names_out()]

print(f"TF-IDF features shape: {X_tfidf.shape}")
print(f"Top TF-IDF features: {list(tfidf_vectorizer.get_feature_names_out())[:10]}")

# Combine basic features with TF-IDF features
import scipy.sparse as sp
X_tfidf_dense = X_tfidf.toarray()
X_combined = np.hstack([X_basic, X_tfidf_dense])
combined_feature_names = feature_names + tfidf_feature_names

print(f"\nCombined features shape: {X_combined.shape}")

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(target)
class_names = le.classes_
print(f"\nTarget classes: {class_names}")
print(f"Class distribution: {dict(zip(class_names, np.bincount(y)))}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Model Training and Evaluation
print("\n=== Model Training and Evaluation ===")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
    'SVM': SVC(random_state=42, probability=True)
}

# Store results
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)
    trained_models[name] = model

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # For multiclass ROC AUC
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = 0.0
    else:
        roc_auc = 0.0

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = trained_models[best_model_name]

print(f"\n=== Best Model: {best_model_name} ===")
print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")

# Generate detailed classification report for best model
y_pred_best = best_model.predict(X_test)
print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Save the best model and preprocessing components
print("\n=== Saving Model and Preprocessing Components ===")
joblib.dump(best_model, '../app/models/best_disease_prediction_model.pkl')
joblib.dump(preprocessor_basic, '../app/models/preprocessor_basic.pkl')
joblib.dump(tfidf_vectorizer, '../app/models/tfidf_vectorizer.pkl')
joblib.dump(le, '../app/models/label_encoder.pkl')
joblib.dump(combined_feature_names, '../app/models/feature_names.pkl')

print(f"Best model ({best_model_name}) saved successfully!")
print("All preprocessing components saved!")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv('../models/model_comparison_results.csv')
print("Model comparison results saved!")

# Feature importance analysis (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n=== Feature Importance Analysis for {best_model_name} ===")
    feature_importance = pd.DataFrame({
        'feature': combined_feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 15 Most Important Features:")
    print(feature_importance.head(15))

    feature_importance.to_csv('../models/feature_importance.csv', index=False)
    print("Feature importance saved!")

print("\n=== Model Training Complete! ===")
print("All files saved in the models/ directory")
