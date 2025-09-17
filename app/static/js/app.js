// Chronic Disease Prediction System JavaScript

// Application data
const APP_DATA = {
    diseases: ["Diabetes", "Heart Disease", "Kidney Disease", "Asthma", "Hypertension"],
    modelPerformance: {
        "Logistic Regression": { accuracy: 95.67, cv_score: 93.92, std: 3.19 },
        "Random Forest": { accuracy: 94.67, cv_score: 95.92, std: 2.76 },
        "SVM": { accuracy: 90.00, cv_score: 87.50, std: 4.62 }
    },
    medicalRanges: {
        bmi: {
            underweight: { min: 0, max: 18.4, label: "Underweight", class: "warning" },
            normal: { min: 18.5, max: 24.9, label: "Normal", class: "success" },
            overweight: { min: 25.0, max: 29.9, label: "Overweight", class: "warning" },
            obese: { min: 30.0, max: 100, label: "Obese", class: "error" }
        },
        bloodPressure: {
            normal: { systolic: [0, 119], diastolic: [0, 79], label: "Normal", class: "success" },
            elevated: { systolic: [120, 129], diastolic: [0, 79], label: "Elevated", class: "warning" },
            stage1: { systolic: [130, 139], diastolic: [80, 89], label: "High Stage 1", class: "error" },
            stage2: { systolic: [140, 999], diastolic: [90, 999], label: "High Stage 2", class: "error" }
        }
    }
};

// Disease risk factors and symptoms for prediction logic
const DISEASE_PATTERNS = {
    "Heart Disease": {
        symptoms: ["chest pain", "shortness of breath", "fatigue", "irregular heartbeat", "dizziness"],
        riskFactors: {
            age: { high: 55, moderate: 45 },
            bmi: { high: 30, moderate: 25 },
            systolic_bp: { high: 140, moderate: 130 },
            cholesterol: { high: 240, moderate: 200 },
            smoking: { "Current": 0.8, "Former": 0.3, "Never": 0 }
        }
    },
    "Diabetes": {
        symptoms: ["frequent urination", "excessive thirst", "blurred vision", "fatigue", "slow healing"],
        riskFactors: {
            age: { high: 45, moderate: 35 },
            bmi: { high: 30, moderate: 25 },
            blood_sugar: { high: 126, moderate: 100 },
            family_history: { "Yes": 0.4, "No": 0 }
        }
    },
    "Hypertension": {
        symptoms: ["headache", "dizziness", "shortness of breath", "nosebleeds", "chest pain"],
        riskFactors: {
            age: { high: 65, moderate: 45 },
            systolic_bp: { high: 140, moderate: 130 },
            diastolic_bp: { high: 90, moderate: 80 },
            bmi: { high: 30, moderate: 25 }
        }
    },
    "Kidney Disease": {
        symptoms: ["fatigue", "swelling", "changes in urination", "nausea", "metallic taste"],
        riskFactors: {
            age: { high: 60, moderate: 45 },
            systolic_bp: { high: 140, moderate: 130 },
            blood_sugar: { high: 126, moderate: 100 },
            family_history: { "Yes": 0.3, "No": 0 }
        }
    },
    "Asthma": {
        symptoms: ["wheezing", "shortness of breath", "chest tightness", "coughing", "difficulty breathing"],
        riskFactors: {
            age: { high: 30, moderate: 18 },
            family_history: { "Yes": 0.5, "No": 0 },
            smoking: { "Current": 0.6, "Former": 0.2, "Never": 0 }
        }
    }
};

// DOM Elements
let currentSection = 'home';
let performanceChart = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeForm();
    initializeBMICalculator();
    renderPerformanceChart();
    console.log('Chronic Disease Prediction System initialized');
});

// Navigation functionality
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const startButton = document.querySelector('[data-section="prediction"]');
    
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetSection = button.getAttribute('data-section');
            showSection(targetSection);
        });
    });
    
    if (startButton && !startButton.classList.contains('nav-btn')) {
        startButton.addEventListener('click', () => {
            showSection('prediction');
        });
    }
}

function showSection(sectionId) {
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-section') === sectionId) {
            btn.classList.add('active');
        }
    });
    
    // Update sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        currentSection = sectionId;
    }
    
    // Special handling for model info section
    if (sectionId === 'model-info' && !performanceChart) {
        setTimeout(() => renderPerformanceChart(), 100);
    }
}

// Form initialization and validation
function initializeForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    
    form.addEventListener('submit', handleFormSubmit);
    form.addEventListener('reset', handleFormReset);
    
    // Add real-time validation
    const inputs = form.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });
}

function validateField(event) {
    const field = event.target;
    const value = field.value.trim();
    const fieldName = field.name;
    let isValid = true;
    let message = '';
    
    // Required field validation
    if (field.hasAttribute('required') && !value) {
        isValid = false;
        message = 'This field is required';
    }
    
    // Specific field validations
    switch (fieldName) {
        case 'age':
            if (value && (parseInt(value) < 18 || parseInt(value) > 120)) {
                isValid = false;
                message = 'Age must be between 18 and 120';
            }
            break;
        case 'systolic_bp':
            if (value && (parseInt(value) < 70 || parseInt(value) > 250)) {
                isValid = false;
                message = 'Systolic BP should be between 70-250 mmHg';
            }
            break;
        case 'diastolic_bp':
            if (value && (parseInt(value) < 40 || parseInt(value) > 150)) {
                isValid = false;
                message = 'Diastolic BP should be between 40-150 mmHg';
            }
            break;
        case 'cholesterol':
            if (value && (parseInt(value) < 100 || parseInt(value) > 400)) {
                isValid = false;
                message = 'Cholesterol should be between 100-400 mg/dL';
            }
            break;
        case 'blood_sugar':
            if (value && (parseInt(value) < 50 || parseInt(value) > 300)) {
                isValid = false;
                message = 'Blood sugar should be between 50-300 mg/dL';
            }
            break;
    }
    
    showFieldValidation(field, isValid, message);
    return isValid;
}

function showFieldValidation(field, isValid, message) {
    // Remove existing validation classes and messages
    field.classList.remove('error', 'success');
    const existingHelp = field.parentNode.querySelector('.form-help');
    
    if (!isValid) {
        field.classList.add('error');
        if (existingHelp) {
            existingHelp.textContent = message;
            existingHelp.className = 'form-help error';
        } else {
            const helpElement = document.createElement('small');
            helpElement.className = 'form-help error';
            helpElement.textContent = message;
            field.parentNode.appendChild(helpElement);
        }
    } else if (field.value.trim()) {
        field.classList.add('success');
        if (existingHelp && existingHelp.classList.contains('error')) {
            existingHelp.remove();
        }
    }
}

function clearFieldError(event) {
    const field = event.target;
    field.classList.remove('error');
    const errorHelp = field.parentNode.querySelector('.form-help.error');
    if (errorHelp) {
        errorHelp.remove();
    }
}

// BMI Calculator
function initializeBMICalculator() {
    const heightField = document.getElementById('height');
    const weightField = document.getElementById('weight');
    const bmiField = document.getElementById('bmi');
    const bmiCategory = document.getElementById('bmi-category');
    
    if (!heightField || !weightField || !bmiField) return;
    
    function calculateBMI() {
        const height = parseFloat(heightField.value);
        const weight = parseFloat(weightField.value);
        
        if (height && weight && height > 0) {
            const bmi = (weight / ((height / 100) ** 2)).toFixed(1);
            bmiField.value = bmi;
            
            // Show BMI category
            const category = getBMICategory(parseFloat(bmi));
            if (bmiCategory) {
                bmiCategory.textContent = `Category: ${category.label}`;
                bmiCategory.className = `form-help ${category.class}`;
            }
        } else {
            bmiField.value = '';
            if (bmiCategory) {
                bmiCategory.textContent = '';
            }
        }
    }
    
    heightField.addEventListener('input', calculateBMI);
    weightField.addEventListener('input', calculateBMI);
}

function getBMICategory(bmi) {
    const ranges = APP_DATA.medicalRanges.bmi;
    for (const [key, range] of Object.entries(ranges)) {
        if (bmi >= range.min && bmi <= range.max) {
            return range;
        }
    }
    return ranges.normal;
}

// Form submission and prediction
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Validate entire form
    const form = event.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    // Validate all required fields
    let isFormValid = true;
    const requiredFields = form.querySelectorAll('[required]');
    
    requiredFields.forEach(field => {
        if (!validateField({ target: field })) {
            isFormValid = false;
        }
    });
    
    if (!isFormValid) {
        showNotification('Please correct the errors in the form', 'error');
        return;
    }
    
    // Show loading
    showLoading(true);
    
    try {
        // Build payload for Flask API
        const payload = {
            age: parseInt(data.age),
            gender: data.gender,
            bmi: parseFloat(data.bmi),
            systolic_bp: parseInt(data.systolic_bp),
            diastolic_bp: parseInt(data.diastolic_bp),
            cholesterol: parseInt(data.cholesterol),
            blood_sugar: parseInt(data.blood_sugar),
            smoking: data.smoking,
            alcohol: data.alcohol,
            activity: data.activity,
            family_history: data.family_history,
            symptoms: data.symptoms || '',
            lifestyle: data.lifestyle_desc || ''
        };

        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        const result = await response.json();
        if (result.status !== 'success') {
            throw new Error(result.error || 'Prediction failed');
        }

        const prediction = {
            predicted_disease: result.predicted_disease,
            confidence: result.confidence,
            all_probabilities: result.all_probabilities
        };

        displayResults(prediction, data);
        showSection('results');
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('An error occurred during prediction. Please ensure the backend is running.', 'error');
    } finally {
        showLoading(false);
    }
}

function handleFormReset() {
    // Clear all validation states
    const form = document.getElementById('prediction-form');
    const fields = form.querySelectorAll('input, select, textarea');
    
    fields.forEach(field => {
        field.classList.remove('error', 'success');
    });
    
    // Clear all help messages
    const helpMessages = form.querySelectorAll('.form-help');
    helpMessages.forEach(msg => {
        if (msg.classList.contains('error') || msg.id !== 'bmi-category') {
            msg.remove();
        }
    });
    
    // Clear BMI category
    const bmiCategory = document.getElementById('bmi-category');
    if (bmiCategory) {
        bmiCategory.textContent = '';
    }
}

// Disease prediction logic
function predictDisease(data) {
    const scores = {};
    
    // Initialize scores
    APP_DATA.diseases.forEach(disease => {
        scores[disease] = 0;
    });
    
    // Calculate scores for each disease based on symptoms and risk factors
    for (const [diseaseName, diseaseData] of Object.entries(DISEASE_PATTERNS)) {
        let score = 0;
        
        // Check symptoms match
        const symptoms = (data.symptoms || '').toLowerCase();
        let symptomMatches = 0;
        diseaseData.symptoms.forEach(symptom => {
            if (symptoms.includes(symptom)) {
                symptomMatches++;
            }
        });
        score += (symptomMatches / diseaseData.symptoms.length) * 0.3;
        
        // Check risk factors
        const riskFactors = diseaseData.riskFactors;
        let riskScore = 0;
        let riskCount = 0;
        
        for (const [factor, thresholds] of Object.entries(riskFactors)) {
            const value = data[factor];
            if (value === undefined) continue;
            
            riskCount++;
            
            if (typeof thresholds === 'object' && 'high' in thresholds) {
                // Numerical threshold
                const numValue = parseFloat(value);
                if (numValue >= thresholds.high) {
                    riskScore += 1;
                } else if (numValue >= thresholds.moderate) {
                    riskScore += 0.5;
                }
            } else if (typeof thresholds === 'object') {
                // Categorical threshold
                riskScore += thresholds[value] || 0;
            }
        }
        
        if (riskCount > 0) {
            score += (riskScore / riskCount) * 0.7;
        }
        
        // Add some randomness to simulate model uncertainty
        score += (Math.random() - 0.5) * 0.1;
        
        scores[diseaseName] = Math.max(0, Math.min(1, score));
    }
    
    // Normalize probabilities
    const totalScore = Object.values(scores).reduce((sum, score) => sum + score, 0);
    if (totalScore > 0) {
        Object.keys(scores).forEach(disease => {
            scores[disease] = scores[disease] / totalScore;
        });
    }
    
    // Find the disease with highest probability
    const predictedDisease = Object.keys(scores).reduce((a, b) => 
        scores[a] > scores[b] ? a : b
    );
    
    const confidence = scores[predictedDisease];
    
    return {
        predicted_disease: predictedDisease,
        confidence: confidence,
        all_probabilities: scores
    };
}

// Results display
function displayResults(prediction, inputData) {
    const resultsContent = document.getElementById('results-content');
    if (!resultsContent) return;
    
    const riskLevel = getRiskLevel(prediction.confidence);
    const recommendations = getRecommendations(prediction.predicted_disease, riskLevel.level);
    
    resultsContent.innerHTML = `
        <div class="results-card">
            <div class="prediction-result">
                <div class="predicted-disease">${prediction.predicted_disease}</div>
                <div class="confidence-score">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
                <div class="risk-level ${riskLevel.class}">${riskLevel.label} Risk</div>
            </div>
            
            <div class="probabilities">
                <h3>Disease Probability Breakdown</h3>
                ${Object.entries(prediction.all_probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .map(([disease, prob]) => `
                        <div class="probability-item">
                            <span>${disease}</span>
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: ${prob * 100}%"></div>
                            </div>
                            <span>${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
            </div>
            
            <div class="recommendations">
                <h3>Medical Recommendations</h3>
                <ul>
                    ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
            
            <div class="form-actions">
                <button class="btn btn--primary" onclick="showSection('prediction')">Make Another Prediction</button>
                <button class="btn btn--secondary" onclick="exportResults()">Export Results</button>
            </div>
        </div>
    `;
}

function getRiskLevel(confidence) {
    if (confidence >= 0.7) {
        return { level: 'high', label: 'High', class: 'risk-high' };
    } else if (confidence >= 0.4) {
        return { level: 'moderate', label: 'Moderate', class: 'risk-moderate' };
    } else {
        return { level: 'low', label: 'Low', class: 'risk-low' };
    }
}

function getRecommendations(disease, riskLevel) {
    const baseRecommendations = {
        "Heart Disease": [
            "Consult with a cardiologist for comprehensive evaluation",
            "Monitor blood pressure and cholesterol levels regularly",
            "Adopt a heart-healthy diet low in saturated fats",
            "Engage in regular moderate exercise as approved by your doctor",
            "Consider stress management techniques"
        ],
        "Diabetes": [
            "Schedule an appointment with an endocrinologist",
            "Monitor blood glucose levels regularly",
            "Follow a balanced, low-glycemic diet",
            "Maintain a healthy weight through diet and exercise",
            "Get regular eye and foot examinations"
        ],
        "Hypertension": [
            "Consult with your primary care physician",
            "Monitor blood pressure regularly at home",
            "Reduce sodium intake in your diet",
            "Increase physical activity gradually",
            "Manage stress through relaxation techniques"
        ],
        "Kidney Disease": [
            "See a nephrologist for specialized care",
            "Monitor kidney function with regular blood tests",
            "Follow a kidney-friendly diet low in protein and phosphorus",
            "Stay well-hydrated but monitor fluid intake",
            "Manage underlying conditions like diabetes and hypertension"
        ],
        "Asthma": [
            "Consult with a pulmonologist or allergist",
            "Identify and avoid asthma triggers",
            "Use prescribed inhaled medications as directed",
            "Monitor peak flow measurements regularly",
            "Develop an asthma action plan with your doctor"
        ]
    };
    
    let recommendations = baseRecommendations[disease] || [
        "Consult with a healthcare professional for proper evaluation",
        "Maintain a healthy lifestyle with balanced diet and exercise",
        "Schedule regular health check-ups and screenings"
    ];
    
    if (riskLevel === 'high') {
        recommendations.unshift("URGENT: Seek immediate medical attention");
    } else if (riskLevel === 'low') {
        recommendations.push("Continue preventive health measures");
    }
    
    return recommendations;
}

// Chart rendering
function renderPerformanceChart() {
    const canvas = document.getElementById('performance-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    const data = APP_DATA.modelPerformance;
    
    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [{
                label: 'Accuracy (%)',
                data: Object.values(data).map(d => d.accuracy),
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C'],
                borderColor: ['#1FB8CD', '#FFC185', '#B4413C'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification--${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--color-${type === 'error' ? 'error' : 'success'});
        color: white;
        padding: var(--space-16);
        border-radius: var(--radius-base);
        z-index: 1001;
        max-width: 300px;
        box-shadow: var(--shadow-lg);
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

function exportResults() {
    const resultsContent = document.getElementById('results-content');
    if (!resultsContent) return;
    
    // Create a simple text export
    const resultText = resultsContent.textContent;
    const blob = new Blob([resultText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'disease_prediction_results.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
    showNotification('Results exported successfully!', 'success');
}

// Global functions for HTML onclick handlers
window.showSection = showSection;
window.exportResults = exportResults;