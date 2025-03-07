from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import bitsandbytes as bnb

app = Flask(__name__)

# Global variables to store the model, tokenizer, and class mapping
model = None
tokenizer = None
id_to_class = {}

def load_model_and_tokenizer(model_path):
    """
    Load the 4-bit quantized model and tokenizer.

    Args:
        model_path (str): Path to the model directory

    Returns:
        tuple: (model, tokenizer, id_to_class mapping)
    """
    global model, tokenizer, id_to_class

    # Load class mapping
    class_mapping_path = os.path.join(model_path, "class_mapping.json")
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)
        class_mapping = class_mapping["id_to_class"] 
        id_to_class = {int(k): v for k, v in class_mapping.items()}
    else:
        print("Warning: Class mapping file not found. Using empty mapping.")
        id_to_class = {}

    # Configure 4-bit quantization
    bnb_config = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        #device_map="auto"
    )

    return model, tokenizer, id_to_class

def predict_disease(symptoms_text):
    """
    Predict disease based on symptoms text using the 4-bit quantized model.

    Args:
        symptoms_text (str): Description of symptoms

    Returns:
        list: List of (disease, probability) tuples
    """
    global model, tokenizer, id_to_class

    # Check if model is loaded
    if model is None or tokenizer is None:
        return [("Error: Model not loaded", 0)]

    # Prepare input
    inputs = tokenizer(
        symptoms_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(model.device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]

    # Get top 3 predictions with probabilities
    top_probs, top_indices = torch.topk(probabilities, min(3, len(id_to_class)))

    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        disease = id_to_class.get(idx, f"Unknown class {idx}")
        results.append((disease, float(prob) * 100))

    return results

# Define example symptoms for easy testing
example_symptoms = {
    "heart_failure": "Patient presents with swollen legs, increasingly swollen ankles, and worsening shortness of breath. Reports significant fatigue, rapid heartbeat, and difficulty breathing when lying down. Physical examination reveals evidence of ascites.",
    
    "pneumonia": "Patient presents with sweating associated with moderate chest pain. Shows confusion related to intense shortness of breath. Temperature is elevated and patient appears disoriented at times.",
    
    "lung_cancer": "Patient reports a persistent cough for several months and significant weight loss without dieting. Also experiencing neurological symptoms including headaches and weakness. Complains of chest pain and swelling in left leg (possible deep vein thrombosis). Has unexplained fever and facial swelling.",
    
    "copd": "Patient has severe persistent cough and audible wheezing. Observed to use pursed-lip breathing with mild prolonged expiration. Reports symptoms are typically worse in the morning. Has history of smoking.",
    
    "tuberculosis": "Persistent cough for over 3 weeks, occasional blood in sputum, night sweats, weight loss, and fever that worsens in the evening. Patient reports fatigue and chest pain when breathing deeply."
}

@app.route('/')
def home():
    """Render the home page"""
    global model
    model_loaded = model is not None

    return render_template(
        'index.html',
        model_loaded=model_loaded,
        examples=example_symptoms
)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """API endpoint to load the model"""
    try:
        model_path = request.form.get('model_path', 'clinicalbert-4bit-quantized')

        # Load the model
        load_model_and_tokenizer(model_path)

        return jsonify({
            'success': True,
            'message': f'Model loaded successfully! Found {len(id_to_class)} disease classes.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading model: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions"""
    try:
        # Get the symptoms text from the form
        symptoms_text = request.form.get('symptoms', '')

        if not symptoms_text.strip():
            return jsonify({
                'success': False,
                'message': 'Please enter symptoms description.'
            })

        # Make prediction
        results = predict_disease(symptoms_text)

        return jsonify({
            'success': True,
            'predictions': [
                {'disease': disease, 'probability': prob}
                for disease, prob in results
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error making prediction: {str(e)}'
        })

@app.route('/get_example', methods=['GET'])
def get_example():
    """API endpoint to get example symptoms"""
    example_type = request.args.get('type', '')
    example = example_symptoms.get(example_type, '')

    return jsonify({
        'success': True,
        'example': example
    })

# Main entry point
if __name__ == '__main__':
    # Optionally load the model at startup
    try:
        print("Loading model at startup...")
        load_model_and_tokenizer('clinicalbert-4bit-quantized')
        print(f"Model loaded successfully! Found {len(id_to_class)} disease classes.")
    except Exception as e:
        print(f"Could not load model at startup: {str(e)}")
        print("You will need to load the model through the web interface.")

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)