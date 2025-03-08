import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import torchvision.models as models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Multiply
from tensorflow.keras.applications import DenseNet201, InceptionV3, Xception, ResNet50V2, EfficientNetB0
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tempfile
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# Load M1.pkl model (assumes a scikit-learn model)
def load_m1_model():
    model = joblib.load("M1.pkl")
    return model

def attention_module(x):
    attention = Dense(x.shape[-1], activation='sigmoid')(x)
    return Multiply()([x, attention])

# Define feature extraction models
def create_model(base_model, input_shape, dropout_rate=0.4):
    base = base_model(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base.output)
    x = attention_module(x)  # Apply attention
    return Model(inputs=base.input, outputs=x)


input_shape = (224, 224, 3)
model_DenseNet201 = create_model(DenseNet201, input_shape)
model_InceptionV3 = create_model(InceptionV3, input_shape)
model_Xception = create_model(Xception, input_shape)
model_ResNet50V2 = create_model(ResNet50V2, input_shape)



# Feature extraction in batches
def extract_features_in_batches(model, data, batch_size=32):
    features = []
    for start in range(0, len(data), batch_size):
        end = start + batch_size
        batch_data = data[start:end]
        batch_tensor = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_features = model.predict(batch_tensor)  # Updated to use .predict()
        features.append(batch_features)
    return np.vstack(features)


# Preprocess image for M1
def preprocess_image_for_m1_m2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    img = cv2.resize(img, (224, 224))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # 🔹 Extract features using the same method as during training
    features_DenseNet201 = extract_features_in_batches(model_DenseNet201, img, batch_size=1)
    features_InceptionV3 = extract_features_in_batches(model_InceptionV3, img, batch_size=1)
    features_Xception = extract_features_in_batches(model_Xception, img, batch_size=1)
    features_ResNet50V2 = extract_features_in_batches(model_ResNet50V2, img, batch_size=1)

    # 🔹 Concatenate extracted features (same as training)
    extracted_features = np.concatenate(
        [features_DenseNet201, features_InceptionV3, features_Xception, features_ResNet50V2], axis=1
    )

    return extracted_features  # Ensure shape (1, 8064)


class EfficientTripleAttention(nn.Module):
    """
    ETA Module: Enhances feature representation by capturing cross-dimensional relationships.
    Fix: Adjusts Conv1D inputs for proper processing.
    """
    def __init__(self, channels):
        super(EfficientTripleAttention, self).__init__()

        # 1D convolutions for channel-wise attention
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is at least 3D for Conv1D
        if x.dim() == 4:  # (B, C, H, W) -> (B, C, 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (B, C)
        
        x = x.unsqueeze(-1)  # (B, C, 1) for Conv1D
        
        x_hc = self.conv1(x)
        x_wc = self.conv2(x)
        x_hw = self.conv3(x)

        attention = self.sigmoid(x_hc + x_wc + x_hw)
        return x * attention  # (B, C, 1)




class CropDiseaseModel(nn.Module):
    def __init__(self):
        super(CropDiseaseModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove last classification layer
        self.eta = EfficientTripleAttention(channels=2048)  # Apply ETA module
        self.fc = nn.Linear(2048, 4)  # Custom classifier

    def forward(self, x):
        x = self.backbone(x)  # ResNet-50 output is (B, 2048)
        x = self.eta(x)  # Apply ETA (now correctly handling shape)
        x = x.view(x.size(0), -1)  # Flatten for classifier
        x = self.fc(x)  # Final classification
        return x
    
# Load M3.pth model (assumes a PyTorch model)
def load_m3_model(m3_path):
    model = CropDiseaseModel()  # Instantiate the model
    model.load_state_dict(torch.load(m3_path, map_location=torch.device('cpu')))  # Load weights
    model.eval()  # Set to evaluation mode
    return model



# Preprocess image for M3 (Torch model)
def preprocess_image_for_m3(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def load_m4_model():
    return joblib.load("M4.pkl")

def preprocess_image_for_m4(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)



# Charger les modèles
@st.cache_resource
def load_models():
    m1_model = joblib.load("M1.pkl")
    m2_model = joblib.load("M2.pkl")
    m3_model = load_m3_model("M3.pth")
    m4_model = joblib.load("M4.pkl")
    return m1_model, m2_model, m3_model, m4_model

# Définir les classes
class_names_m4 = ["Common Root Rot", "Fusarium", "Loose Smut", "Septoria"]
class_names_m3 = ["Brown Rust", "Pawdery mildew", "Stem rust", "Yellow rust"]

# Charger les modèles
m1_model, m2_model, m3_model, m4_model = load_models()

# Prédiction avec M1
def predict_m1(image):
    return m1_model.predict(image)[0]

# Prédiction avec M2
def predict_m2(image):
    return m2_model.predict(image)[0]

# Prédiction avec M3
def predict_m3(image):
    with torch.no_grad():
        output = m3_model(image)
        _, predicted = torch.max(output, 1)
    return class_names_m3[int(predicted[0])]

# Prédiction avec M4
def predict_m4(image):
    features = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg').predict(image)
    prediction = m4_model.predict(features)
    return class_names_m4[int(prediction[0])]

# Pipeline de prédiction
def pipeline(image_path):
    image_m1 = preprocess_image_for_m1_m2(image_path)
    pred_m1 = predict_m1(image_m1)

    if pred_m1 == 0:
        return "✅ Healthy"

    pred_m2 = predict_m2(image_m1)
    
    if pred_m2 == 1:
        image_m3 = preprocess_image_for_m3(image_path)
        pred_m3 = predict_m3(image_m3)
        return f"Rust Disease - Model 3 Prediction: {pred_m3}"
    
    image_m4 = preprocess_image_for_m4(image_path)
    pred_m4 = predict_m4(image_m4)
    return f"Not Rust Disease - Model 4 Prediction: {pred_m4}"

# Interface Streamlit
st.title("🌱 Crop Disease Detection")
st.write("Upload an image of a crop leaf to detect potential diseases.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Enregistrer l'image temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_filepath = temp_file.name

    # Charger l'image avec PIL et afficher
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # Exécuter la prédiction
    with st.spinner("Analyzing image..."):
        result = pipeline(temp_filepath)

    # Afficher le résultat
    st.success(f"**Prediction:** {result}")
