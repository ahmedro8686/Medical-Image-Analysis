import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2

LABELS = ["Normal", "Pneumonia", "COVID-19", "Other Diseases"]

def load_model():
    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))

        try:
            model.load_state_dict(torch.load("models/xray_model.pth", map_location='cpu'))
            print("Trained model loaded successfully")
        except:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            print("Using ImageNet weights")

        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_with_heatmap(model, image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        label = LABELS[pred_class.item()]

    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model, input_tensor, pred_class.item())
    return label, confidence.item(), heatmap

# ---------------------------
# Grad-CAM
# ---------------------------
def grad_cam(model, input_tensor, class_idx):
    model.eval()
    gradients = []
    activations = []

    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2
    target_layer.register_forward_hook(save_activations)
    target_layer.register_backward_hook(save_gradients)

    output = model(input_tensor)
    model.zero_grad()
    loss = output[0, class_idx]
    loss.backward()

    gradient = gradients[0].detach().numpy()[0]
    activation = activations[0].detach().numpy()[0]

    weights = np.mean(gradient, axis=(1,2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam)+1e-8)
    return cam



