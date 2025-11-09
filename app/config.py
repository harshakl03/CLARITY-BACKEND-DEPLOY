import os
import torch
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Model paths (Docker container paths)
    DENSENET121_PATH: str = "/app/models/densenet121.pth"
    RESNET152_PATH: str = "/app/models/resnet152.pth"
    
    IMAGE_SIZE = 224
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Gemini API (read from environment on Render)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
    
    LABEL_COLS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
        'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ]
    
    CONFIDENCE_THRESHOLD = 0.5
    UPLOAD_DIR = "/app/uploads"      # Docker absolute path
    OUTPUT_DIR = "/app/outputs"      # Docker absolute path
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    DENSENET121_LAYERS = [
        "features.denseblock1.denselayer1",
        "features.denseblock1.denselayer3",
        "features.denseblock1.denselayer6",
        "features.denseblock2.denselayer1",
        "features.denseblock2.denselayer6",
        "features.denseblock2.denselayer12",
        "features.denseblock3.denselayer1",
        "features.denseblock3.denselayer8",
        "features.denseblock3.denselayer16",
        "features.denseblock3.denselayer24",
        "features.denseblock4.denselayer1",
        "features.denseblock4.denselayer8",
        "features.denseblock4.denselayer16",
    ]
    
    RESNET152_LAYERS = [
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.3",
        "layer2.7",
        "layer3.0",
        "layer3.11",
        "layer3.23",
        "layer3.35",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ]
    
    HEATMAP_METHODS_DENSENET = [
        "gradcam_pp",
        "layercam",
        "scorecam",
        "integrated_gradients",
        "saliency",
    ]
    
    HEATMAP_METHODS_RESNET = [
        "gradcam_pp",
        "layercam",
        "scorecam",
        "integrated_gradients",
        "saliency",
    ]

settings = Settings()

# Create directories in Docker container
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
