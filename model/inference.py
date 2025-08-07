from PIL import Image
import torch
from .resnet_embedding import ResNet50Embedding

model = ResNet50Embedding()

def get_embedding(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    input_tensor = model.transforms(image).unsqueeze(0)
    return model(input_tensor).squeeze(0)

def compute_similarity(query_emb, db_emb):
    return torch.nn.functional.cosine_similarity(query_emb, db_emb, dim=0).item()
