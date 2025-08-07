import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import normalize, cosine_similarity


# ResNet50 불러오기
resnet = models.resnet50(pretrained=True)

# 마지막 fully connected layer 제거 → feature extractor만 사용
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # avgpool까지 포함
resnet.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 입력 크기 맞춤
    transforms.ToTensor(),          # PIL → Tensor
    transforms.Normalize(           # ImageNet 학습 시 사용된 값
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_resnet_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        features = resnet(input_tensor)       # (1, 2048, 1, 1)
        flattened = features.view(1, -1)      # (1, 2048)
        embedding = normalize(flattened, p=2, dim=1)  # 정규화
    return embedding


# 두 이미지 임베딩 추출
emb1 = get_resnet_embedding("cat1.jpg")
emb2 = get_resnet_embedding("person.jpg")

# Cosine similarity 계산
sim = cosine_similarity(emb1, emb2)
print(f"Cosine similarity: {sim.item():.4f}")