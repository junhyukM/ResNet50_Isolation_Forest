import os
from model.inference import get_embedding, compute_similarity
from config import image_dir
import torch

class ImageDatabase:
    def __init__(self):
        self.image_paths = []
        self.embeddings = []

    def build(self):
        self.image_paths.clear()
        self.embeddings.clear()

        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
                path = os.path.join(image_dir, fname)
                try:
                    emb = get_embedding(path)
                    self.image_paths.append(path)
                    self.embeddings.append(emb)
                except Exception as e:
                    print(f"[Error] {fname}: {e}")

    def search(self, query_path, top_k=3):
        if not self.embeddings:
            return [], [], []

        query_emb = get_embedding(query_path)
        scores = [compute_similarity(query_emb, emb) for emb in self.embeddings]
        topk = torch.tensor(scores).topk(k=min(top_k, len(scores)))

        top_paths = [self.image_paths[i] for i in topk.indices]
        top_scores = [scores[i] for i in topk.indices]

        # 클래스 추출 (예: 파일명에서)
        top_classes = [os.path.basename(p).split('.')[0] for p in top_paths]

        return top_paths, top_scores, top_classes

