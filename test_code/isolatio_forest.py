import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageTk
import tkinterdnd2 as tkdnd
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import IsolationForest
import numpy as np

# --- 모델 및 임베딩 세팅 ---
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

class ResNet50Embedding(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

embedding_model = ResNet50Embedding(model)
preprocess = weights.transforms()

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = embedding_model(input_tensor).squeeze(0)
    return embedding

# --- 이미지 데이터베이스 구축 ---
image_dir = "./images"
image_paths = []
image_embeddings = []

def build_database():
    global image_paths, image_embeddings
    image_paths = []
    image_embeddings = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
            path = os.path.join(image_dir, fname)
            try:
                emb = get_embedding(path)
                image_paths.append(path)
                image_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

build_database()

# --- 이상치 탐지를 위한 Isolation Forest 학습 ---
isolation_model = None

def train_isolation_forest():
    global isolation_model
    if not image_embeddings:
        return
    X = torch.stack(image_embeddings).numpy()
    isolation_model = IsolationForest(contamination=0.01, random_state=42)
    isolation_model.fit(X)

def detect_outlier(embedding):
    if isolation_model is None:
        return None
    emb_np = embedding.numpy().reshape(1, -1)
    pred = isolation_model.predict(emb_np)  # 1: 정상, -1: 이상치
    return pred[0]

train_isolation_forest()

# --- 유사 이미지 검색 ---
def find_similar_images(query_image_path, top_k=3):
    if not image_embeddings:
        return [], [], None
    query_emb = get_embedding(query_image_path)
    similarities = []
    for emb in image_embeddings:
        sim = torch.nn.functional.cosine_similarity(query_emb, emb, dim=0)
        similarities.append(sim.item())
    similarities = torch.tensor(similarities)
    k = min(top_k, len(similarities))
    top_k_idx = torch.topk(similarities, k=k).indices
    top_paths = [image_paths[i] for i in top_k_idx]
    top_scores = [similarities[i].item() for i in top_k_idx]

    # 이상치 판별
    outlier_flag = detect_outlier(query_emb)
    return top_paths, top_scores, outlier_flag

# --- GUI 구현 ---
class ImageSearchApp(tkdnd.TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("이미지 유사도 검색기 (드래그 앤 드롭)")
        self.geometry("800x600")

        self.label = tk.Label(self, text="검색할 이미지를 드래그 앤 드롭 하세요", font=("Arial", 16))
        self.label.pack(pady=10)

        self.canvas_query = tk.Canvas(self, width=224, height=224)
        self.canvas_query.pack(pady=10)

        self.result_frame = tk.Frame(self)
        self.result_frame.pack(pady=10)

        self.status_label = tk.Label(self, text="", font=("Arial", 14), fg="red")
        self.status_label.pack(pady=5)

        self.drop_target_register(tkdnd.DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        filepath = event.data.strip('{}')
        if not os.path.isfile(filepath):
            messagebox.showerror("오류", "유효한 이미지 파일을 드롭하세요")
            return
        try:
            self.show_query_image(filepath)
            top_paths, top_scores, outlier_flag = find_similar_images(filepath, top_k=3)
            self.show_results(top_paths, top_scores, outlier_flag)
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def show_query_image(self, filepath):
        img = Image.open(filepath).resize((224, 224))
        self.query_imgtk = ImageTk.PhotoImage(img)
        self.canvas_query.create_image(0, 0, anchor='nw', image=self.query_imgtk)

    def show_results(self, paths, scores, outlier_flag):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        for i, (p, s) in enumerate(zip(paths, scores)):
            img = Image.open(p).resize((150, 150))
            imgtk = ImageTk.PhotoImage(img)
            panel = tk.Label(self.result_frame, image=imgtk)
            panel.image = imgtk
            panel.grid(row=0, column=i, padx=5)
            label = tk.Label(self.result_frame, text=f"{os.path.basename(p)}\n유사도: {s:.3f}")
            label.grid(row=1, column=i, padx=5)

        if outlier_flag == -1:
            self.status_label.config(text="⚠ 이 이미지는 이상치로 탐지되었습니다.", fg="red")
        elif outlier_flag == 1:
            self.status_label.config(text="✔ 이 이미지는 정상으로 판단됩니다.", fg="green")
        else:
            self.status_label.config(text="이상치 판별 정보를 불러올 수 없습니다.", fg="gray")

if __name__ == "__main__":
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print(f"❗'{image_dir}' 폴더에 이미지가 없습니다. 이미지 추가 후 실행하세요.")
        exit()

    build_database()
    train_isolation_forest()
    app = ImageSearchApp()
    app.mainloop()