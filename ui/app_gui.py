import tkinter as tk
import tkinterdnd2 as tkdnd
from tkinter import messagebox
from PIL import Image, ImageTk
from database.image_database import ImageDatabase
from config import top_k
from utils.image_utils import load_image
import os


class ImageSearchApp(tkdnd.TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("이미지 유사도 검색기")
        self.geometry("800x600")

        self.db = ImageDatabase()
        self.db.build()

        self.label = tk.Label(self, text="검색 이미지 첨부", font=("Arial", 13))
        self.label.pack(pady=10)

        self.canvas_query = tk.Canvas(self, width=224, height=224)
        self.canvas_query.pack(pady=10)

        self.result_frame = tk.Frame(self)
        self.result_frame.pack(pady=10)

        self.drop_target_register(tkdnd.DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        filepath = event.data.strip('{}')
        if not os.path.isfile(filepath):
            messagebox.showerror("오류", "유효한 이미지 파일 첨부하세요.")
            return
        try:
            self.show_query_image(filepath)
            paths, scores, classes = self.db.search(filepath, top_k=top_k)
            predicted_class = classes[0] if classes else "Unknown"
            print(f"예측된 클래스: {predicted_class}")  # 또는 다른 모델에 전달

            self.show_results(paths, scores)
        except Exception as e:
            messagebox.showerror("검색 실패", str(e))

    def show_query_image(self, filepath):
        img = load_image(filepath, size=(224, 224))
        self.query_imgtk = ImageTk.PhotoImage(img)
        self.canvas_query.create_image(0, 0, anchor='nw', image=self.query_imgtk)

    def show_results(self, paths, scores):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        for i, (p, s) in enumerate(zip(paths, scores)):
            # 각 열마다 개별 frame 생성
            col_frame = tk.Frame(self.result_frame)
            col_frame.grid(row=0, column=i, padx=10)

            # 이미지 삽입
            img = load_image(p, size=(224, 224))
            imgtk = ImageTk.PhotoImage(img)
            panel = tk.Label(col_frame, image=imgtk)
            panel.image = imgtk
            panel.pack()

            # 텍스트 삽입
            label = tk.Label(col_frame, text=f"{os.path.basename(p)}\n유사도: {s:.3f}")
            label.pack()