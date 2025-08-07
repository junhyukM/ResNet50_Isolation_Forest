import os
from config import image_dir
from ui.app_gui import ImageSearchApp

if __name__ == "__main__":
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print(f"[!] '{image_dir}' 폴더에 이미지가 없습니다. 이미지 추가 후 실행하세요.")
        exit()

    app = ImageSearchApp()
    app.mainloop()
