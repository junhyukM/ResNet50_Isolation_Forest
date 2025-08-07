from PIL import Image

def load_image(filepath, size=None):
    """
    이미지 파일을 열고 RGB로 변환하며, 필요시 크기 조정까지 수행.
    """
    image = Image.open(filepath).convert("RGB")
    if size:
        image = image.resize(size)
    return image
