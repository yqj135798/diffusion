import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import List


def show_images(images: List[Image.Image], nrow: int = 3):
    """
    显示生成的图片
    """
    plt.figure(figsize=(8, 8))
    for i, img in enumerate(images):
        plt.subplot(nrow, nrow, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.show()


def save_images(images: List[Image.Image], path: str, nrow: int = 3):
    """
    保存图片到文件
    """
    w, h = images[0].size
    grid = Image.new('L', (w * nrow, h * nrow))

    for i, img in enumerate(images):
        grid.paste(img, ((i % nrow) * w, (i // nrow) * h))

    grid.save(path)
    print(f"图片已保存到: {path}")