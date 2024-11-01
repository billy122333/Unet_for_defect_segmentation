import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class MaskSelectorApp:
    def __init__(self, root, folder_image, folder_mask1, folder_mask2, output_folder):
        self.root = root
        self.folder_image = folder_image
        self.folder_mask1 = folder_mask1
        self.folder_mask2 = folder_mask2
        self.output_folder = output_folder
        self.images = os.listdir(folder_image)
        self.current_index = 0

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.button_mask1 = tk.Button(root, text="選擇 Mask1", command=self.select_mask1)
        self.button_mask1.pack(side=tk.LEFT, padx=10)

        self.button_mask2 = tk.Button(root, text="選擇 Mask2", command=self.select_mask2)
        self.button_mask2.pack(side=tk.RIGHT, padx=10)

        self.display_image()

    def display_image(self):
        if self.current_index < len(self.images):
            image_path = os.path.join(self.folder_image, self.images[self.current_index])
            image = Image.open(image_path)
            image.thumbnail((500, 500))  # 調整圖像大小以適應窗口
            photo = ImageTk.PhotoImage(image)

            self.image_label.config(image=photo)
            self.image_label.image = photo
        else:
            self.image_label.config(text="所有圖像均已處理完成。")

    def select_mask1(self):
        self.save_mask(self.folder_mask1)
        self.next_image()

    def select_mask2(self):
        self.save_mask(self.folder_mask2)
        self.next_image()

    def save_mask(self, mask_folder):
        mask_path = os.path.join(mask_folder, self.images[self.current_index])
        output_path = os.path.join(self.output_folder, self.images[self.current_index])
        if os.path.exists(mask_path):
            shutil.copy(mask_path, output_path)

    def next_image(self):
        self.current_index += 1
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Mask 選擇器")

    folder_image = filedialog.askdirectory(title="選擇包含比較圖的資料夾")
    folder_mask1 = filedialog.askdirectory(title="選擇 Mask1 資料夾")
    folder_mask2 = filedialog.askdirectory(title="選擇 Mask2 資料夾")
    output_folder = filedialog.askdirectory(title="選擇輸出資料夾")

    app = MaskSelectorApp(root, folder_image, folder_mask1, folder_mask2, output_folder)
    root.mainloop()
