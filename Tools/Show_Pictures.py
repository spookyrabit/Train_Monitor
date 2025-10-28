import tkinter as tk
from PIL import Image, ImageTk
import csv
import os

PANORAMA_PATH = "TEMP/train_panorama.jpg"
METADATA_PATH = "TEMP/split_metadata.csv"
CARINFO_PATH = "TEMP/Train_car_IDs.csv"

LAYERS = [
    ("TEMP/train_cars", ""),                
    ("TEMP/train_cars_cropped", ""),        
    ("TEMP/train_cars_filtered", "_region1")
]

class PanoramaViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Panorama Multi-Layer + Car Info Viewer")

        # Load all data
        self.panorama = Image.open(PANORAMA_PATH)
        self.pano_w, self.pano_h = self.panorama.size
        self.splits = self.load_metadata()
        self.carinfo = self.load_carinfo()

        # Auto-fit scale
        screen_w = self.winfo_screenwidth()
        self.scale = screen_w / self.pano_w
        self.min_scale = 0.1
        self.max_scale = 5.0

        # Canvas and Scrollbar
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.h_scroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind zoom events
        self.bind("<MouseWheel>", self.on_zoom)
        self.bind("<Button-4>", self.on_zoom)
        self.bind("<Button-5>", self.on_zoom)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Internal refs
        self.images = []
        self.redraw()

    def load_metadata(self):
        splits = []
        with open(METADATA_PATH, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row = {k: (int(v) if k in ['x1','x2','width_px','car_id'] else v) for k, v in row.items()}
                splits.append(row)
        return splits

    def load_carinfo(self):
        info = {}
        if not os.path.exists(CARINFO_PATH):
            return info
        with open(CARINFO_PATH, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row['filename'].strip()
                info[filename] = {
                    "Car_Type": row.get("Car_Type", "").strip(),
                    "ID": row.get("ID", "").strip()
                }
        return info

    def redraw(self):
        self.canvas.delete("all")
        self.images.clear()

        # --- Draw Panorama ---
        scaled_w = int(self.pano_w * self.scale)
        scaled_h = int(self.pano_h * self.scale)
        pano_img = self.panorama.resize((scaled_w, scaled_h), Image.LANCZOS)
        pano_tk = ImageTk.PhotoImage(pano_img)
        self.images.append(pano_tk)
        self.canvas.create_image(0, 0, anchor="nw", image=pano_tk)

        # --- Draw Split Layers ---
        y_offset = scaled_h + 10
        row_spacing = 10
        total_height = scaled_h

        for layer_dir, suffix in LAYERS:
            max_h_this_row = 0
            for split in self.splits:
                x1 = int(split['x1'] * self.scale)
                filename = os.path.splitext(split['filename'])[0] + suffix + ".jpg"
                file_path = os.path.join(layer_dir, os.path.basename(filename))
                if not os.path.exists(file_path):
                    continue

                try:
                    img = Image.open(file_path)
                except Exception:
                    continue

                scaled_w_img = int(img.width * self.scale)
                scaled_h_img = int(img.height * self.scale)
                img = img.resize((scaled_w_img, scaled_h_img), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.images.append(img_tk)
                self.canvas.create_image(x1, y_offset, anchor="nw", image=img_tk)
                max_h_this_row = max(max_h_this_row, scaled_h_img)
            y_offset += max_h_this_row + row_spacing
            total_height += max_h_this_row + row_spacing

        # --- Draw Car Info Text Layer ---
        font_color = "white"
        for split in self.splits:
            x1 = int(split['x1'] * self.scale)
            width = int(split['width_px'] * self.scale)
            center_x = x1 + width // 2
            fname = os.path.basename(split['filename'])
            info = self.carinfo.get(fname, {})
            car_type = info.get("Car_Type", "")
            car_id = info.get("ID", "")
            if not car_type and not car_id:
                continue

            text_y = y_offset + 5  # start a bit below last row
            text_lines = []
            if car_type:
                text_lines.append(car_type)
            if car_id:
                text_lines.append(car_id)
            for i, line in enumerate(text_lines):
                self.canvas.create_text(
                    center_x, text_y + i * 18,
                    text=line,
                    fill=font_color,
                    font=("Arial", 12),
                    anchor="n"
                )

        total_height = text_y + 40
        self.canvas.config(scrollregion=(0, 0, scaled_w, total_height))

    def on_zoom(self, event):
        if event.delta > 0 or event.num == 4:
            new_scale = self.scale * 1.1
        else:
            new_scale = self.scale / 1.1
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        if abs(new_scale - self.scale) > 1e-3:
            self.scale = new_scale
            self.redraw()


if __name__ == "__main__":
    app = PanoramaViewer()
    app.mainloop()
