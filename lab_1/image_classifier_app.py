import tkinter as tk
from tkinter import ttk
from tkinter import IntVar, DoubleVar
from PIL import ImageTk, Image
import numpy as np
import cv2
from skimage.filters import median
from skimage.morphology import disk
from lab_1.image_generator import generate_binary_image_with_border_crossing_lines, add_varying_intensity_noise
from lab_1.hough import hough, hough_cv
from lab_1.consts import (
    DEFAULT_LINE_WIDTH, DEFAULT_LINE_LENGTH, IMAGE_SIZE,
    BASE_AMOUNT_OF_LINES, INTERSECTION_PROB, OUTPUT_NAME,
    MAX_NOISE_LEVEL
)

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Генератор зображень та класифікатор")
        
        # Ініціалізуємо словник для збереження параметрів
        self.generation_entries = {}  # <--- Додаємо цю лінію

        # Створюємо основний контейнер
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Створюємо фрейм для параметрів (праворуч)
        self.params_frame = ttk.Frame(self.main_frame)
        self.params_frame.grid(row=0, column=1, sticky="n", padx=(10, 0))

        # Параметри для генерації зображення
        self.create_generation_controls()
        
        # Параметри для Hough перетворення
        self.create_hough_controls()
        
        # Кнопки керування
        self.create_control_buttons()
        
        # Канваси для відображення зображень (ліворуч)
        self.create_image_canvases()
        
        self.noisy_image = None
        self.binary_image = None

    def create_generation_controls(self):
        generation_frame = ttk.LabelFrame(self.params_frame, text="Параметри генерації зображення", padding="5")
        generation_frame.pack(fill="x", pady=(0, 10))

        # Випадаючий список для параметра "Довжина ліній"
        line_length_var = IntVar(value=DEFAULT_LINE_LENGTH)
        line_length_frame = ttk.Frame(generation_frame)
        line_length_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(line_length_frame, text="Довжина ліній:", width=20).pack(side=tk.LEFT)
        line_length_combo = ttk.Combobox(line_length_frame, textvariable=line_length_var, values=[20, 50, 100, 150, 250], state="readonly")
        line_length_combo.pack(side=tk.LEFT)
        line_length_combo.set(100)
        self.generation_entries["Довжина ліній:"] = line_length_var

        # Випадаючий список для параметра "Ширина ліній"
        line_width_var = IntVar(value=DEFAULT_LINE_WIDTH)
        line_width_frame = ttk.Frame(generation_frame)
        line_width_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(line_width_frame, text="Ширина ліній:", width=20).pack(side=tk.LEFT)
        line_width_combo = ttk.Combobox(line_width_frame, textvariable=line_width_var, values=[0.5, 1.5, 2.5], state="readonly")
        line_width_combo.pack(side=tk.LEFT)
        line_width_combo.set(1.5)
        self.generation_entries["Ширина ліній:"] = line_width_var

        # Повзунок для "Кількість ліній:" 
        num_lines_var = IntVar(value=BASE_AMOUNT_OF_LINES)
        num_lines_frame = ttk.Frame(generation_frame)
        num_lines_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(num_lines_frame, text="Кількість ліній:", width=20).pack(side=tk.LEFT)
        num_lines_scale = ttk.Scale(num_lines_frame, from_=1, to=15, variable=num_lines_var, orient="horizontal",
                                     command=lambda value: self.round_to_nearest_int(value, num_lines_value_label))
        num_lines_scale.pack(side=tk.LEFT, fill="x", expand=True)
        num_lines_value_label = ttk.Label(num_lines_frame, text=str(BASE_AMOUNT_OF_LINES))  # Додаємо мітку для значення
        num_lines_value_label.pack(side=tk.LEFT, padx=(5, 0))
        self.generation_entries["Кількість ліній:"] = num_lines_var

        # Повзунок для "Імовірність перетину:" 
        intersection_prob_var = DoubleVar(value=INTERSECTION_PROB)
        intersection_prob_frame = ttk.Frame(generation_frame)
        intersection_prob_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(intersection_prob_frame, text="Імовірність перетину:", width=20).pack(side=tk.LEFT)
        intersection_prob_scale = ttk.Scale(intersection_prob_frame, from_=0.1, to=1, variable=intersection_prob_var, orient="horizontal", 
                                            command=lambda value: self.round_to_nearest_float(value, intersection_prob_value_label, 0.1))
        intersection_prob_scale.pack(side=tk.LEFT, fill="x", expand=True)
        intersection_prob_value_label = ttk.Label(intersection_prob_frame, text=f"{INTERSECTION_PROB:.2f}")  # Додаємо мітку для значення
        intersection_prob_value_label.pack(side=tk.LEFT, padx=(5, 0))
        self.generation_entries["Імовірність перетину:"] = intersection_prob_var

        # Випадаючий список для параметра "Рівень шуму"
        noise_level_var = DoubleVar(value=0.3)
        noise_level_frame = ttk.Frame(generation_frame)
        noise_level_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(noise_level_frame, text="Рівень шуму:", width=20).pack(side=tk.LEFT)
        noise_level_combo = ttk.Combobox(noise_level_frame, textvariable=noise_level_var, values=[0.1, 0.2, 0.3, 0.4], state="readonly")
        noise_level_combo.pack(side=tk.LEFT)
        noise_level_combo.set(0.3)
        self.generation_entries["Рівень шуму:"] = noise_level_var

    def create_hough_controls(self):
        self.hough_entries = {}  # Додаємо ініціалізацію словника
        hough_frame = ttk.LabelFrame(self.params_frame, text="Параметри трансформації Хофа", padding="5")
        hough_frame.pack(fill="x", pady=(0, 10))

        # Повзунок для "Rho:"
        rho_var = DoubleVar(value=1)
        rho_frame = ttk.Frame(hough_frame)
        rho_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(rho_frame, text="Rho:", width=20).pack(side=tk.LEFT)
        rho_scale = ttk.Scale(rho_frame, from_=1, to=10, variable=rho_var, orient="horizontal", 
                            command=lambda value: self.round_to_nearest_rho(value, rho_value_label))
        rho_scale.pack(side=tk.LEFT, fill="x", expand=True)
        rho_value_label = ttk.Label(rho_frame, text="1.00")  # Додаємо мітку для значення
        rho_value_label.pack(side=tk.LEFT, padx=(5, 0))
        self.hough_entries["Rho"] = rho_var

        # Повзунок для "Theta:"
        theta_var = DoubleVar(value=1)  # Зміна з 180/np.pi на 1
        theta_frame = ttk.Frame(hough_frame)
        theta_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(theta_frame, text="Theta:", width=20).pack(side=tk.LEFT)
        theta_scale = ttk.Scale(theta_frame, from_=1, to=180, variable=theta_var, orient="horizontal", 
                                command=lambda value: theta_value_label.config(text=f"{float(value):.2f}"))
        theta_scale.pack(side=tk.LEFT, fill="x", expand=True)
        theta_value_label = ttk.Label(theta_frame, text="1.00")  # Додаємо мітку для значення
        theta_value_label.pack(side=tk.LEFT, padx=(5, 0))
        self.hough_entries["Theta"] = theta_var

        # Випадаючий список для параметра "Threshold"
        threshold_var = IntVar(value=110)
        threshold_frame = ttk.Frame(hough_frame)
        threshold_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(threshold_frame, text="Threshold:", width=20).pack(side=tk.LEFT)
        threshold_combo = ttk.Combobox(threshold_frame, textvariable=threshold_var, values=[50, 80, 110, 140, 170, 200], state="readonly")
        threshold_combo.pack(side=tk.LEFT)
        threshold_combo.set(110)
        self.hough_entries["Threshold"] = threshold_var

        # Випадаючий список для параметра "Min Line Length"
        min_line_length_var = IntVar(value=70)
        min_line_length_frame = ttk.Frame(hough_frame)
        min_line_length_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(min_line_length_frame, text="Min Line Length:", width=20).pack(side=tk.LEFT)
        min_line_length_combo = ttk.Combobox(min_line_length_frame, textvariable=min_line_length_var, values=[10, 50, 70, 100], state="readonly")
        min_line_length_combo.pack(side=tk.LEFT)
        min_line_length_combo.set(70)
        self.hough_entries["Min Line Length"] = min_line_length_var

        # Випадаючий список для параметра "Max Line Gap"
        max_line_gap_var = IntVar(value=2)
        max_line_gap_frame = ttk.Frame(hough_frame)
        max_line_gap_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(max_line_gap_frame, text="Max Line Gap:", width=20).pack(side=tk.LEFT)
        max_line_gap_combo = ttk.Combobox(max_line_gap_frame, textvariable=max_line_gap_var, values=[1, 2, 5, 10], state="readonly")
        max_line_gap_combo.pack(side=tk.LEFT)
        max_line_gap_combo.set(2)
        self.hough_entries["Max Line Gap"] = max_line_gap_var



    def create_control_buttons(self):
        button_frame = ttk.Frame(self.params_frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Згенерувати зображення", command=self.generate_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Класифікувати", command=self.process_image).pack(side=tk.LEFT, padx=5)

    def create_image_canvases(self):
        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")

        self.canvases = []
        titles = ["Згенероване зображення", "Трансформація Хафа", "Трансформація OpenCV", "Ймовірнісна трансформація"]
        
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                frame = ttk.LabelFrame(canvas_frame, text=titles[idx], padding="5")
                frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                
                canvas = tk.Canvas(frame, width=256, height=256)
                canvas.pack(fill=tk.BOTH, expand=True)
                self.canvases.append(canvas)

        # Налаштування розтягування рядків і стовпців
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(1, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(1, weight=1)

    def round_to_nearest_int(self, value, label): # Оновлення для значень повзунка
        rounded_value = round(float(value))
        label.config(text=str(rounded_value))
        self.generation_entries["Кількість ліній:"].set(rounded_value)

    def round_to_nearest_float(self, value, label, step): # Оновлення для значень повзунка
        rounded_value = round(float(value) / step) * step
        label.config(text=f"{rounded_value:.2f}")
        self.generation_entries["Імовірність перетину:"].set(rounded_value)

    def round_to_nearest_rho(self, value, label): # Оновлення для значень повзунка
        rounded_value = round(float(value))
        label.config(text=str(rounded_value))
        self.hough_entries["Rho"].set(rounded_value)  

    def generate_image(self):
        try:
            # Отримуємо параметри з полів вводу
            line_length = int(self.generation_entries["Довжина ліній:"].get())
            line_width = float(self.generation_entries["Ширина ліній:"].get())
            num_lines = int(self.generation_entries["Кількість ліній:"].get())
            intersect_prob = float(self.generation_entries["Імовірність перетину:"].get())
            noise_level = self.generation_entries["Рівень шуму:"].get()

            # Генеруємо базове зображення
            self.binary_image = generate_binary_image_with_border_crossing_lines(
                line_length=line_length,
                line_width=line_width,
                num_lines=num_lines,
                intersect_prob=intersect_prob
            )
            
            # Додаємо шум
            self.noisy_image = add_varying_intensity_noise(self.binary_image, noise_level)
            
            # Відображаємо згенероване зображення тільки в першому канвасі
            self.display_image(self.noisy_image * 255, self.canvases[0])
            
            # Очищаємо інші канваси
            for i in range(1, 4):
                self.canvases[i].delete("all")
            
        except Exception as e:
            print(f"Помилка генерації картинки: {e}")

    def process_image(self):
        if self.noisy_image is None:
            print("Спершу згенеруйте зображення")
            return

        try:
            # Підготовка зображення
            output_median = median(self.noisy_image * 255, disk(1))
            output_median_rgb = cv2.cvtColor(output_median.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Метод 1: Custom Hough
            line_image = np.copy(output_median_rgb)
            lines = hough(output_median // 255, rho=1, theta=np.pi/180, threshold=60, 
                        min_line_length=10, max_line_gap=5)
            for x1,y1,x2,y2 in lines:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
            self.display_image(line_image, self.canvases[1])

            # Метод 2: Our OpenCV wrapper
            line_image_cv2 = np.copy(output_median_rgb)
            lines = hough_cv(output_median // 255, rho=1, theta=np.pi/180, threshold=60, 
                            min_line_length=10, max_line_gap=5)
            for x1,y1,x2,y2 in lines:
                try:
                    cv2.line(line_image_cv2,(x1,y1),(x2,y2),(255,0,0),2)
                except:
                    print("Invalid line")
            self.display_image(line_image_cv2, self.canvases[2])

            # Метод 3: Pure OpenCV
            line_image_cv2_p = np.copy(output_median_rgb)
            lines = cv2.HoughLinesP(output_median // 255, rho=1, theta=np.pi/180, 
                                threshold=30, minLineLength=5)
            if lines is not None:
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(line_image_cv2_p,(x1,y1),(x2,y2),(255,0,0),2)
            self.display_image(line_image_cv2_p, self.canvases[3])
                
        except Exception as e:
            print(f"Помилка трансформації зображень: {e}")

    def display_image(self, image_array, canvas):
        if image_array is not None:
            # Конвертуємо в формат PIL
            if len(image_array.shape) == 3:
                image_pil = Image.fromarray(image_array.astype('uint8'))
            else:
                image_pil = Image.fromarray(image_array.astype('uint8'), 'L')
            
            # Змінюємо розмір, якщо потрібно
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width > 0 and canvas_height > 0:
                image_pil = image_pil.resize((canvas_width, canvas_height))
            
            # Конвертуємо в PhotoImage і відображаємо
            image_tk = ImageTk.PhotoImage(image_pil)
            canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            canvas.image = image_tk  # Зберігаємо посилання

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
