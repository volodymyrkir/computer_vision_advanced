import threading
import tkinter as tk
from tkinter import ttk
from tkinter import IntVar, DoubleVar, BooleanVar

from PIL import ImageTk, Image
import numpy as np
import cv2
from skimage.filters import median
from skimage.morphology import disk

from traceback import print_exc

import asyncio

from image_generator import generate_binary_image_with_border_crossing_lines, add_varying_intensity_noise
from hough import hough, hough_cv, hough_cv_p
from consts import (
    DEFAULT_LINE_WIDTH, DEFAULT_LINE_LENGTH, IMAGE_SIZE,
    BASE_AMOUNT_OF_LINES, INTERSECTION_PROB, OUTPUT_NAME,
    MAX_NOISE_LEVEL
)

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Перетворення Хафа")

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
        self.task = None

    def create_generation_controls(self):
        generation_frame = ttk.LabelFrame(self.params_frame, text="Параметри генерації зображення", padding="5")
        generation_frame.pack(fill="x", pady=(0, 10))

        # Випадаючий список для параметра "Довжина ліній"
        self.line_length = IntVar(value=DEFAULT_LINE_LENGTH)
        line_length_frame = ttk.Frame(generation_frame)
        line_length_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(line_length_frame, text="Довжина відрізків:", width=35).pack(side=tk.LEFT)
        line_length_combo = ttk.Combobox(line_length_frame, textvariable=self.line_length,
                                         values=[20, 50, 100, 150, 250], state="readonly")
        line_length_combo.pack(side=tk.LEFT)
        line_length_combo.set(100)

        # Випадаючий список для параметра "Ширина ліній"
        self.line_width = IntVar(value=DEFAULT_LINE_WIDTH)
        line_width_frame = ttk.Frame(generation_frame)
        line_width_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(line_width_frame, text="Ширина відрізків:", width=35).pack(side=tk.LEFT)
        line_width_combo = ttk.Combobox(line_width_frame, textvariable=self.line_width, values=[0.5, 1.5, 2.5],
                                        state="readonly")
        line_width_combo.pack(side=tk.LEFT)
        line_width_combo.set(1.5)

        # Повзунок для "Кількість ліній:" 
        self.num_lines = IntVar(value=BASE_AMOUNT_OF_LINES)
        num_lines_frame = ttk.Frame(generation_frame)
        num_lines_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(num_lines_frame, text="Кількість відрізків:", width=35).pack(side=tk.LEFT)
        num_lines_scale = ttk.Scale(num_lines_frame, from_=1, to=15, variable=self.num_lines, orient="horizontal",
                                    command=lambda value: self.round_to_nearest_int(value, self.num_lines, num_lines_value_label))
        num_lines_scale.pack(side=tk.LEFT, fill="x", expand=True)
        num_lines_value_label = ttk.Label(num_lines_frame, text=str(BASE_AMOUNT_OF_LINES))  # Додаємо мітку для значення
        num_lines_value_label.pack(side=tk.LEFT, padx=(5, 0))

        # Повзунок для параметра "Імовірність перетину:" 
        self.intersection_prob = DoubleVar(value=INTERSECTION_PROB)
        intersection_prob_frame = ttk.Frame(generation_frame)
        intersection_prob_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(intersection_prob_frame, text="Імовірність перетину:", width=35).pack(side=tk.LEFT)
        intersection_prob_scale = ttk.Scale(
            intersection_prob_frame, from_=0.1, to=1, variable=self.intersection_prob, orient="horizontal",
            command=lambda value: self.round_to_nearest_float(value, self.intersection_prob, intersection_prob_value_label, 0.1)
        )
        intersection_prob_scale.pack(side=tk.LEFT, fill="x", expand=True)
        intersection_prob_value_label = ttk.Label(
            intersection_prob_frame, text=f"{INTERSECTION_PROB:.2f}"
        )  # Додаємо мітку для значення
        intersection_prob_value_label.pack(side=tk.LEFT, padx=(5, 0))

        # Повзунок для параметра "Рівень шуму"
        self.noise_level = DoubleVar(value=0.3)
        noise_level_frame = ttk.Frame(generation_frame)
        noise_level_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(noise_level_frame, text="Рівень шуму:", width=35).pack(side=tk.LEFT)
        noise_level_scale = ttk.Scale(
            noise_level_frame, from_=0.0, to=1.0, variable=self.noise_level, orient="horizontal",
            command=lambda value: self.round_to_nearest_float(value, self.noise_level, noise_level_value_label, 0.05)
        )
        noise_level_scale.pack(side=tk.LEFT, fill="x", expand=True)
        noise_level_value_label = ttk.Label(
            noise_level_frame, text=f"{0.3:.2f}"
        )  # Додаємо мітку для значення
        noise_level_value_label.pack(side=tk.LEFT, padx=(5, 0))

    def create_hough_controls(self):
        hough_frame = ttk.LabelFrame(self.params_frame, text="Параметри перетворення Хафа", padding="5")
        hough_frame.pack(fill="x", pady=(0, 10))

        # Повзунок для "Rho:"
        self.rho = DoubleVar(value=1)
        rho_frame = ttk.Frame(hough_frame)
        rho_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(rho_frame, text="Крок параметру 'Ро':", width=35).pack(side=tk.LEFT)
        rho_scale = ttk.Scale(rho_frame, from_=1, to=10, variable=self.rho, orient="horizontal",
                              command=lambda value: self.round_to_nearest_int(value, self.rho, rho_value_label))
        rho_scale.pack(side=tk.LEFT, fill="x", expand=True)
        rho_value_label = ttk.Label(rho_frame, text="1.00")  # Додаємо мітку для значення
        rho_value_label.pack(side=tk.LEFT, padx=(5, 0))

        # Повзунок для "Theta:"
        self.theta = DoubleVar(value=1)  # Зміна з 180/np.pi на 1
        theta_frame = ttk.Frame(hough_frame)
        theta_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(theta_frame, text="Крок параметру 'Тета':", width=35).pack(side=tk.LEFT)
        theta_scale = ttk.Scale(theta_frame, from_=1, to=180, variable=self.theta, orient="horizontal",
                                command=lambda value: theta_value_label.config(text=f"{float(value):.2f}"))
        theta_scale.pack(side=tk.LEFT, fill="x", expand=True)
        theta_value_label = ttk.Label(theta_frame, text="1.00")  # Додаємо мітку для значення
        theta_value_label.pack(side=tk.LEFT, padx=(5, 0))

        # Повзуок для параметра "Threshold"
        self.threshold = IntVar(value=50)
        threshold_frame = ttk.Frame(hough_frame)
        threshold_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(threshold_frame, text="Порогове значення:", width=35).pack(side=tk.LEFT)
        threshold_scale = ttk.Scale(
            threshold_frame, from_=30, to=200, variable=self.threshold, orient="horizontal",
            command=lambda value: threshold_value_label.config(text=f"{int(float(value))}")
        )
        threshold_scale.pack(side=tk.LEFT, fill="x", expand=True)
        threshold_value_label = ttk.Label(threshold_frame, text=f"{int(float(self.threshold.get()))}")  # Додаємо мітку для значення
        threshold_value_label.pack(side=tk.LEFT, padx=(5, 0))

        # Повзуок для параметра "Min Line Length"
        self.min_line_length = IntVar(value=40)
        min_line_length_frame = ttk.Frame(hough_frame)
        min_line_length_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(min_line_length_frame, text="Мінімальна довжина відрізку:", width=35).pack(side=tk.LEFT)
        min_line_length_scale = ttk.Scale(
            min_line_length_frame, from_=1, to=150, variable=self.min_line_length, orient="horizontal",
            command=lambda value: min_line_length_label.config(text=f"{int(float(value))}")
        )
        min_line_length_scale.pack(side=tk.LEFT, fill="x", expand=True)
        min_line_length_label = ttk.Label(min_line_length_frame, text=f"{int(float(self.min_line_length.get()))}")  # Додаємо мітку для значення
        min_line_length_label.pack(side=tk.LEFT, padx=(5, 0))

        # Повзунок для параметра "Max Line Gap"
        self.max_line_gap = IntVar(value=2)
        max_line_gap_frame = ttk.Frame(hough_frame)
        max_line_gap_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(max_line_gap_frame, text="Максимальний розрив на відрізку:", width=35).pack(side=tk.LEFT)
        max_line_gap_scale = ttk.Scale(
            max_line_gap_frame, from_=1, to=25, variable=self.max_line_gap, orient="horizontal",
            command=lambda value: max_line_gap_label.config(text=f"{int(float(value))}")
        )
        max_line_gap_scale.pack(side=tk.LEFT, fill="x", expand=True)
        max_line_gap_label = ttk.Label(max_line_gap_frame, text=f"{int(float(self.max_line_gap.get()))}")  # Додаємо мітку для значення
        max_line_gap_label.pack(side=tk.LEFT, padx=(5, 0))

    def create_control_buttons(self):
        control_frame = ttk.Frame(self.params_frame, padding=5)
        control_frame.pack(pady=(0,10))

        checkbox_frame = ttk.Frame(control_frame)
        checkbox_frame.pack(fill="x", pady=10)

        self.auto_classify = BooleanVar()
        auto_classify_check = ttk.Checkbutton(checkbox_frame, text="Автоматична класифікація", variable=self.auto_classify, onvalue = True, offvalue = False)
        auto_classify_check.pack(side=tk.LEFT, padx=5, expand=True)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=10)

        ttk.Button(button_frame, text="Згенерувати зображення", command=self.generate_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Класифікувати", command=lambda : asyncio.run(self.process_image())).pack(side=tk.LEFT, padx=5)

    def reset_canvases(self):
        for canvas in self.canvases:
            self.display_image(np.zeros((256, 256), dtype='uint8'), canvas)

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

    def round_to_nearest_int(self, value, variable, label):  # Оновлення для значень повзунка
        rounded_value = round(float(value))
        label.config(text=str(rounded_value))
        variable.set(rounded_value)

    def round_to_nearest_float(self, value, variable, label, step):  # Оновлення для значень повзунка
        rounded_value = round(float(value) / step) * step
        label.config(text=f"{rounded_value:.2f}")
        variable.set(rounded_value)

    def cancel_pending_classification(self):
        if self.task is None:
            return
        
        self.task.cancel()

    def generate_image(self):
        try:
            self.cancel_pending_classification()
            self.reset_canvases()

            # Отримуємо параметри з полів вводу
            line_length = self.line_length.get()
            line_width = self.line_width.get()
            num_lines = self.num_lines.get()
            intersect_prob = self.intersection_prob.get()
            noise_level = self.noise_level.get()

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

            auto_classify = self.auto_classify.get()
            if auto_classify:
                asyncio.run(self.process_image())

        except Exception as e:
            print(f"Помилка генерації картинки: {e}")

    async def get_lines(
            self, input_image, function,
            rho, theta, threshold, overlap_threshold, min_line_length, max_line_gap
        ):
        return function(input_image, rho, theta, threshold, overlap_threshold, min_line_length, max_line_gap)

    async def classify_async(
        self, background_image, input_image, functions,
        rho, theta, threshold, overlap_threshold, min_line_length, max_line_gap
    ):
        tasks = []
        for function in functions:
            line_image = np.copy(background_image)

            task = asyncio.create_task(self.get_lines(
                input_image, function,
                rho, theta, threshold, overlap_threshold, min_line_length, max_line_gap
            ))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        for idx, task in enumerate(tasks):
            lines = task.result()

            if lines is not None:
                for x1, y1, x2, y2 in lines:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            self.display_image(line_image, self.canvases[idx + 1])

    def classify_non_blocking(
            self, background_image, input_image, functions,
            rho, theta, threshold, overlap_threshold, min_line_length, max_line_gap
        ):

        async def set_task():
            task = asyncio.create_task(self.classify_async(
                background_image, input_image, functions,
                rho, theta, threshold, overlap_threshold,
                min_line_length, max_line_gap
            ))

            self.task = task
            await task

        asyncio.run(set_task())

    async def process_image(self):
        if self.noisy_image is None:
            print("Спершу згенеруйте зображення")
            return

        # Отримуємо параметри з полів вводу
        rho = self.rho.get()
        theta = np.pi / (180 / self.theta.get())
        threshold = self.threshold.get()
        min_line_length = self.min_line_length.get()
        max_line_gap = self.max_line_gap.get()

        # Підготовка зображення
        output_median = median(self.noisy_image * 255, disk(1))
        output_median_rgb = cv2.cvtColor(output_median.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        output_median_scaled = output_median // 255

        self.cancel_pending_classification()
        thread = threading.Thread(target=self.classify_non_blocking, 
            args=(output_median_rgb, output_median_scaled, [hough, hough_cv, hough_cv_p],
            rho, theta, threshold, 5, min_line_length, max_line_gap),
            name=f'IO Block Thread'
        )
        
        thread.start()

    def display_image(self, image_array, canvas):
        if image_array is None:
            return
        
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

    root.after(1, app.reset_canvases)
    root.mainloop()
