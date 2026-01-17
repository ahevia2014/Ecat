import cv2
import numpy as np
import os
import shutil
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkintermapview import TkinterMapView
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from PIL.ExifTags import TAGS
from fpdf import FPDF
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

# --- CONFIGURACIÓN TÉCNICA ---
UMBRAL_TEXTURA = 0.75
CAT_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

# --- CLASE REPORTE PDF ---
class ReporteForense(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'eCat - INFORME TÉCNICO DE IDENTIFICACIÓN FELINA', 0, 1, 'C')
        self.ln(5)

# --- CLASE SELECTOR DE MAPA ---
class SelectorMapaEcat:
    def __init__(self):
        self.ventana = tk.Toplevel()
        self.ventana.title("eCat Pro - Ubicación del Peritaje")
        self.ventana.geometry("900x700")
        self.coords = None
        self.map_widget = TkinterMapView(self.ventana, width=900, height=600, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", max_zoom=22)
        self.map_widget.set_position(-33.45, -70.66)
        self.map_widget.add_right_click_menu_command(label="Fijar Punto", command=self.fijar, pass_coords=True)
        tk.Button(self.ventana, text="CONFIRMAR UBICACIÓN", command=self.ventana.destroy).pack(pady=10)
        self.ventana.wait_window()

    def fijar(self, coords):
        self.coords = coords
        self.map_widget.delete_all_marker()
        self.map_widget.set_marker(coords[0], coords[1], text="Punto de Desaparición")

# --- FUNCIONES CORE ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def obtener_color_dominante(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_avg, s_avg, v_avg = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])
    if s_avg < 40 and v_avg > 200: return "BLANCO"
    if s_avg < 40 and v_avg < 50: return "NEGRO"
    if 5 <= h_avg <= 25: return "NARANJA_CAFE"
    return "GRIS_O_TRICOLOR"

def extraer_biometria_nariz(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None, None
    scale = 0.3 if img.shape[1] > 2000 else 0.7
    disp = cv2.resize(img, None, fx=scale, fy=scale)
    roi = cv2.selectROI(f"eCat - SELECCIONE NARIZ: {os.path.basename(img_path)}", disp)
    cv2.destroyAllWindows()
    if roi[2] == 0: return None, None, None
    x, y, w, h = [int(v / scale) for v in roi]
    crop = img[y:y+h, x:x+w]
    norm = cv2.cvtColor(cv2.resize(crop, (100, 100)), cv2.COLOR_BGR2GRAY)
    color = cv2.mean(crop)[:3]
    return norm, color, crop

# --- FLUJO PRINCIPAL eCat ---
def ejecutar_ecat():
    root = tk.Tk()
    root.withdraw()

    # 1. Registro de Datos
    nombre_gato = simpledialog.askstring("eCat", "Nombre del gato:")
    contacto = simpledialog.askstring("eCat", "Nombre del contacto:")
    if not nombre_gato or not contacto: return

    # 2. Mapa y Tiempo
    mapa = SelectorMapaEcat()
    radio = float(simpledialog.askstring("eCat", "Radio de búsqueda (km):", initialvalue="5"))
    f_inicio = datetime.datetime.strptime(simpledialog.askstring("eCat", "Fecha inicio (DD/MM/AAAA):"), "%d/%m/%Y")
    f_fin = datetime.datetime.strptime(simpledialog.askstring("eCat", "Fecha fin (DD/MM/AAAA):"), "%d/%m/%Y")

    # 3. Selección de archivos
    foto_obj_path = filedialog.askopenfilename(title="Seleccione Foto Objetivo")
    carpeta_lib = filedialog.askdirectory(title="Seleccione Carpeta de Biblioteca")
    
    # Análisis Gato Objetivo
    obj_textura, obj_color, obj_crop = extraer_biometria_nariz(foto_obj_path)
    color_cat_obj = obtener_color_dominante(cv2.imread(foto_obj_path))

    # 4. Procesamiento de Biblioteca
    match_list = []
    archivos = [f for f in os.listdir(carpeta_lib) if f.lower().endswith(('.jpg', '.png'))]
    
    for archivo in tqdm(archivos, desc="Analizando"):
        ruta = os.path.join(carpeta_lib, archivo)
        img = cv2.imread(ruta)
        
        # Filtro de Color Dinámico
        if obtener_color_dominante(img) != color_cat_obj:
            dest = os.path.join(carpeta_lib, "DESCARTES_COLOR", color_cat_obj)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(ruta, dest); continue

        # Biometría
        ref_textura, ref_color, ref_crop = extraer_biometria_nariz(ruta)
        if ref_textura is not None:
            score, _ = ssim(obj_textura, ref_textura, full=True)
            if score >= UMBRAL_TEXTURA:
                match_list.append(archivo)
                dest = os.path.join(carpeta_lib, "COINCIDENCIAS")
                os.makedirs(dest, exist_ok=True)
                shutil.copy(ruta, dest)

    # 5. Exportar PDF
    pdf = ReporteForense()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Caso: {nombre_gato} | Contacto: {contacto}", ln=1, align='L')
    pdf.cell(200, 10, txt=f"Coincidencias encontradas: {len(match_list)}", ln=1, align='L')
    pdf.output(os.path.join(carpeta_lib, f"REPORTE_ECAT_{nombre_gato}.pdf"))
    
    messagebox.showinfo("eCat", f"Proceso terminado. Reporte generado en {carpeta_lib}")

if __name__ == "__main__":
    ejecutar_ecat()
