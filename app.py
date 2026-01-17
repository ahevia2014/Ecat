import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from fpdf import FPDF
import io

# --- CLASE PARA REPORTE PDF ---
class ReportePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'eCat Pro - Informe de Identificación Felina', 0, 1, 'C')
        self.ln(5)

# --- FUNCIÓN PARA GENERAR PDF ---
def generar_pdf(nombre, dueno, resultados):
    pdf = ReportePDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Nombre del Gato: {nombre}", ln=1)
    pdf.cell(0, 10, f"Dueño: {dueno}", ln=1)
    pdf.cell(0, 10, f"Fecha de Peritaje: {np.datetime64('now')}", ln=1)
    pdf.ln(10)
    pdf.cell(0, 10, "RESULTADOS DEL ESCANEO:", ln=1, b=True)
    
    for res in resultados:
        pdf.cell(0, 10, f"- Archivo: {res[0]} | Similitud: {res[1]:.2%}", ln=1)
    
    return pdf.output(dest='S').encode('latin-1')

# --- INTERFAZ WEB ---
st.set_page_config(page_title="eCat Web Pro", layout="wide")
st.title("🐾 eCat Pro: Buscador Biométrico Web")

# Ficha del Caso
with st.sidebar:
    st.header("📋 Datos del Caso")
    nombre_gato = st.text_input("Nombre del Gato", "Botitas")
    nombre_dueno = st.text_input("Dueño")
    umbral = st.slider("Sensibilidad (Match)", 0.50, 0.95, 0.75)

# Carga de Archivos
col1, col2 = st.columns(2)
with col1:
    foto_obj = st.file_uploader("Foto Patrón (Botitas)", type=["jpg", "png", "jpeg"])
with col2:
    fotos_lib = st.file_uploader("Biblioteca de Búsqueda", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Procesamiento
if st.button("🚀 INICIAR ESCANEO Y GENERAR REPORTE"):
    if foto_obj and fotos_lib:
        # Procesar Objetivo
        img_obj = Image.open(foto_obj).convert('L')
        img_obj_norm = np.array(img_obj.resize((100, 100)))

        matches = []
        st.subheader("Resultados de la Búsqueda:")
        grid = st.columns(4)
        
        for idx, uploaded_file in enumerate(fotos_lib):
            img_ref = Image.open(uploaded_file).convert('L')
            img_ref_norm = np.array(img_ref.resize((100, 100)))
            
            score, _ = ssim(img_obj_norm, img_ref_norm, full=True)

            if score >= umbral:
                matches.append((uploaded_file.name, score))
                with grid[len(matches) % 4]:
                    st.image(uploaded_file, caption=f"{score:.2%}")

        if matches:
            st.success(f"¡Escaneo finalizado! {len(matches)} coincidencias.")
            
            # Botón de Descarga de PDF
            pdf_data = generar_pdf(nombre_gato, nombre_dueno, matches)
            st.download_button(
                label="📥 Descargar Reporte PDF",
                data=pdf_data,
                file_name=f"Reporte_eCat_{nombre_gato}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No se encontraron coincidencias con el umbral seleccionado.")
    else:
        st.error("Faltan archivos para procesar.")
