import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
from utils_model import entrenar_modelo

# ================================
#  Función principal
# ================================
def predecir_y_entrenar():
    try:
        # Mostrar indicador de carga
        loading_label.config(text="🔄 Analizando datos...", fg="#e67e22")
        root.update()
        
        # Validar campos numéricos
        try:
            edad = int(edad_var.get())
            distancia = int(distancia_var.get())
            entretenimiento = int(entretenimiento_var.get())
            servicio_bordo = int(servicio_bordo_var.get())
            limpieza = int(limpieza_var.get())
            llegada = int(llegada_var.get())
            salida = int(salida_var.get())
        except ValueError:
            loading_label.config(text=" Error en los datos", fg="#e74c3c")
            messagebox.showerror(" Error de Validación", "Por favor, ingrese valores numéricos válidos en todos los campos")
            return

        # Validar rangos
        if not (18 <= edad <= 100):
            loading_label.config(text=" Edad inválida", fg="#e74c3c")
            messagebox.showerror(" Error de Validación", "La edad debe estar entre 18 y 100 años")
            return
            
        if not (0 <= entretenimiento <= 5) or not (0 <= servicio_bordo <= 5) or not (0 <= limpieza <= 5):
            loading_label.config(text=" Valores fuera de rango", fg="#e74c3c")
            messagebox.showerror(" Error de Validación", "Los valores de satisfacción deben estar entre 0 y 5")
            return

        id_modelo = len([f for f in os.listdir("models") if f.startswith("model_")]) + 1
        entrenar_modelo("data/Airline_customer_satisfaction.csv", id_modelo)

        model = joblib.load(f"models/model_{id_modelo}.pkl")
        encoders = joblib.load(f"encoders/encoder_{id_modelo}.pkl")

        entrada = {
            "Age": edad,
            "Type of Travel": tipo_viaje_var.get(),
            "Class": clase_var.get(),
            "Flight Distance": distancia,
            "Inflight entertainment": entretenimiento,
            "On-board service": servicio_bordo,
            "Cleanliness": limpieza,
            "Arrival Delay in Minutes": llegada,
            "Departure Delay in Minutes": salida
        }

        df_input = pd.DataFrame([entrada])
        for col in df_input.columns:
            if col in encoders:
                le = encoders[col]
                df_input[col] = df_input[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        df_input = df_input[model.feature_names_in_]
        pred = model.predict(df_input)[0]
        resultado = "SATISFECHO" if pred == 1 else "INSATISFECHO"
        icono_resultado = "😊" if pred == 1 else "😞"

        df_csv = pd.read_csv("data/Airline_customer_satisfaction.csv")
        match = df_csv[
            (df_csv["Age"] == edad) &
            (df_csv["Type of Travel"] == tipo_viaje_var.get()) &
            (df_csv["Class"] == clase_var.get()) &
            (df_csv["Flight Distance"] == distancia) &
            (df_csv["Inflight entertainment"] == entretenimiento) &
            (df_csv["On-board service"] == servicio_bordo) &
            (df_csv["Cleanliness"] == limpieza) &
            (df_csv["Arrival Delay in Minutes"] == llegada) &
            (df_csv["Departure Delay in Minutes"] == salida)
        ]

        if not match.empty:
            real = match.iloc[0]["satisfaction"]
            coincide = (real == "satisfied" and pred == 1) or (real != "satisfied" and pred == 0)
            precision = "ALTA PRECISIÓN" if coincide else "DIVERGENCIA DETECTADA"
            
            mensaje = f"""
╔══════════════════════════════════════════════════════╗
║                  📊 ANÁLISIS COMPLETO                ║
╠══════════════════════════════════════════════════════╣
║  🎯 PREDICCIÓN: {resultado} {icono_resultado}        ║
║  📈 VALOR REAL: {real.upper():<30}                   ║
║  🔍 PRECISIÓN: {precision:<30}                       ║
║  🤖 MODELO ID: #{id_modelo:<35}                      ║
║  ⚡ CONFIANZA: {'ALTA' if coincide else 'MEDIA':<33} ║
╚══════════════════════════════════════════════════════╝
"""
        else:
            mensaje = f"""
╔══════════════════════════════════════════════════════╗
║                  📊 ANÁLISIS COMPLETO                ║
╠══════════════════════════════════════════════════════╣
║  🎯 PREDICCIÓN: {resultado} {icono_resultado}        ║
║  🤖 MODELO ID: #{id_modelo:<35}                     ║
╚══════════════════════════════════════════════════════╝
"""
        
        loading_label.config(text=" Análisis completado exitosamente", fg="#27ae60")
        messagebox.showinfo("🎉 Resultado del Análisis", mensaje)

    except Exception as e:
        loading_label.config(text=" Error en el sistema", fg="#e74c3c")
        messagebox.showerror(" Error del Sistema", f"Se produjo un error durante el análisis:\n\n{str(e)}")

def limpiar_campos():
    """Función para limpiar todos los campos del formulario"""
    edad_var.set("35")
    tipo_viaje_var.set("Personal Travel")
    clase_var.set("Eco")
    distancia_var.set("1000")
    entretenimiento_var.set("3")
    servicio_bordo_var.set("3")
    limpieza_var.set("3")
    llegada_var.set("15")
    salida_var.set("10")
    
    loading_label.config(text="🚀 Sistema listo para analizar", fg="#3498db")

# ================================
# 🎨 Configuración de la ventana principal
# ================================
root = tk.Tk()
root.title("🌟 Sistema Inteligente de Satisfacción")
root.geometry("800x600")
root.resizable(True, True)
root.minsize(700, 500)

# Fondo claro
root.configure(bg="#ecf0f1")

# ================================
#  Configuración de estilos
# ================================
style = ttk.Style()
style.theme_use("clam")

# Estilos principales
style.configure("MainTitle.TLabel", 
                font=("Helvetica", 20, "bold"), 
                background="#063B70", 
                foreground="#ffffff")

style.configure("FormTitle.TLabel", 
                font=("Segoe UI", 14, "bold"), 
                background="#e74c3c", 
                foreground="#ffffff")

style.configure("FieldLabel.TLabel", 
                font=("Segoe UI", 10, "bold"), 
                background="#ffffff", 
                foreground="#2c3e50")

# Estilos de entrada
style.configure("Bright.TEntry", 
                fieldbackground="#ffffff", 
                foreground="#2c3e50", 
                borderwidth=2, 
                relief="solid",
                insertcolor="#3498db")

style.configure("Bright.TCombobox", 
                fieldbackground="#ffffff", 
                foreground="#2c3e50", 
                borderwidth=2, 
                relief="solid",
                arrowcolor="#3498db")

# Estilos de botones
style.configure("Primary.TButton", 
                font=("Segoe UI", 12, "bold"), 
                foreground="#ffffff", 
                background="#3498db", 
                borderwidth=0, 
                relief="flat",
                padding=(20, 10))

style.configure("Warning.TButton", 
                font=("Segoe UI", 12, "bold"), 
                foreground="#ffffff", 
                background="#f39c12", 
                borderwidth=0, 
                relief="flat",
                padding=(20, 10))

# Efectos hover
style.map("Primary.TButton", 
          background=[("active", "#2980b9"), ("pressed", "#21618c")])
style.map("Warning.TButton", 
          background=[("active", "#d68910"), ("pressed", "#b7950b")])

# ================================
#  Variables del formulario
# ================================
edad_var = tk.StringVar(value="35")
tipo_viaje_var = tk.StringVar(value="Personal Travel")
clase_var = tk.StringVar(value="Eco")
distancia_var = tk.StringVar(value="1000")
entretenimiento_var = tk.StringVar(value="3")
servicio_bordo_var = tk.StringVar(value="3")
limpieza_var = tk.StringVar(value="3")
llegada_var = tk.StringVar(value="15")
salida_var = tk.StringVar(value="10")

# ================================
#  Estructura principal
# ================================

# Header
header_frame = tk.Frame(root, bg="#2c3e50", height=80)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

main_title = tk.Label(header_frame, 
                     text="🌟 Sistema Inteligente de Satisfacción",
                     font=("Helvetica", 18, "bold"),
                     bg="#2c3e50", fg="#ecf0f1")
main_title.pack(pady=15)

subtitle = tk.Label(header_frame,
                   text="SkySense: Tu compañero de vuelo inteligente",
                   font=("Segoe UI", 10),
                   bg="#2c3e50", fg="#bdc3c7")
subtitle.pack()

# Contenedor principal con logo
main_frame = tk.Frame(root, bg="#ecf0f1")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Frame izquierdo para formulario
left_frame = tk.Frame(main_frame, bg="#ecf0f1")
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

# Frame derecho para logo
right_frame = tk.Frame(main_frame, bg="#ffffff", relief="raised", bd=2, width=280)
right_frame.pack(side="right", fill="y", padx=(10, 0))
right_frame.pack_propagate(False)

# Contenedor del logo
logo_container = tk.Frame(right_frame, bg="#ffffff")
logo_container.pack(fill="both", expand=True, padx=20, pady=20)

# Cargar logo desde el backend
def cargar_logo_backend():
    """Función para cargar el logo desde el backend del sistema"""
    try:
        # Ruta del logo en el backend
        logo_path = "imagen/logo.jpeg"  # Ruta específica solicitada
        
        # Verificar si existe el archivo
        if os.path.exists(logo_path):
            # Cargar imagen con PIL
            from PIL import Image, ImageTk
            img = Image.open(logo_path)
            # Redimensionar manteniendo proporción
            img.thumbnail((240, 240), Image.Resampling.LANCZOS)
            logo_img = ImageTk.PhotoImage(img)
            
            # Mostrar imagen en el label
            logo_label = tk.Label(logo_container, image=logo_img, bg="#ffffff")
            logo_label.image = logo_img  # Mantener referencia
            logo_label.pack(expand=True, pady=20)
            
            # Texto informativo debajo del logo
            info_text = tk.Label(logo_container, 
                               text="Sistema SkySense \n\nAuthors: Klismann, Alan\nSebastian, Kevin\n Version: 1.0.0",
                               font=("Segoe UI", 10, "bold"),
                               bg="#ffffff", fg="#000407",
                               justify="center")
            info_text.pack(pady=(10, 0))
            
            return True
        else:
            # Si no existe, mostrar mensaje de error
            error_label = tk.Label(logo_container, 
                                 text=" Logo no encontrado\n\nArchivo: imagen/unsa.png\nVerifique la ruta del archivo",
                                 font=("Segoe UI", 10),
                                 bg="#ffffff", fg="#e74c3c",
                                 justify="center")
            error_label.pack(expand=True)
            return False
            
    except Exception as e:
        # Error al cargar la imagen
        error_label = tk.Label(logo_container, 
                             text=f" Error al cargar logo\n\n{str(e)}",
                             font=("Segoe UI", 10),
                             bg="#ffffff", fg="#e74c3c",
                             justify="center")
        error_label.pack(expand=True)
        return False

# Cargar el logo
cargar_logo_backend()

# Canvas y scrollbar para el formulario
canvas = tk.Canvas(left_frame, bg="#ffffff", highlightthickness=0)
scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#ffffff")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Empaquetar canvas y scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Configurar scrolling con rueda del mouse
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

# Contenedor principal (ahora dentro del scrollable_frame)
main_container = tk.Frame(scrollable_frame, bg="#ffffff", relief="raised", bd=2)
main_container.pack(fill="both", expand=True, padx=5, pady=5)

# Header del formulario
form_header = tk.Frame(main_container, bg="#e74c3c", height=50)
form_header.pack(fill="x")
form_header.pack_propagate(False)

form_title = tk.Label(form_header, 
                     text="📋 FORMULARIO DE DATOS DEL CLIENTE",
                     font=("Segoe UI", 14, "bold"),
                     bg="#e74c3c", fg="#ffffff")
form_title.pack(pady=12)

# Contenedor del formulario
form_container = tk.Frame(main_container, bg="#ffffff")
form_container.pack(fill="both", expand=True, padx=20, pady=20)

# Configuración de campos
campos_config = [
    ("👤 Edad del Cliente", edad_var, None, "Edad en años (18-100)"),
    ("🎯 Tipo de Viaje", tipo_viaje_var, ["Business travel", "Personal Travel"], "Propósito del viaje"),
    ("🏆 Clase de Vuelo", clase_var, ["Eco", "Business", "Eco Plus"], "Categoría del asiento"),
    ("📏 Distancia de Vuelo", distancia_var, None, "Distancia en kilómetros"),
    ("🎬 Entretenimiento", entretenimiento_var, None, "Calificación 0-5"),
    ("🛎️ Servicio a Bordo", servicio_bordo_var, None, "Calificación 0-5"),
    ("🧹 Limpieza", limpieza_var, None, "Calificación 0-5"),
    ("⏰ Retraso Llegada", llegada_var, None, "Minutos de retraso"),
    ("🚀 Retraso Salida", salida_var, None, "Minutos de retraso")
]

# Crear campos en grid
for i, (label, var, options, tooltip) in enumerate(campos_config):
    row = i // 3
    col = i % 3
    
    # Frame para cada campo
    field_frame = tk.Frame(form_container, bg="#f8f9fa", relief="solid", bd=1)
    field_frame.grid(row=row, column=col, padx=8, pady=8, sticky="ew")
    
    # Subframe interno
    inner_frame = tk.Frame(field_frame, bg="#f8f9fa")
    inner_frame.pack(fill="both", expand=True, padx=8, pady=8)
    
    # Label
    label_widget = tk.Label(inner_frame, text=label, 
                           font=("Segoe UI", 10, "bold"),
                           bg="#f8f9fa", fg="#2c3e50")
    label_widget.pack(anchor="w")
    
    # Tooltip
    tooltip_label = tk.Label(inner_frame, text=tooltip,
                            font=("Segoe UI", 8),
                            bg="#f8f9fa", fg="#7f8c8d")
    tooltip_label.pack(anchor="w", pady=(0, 5))
    
    # Widget de entrada
    if options:
        widget = ttk.Combobox(inner_frame, textvariable=var, values=options, 
                             state="readonly", width=15, style="Bright.TCombobox")
    else:
        widget = ttk.Entry(inner_frame, textvariable=var, width=17, style="Bright.TEntry")
    
    widget.pack(fill="x")

# Configurar columnas
for i in range(3):
    form_container.grid_columnconfigure(i, weight=1)

# ================================
#  Estado del sistema
# ================================
status_frame = tk.Frame(main_container, bg="#ecf0f1", height=40)
status_frame.pack(fill="x", pady=(10, 0))
status_frame.pack_propagate(False)

loading_label = tk.Label(status_frame, 
                        text="🚀 Sistema listo para analizar",
                        font=("Segoe UI", 10, "bold"),
                        bg="#ecf0f1", fg="#3498db")
loading_label.pack(pady=12)

# ================================
#  Panel de botones
# ================================
button_panel = tk.Frame(main_container, bg="#ffffff", height=60)
button_panel.pack(fill="x", pady=(10, 0))
button_panel.pack_propagate(False)

button_container = tk.Frame(button_panel, bg="#ffffff")
button_container.pack(expand=True)

# Botón principal
predict_btn = ttk.Button(button_container, 
                        text="🚀 ANALIZAR SATISFACCIÓN", 
                        command=predecir_y_entrenar, 
                        style="Primary.TButton")
predict_btn.pack(side="left", padx=10)

# Botón de limpiar
clear_btn = ttk.Button(button_container, 
                      text="🧹 LIMPIAR DATOS", 
                      command=limpiar_campos, 
                      style="Warning.TButton")
clear_btn.pack(side="left", padx=10)

# ================================
#  Iniciar aplicación
# ================================
if __name__ == "__main__":
    root.mainloop()