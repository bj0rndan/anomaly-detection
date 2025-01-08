# Anomaly Detection

## 📋 Descripción
Modelo de detección de anomalías basado en autoencoders convolucionales (CAE), entrenado únicamente con imágenes OK para identificar imágenes KO mediante reconstrucción y análisis de desviaciones. Enfoque semi-supervisado diseñado para aplicaciones de control de calidad.

## 🚀 Inicio Rápido

### Instalación

1. Clonar el repositorio:
```bash
git clone <URL-del-repositorio>
cd anomaly-detection
```

2. Preparar datos en un formato como se demuestra en el esquema (incluidos los nombres de  directorios) para entrenamiento con imágenes OK:

```shell
${BASE_DIR}/train
├── good
│   ├── 000000000.jpg
│   ├── 000000001.jpg
│   ├── 000000002.jpg
│   └── ... (more images)
```
3. Repetir el procedimiento para imágenes KO:
```shell
${BASE_DIR}/test
├── good
│   ├── 000000032.jpg
│   ├── 000000033.jpg
│   ├── 000000034.jpg
│   └── ... (more images)
├── bad
│   ├── 0000000112.jpg
│   ├── 0000000113.jpg
│   ├── 0000000114.jpg
│   └── ... (more images)
```
❗Es importante añadir imágenes OK para el conjunto de entrenamiento para asegurarse de que el modelo los identifica de los imágenes KO

4. Modificar el config.yaml con los datos necesarios:
```yaml
paths:
  train_path: "ruta_del_conjunto_train"
  test_path: "ruta_del_conjunto_test"
  output_path: "ruta_para_guardar_outputs_y_gráficas"
  model_save_path: "ruta_para_guardar_modelo"
```

## 💻 Uso

### Ejecución del script para entrenamiento y validación
```bash
python /anomaly-detection/main.py --config="ruta_del_config" --mode=full --test_path="ruta_del_conjunto_test" --model_path="ruta_para_guardar_modelo" --output_dir="ruta_para_guardar_outputs_y_gráficas"
```

## 🛠️ Funcionalidades Principales

- Detección de anomalías mediante autoencoder convolucional
- Extracción de características usando ResNet50 pre-entrenado
- Entrenamiento configurable mediante archivo YAML
- Visualización de mapas de calor para anomalías
- Pipeline completo de entrenamiento y evaluación
- Soporte para GPU y CPU

## 📊 Visualización de Resultados
Los resultados se guardan automáticamente, incluyendo:
- Curvas de pérdida de entrenamiento y validación
- Curva ROC y métricas de evaluación
- Mapas de calor de anomalías detectadas
- Visualización de imágenes con superposición de detecciones

## 🔧 Características Técnicas
- Backbone: ResNet50 con extracción de características multicapa
- Detector: Autoencoder convolucional personalizado
- Umbral adaptativo basado en estadísticas de reconstrucción
- Soporte para procesamiento por lotes
- Visualización personalizable de resultados


## 🖥️ Interfaz gráfica con Gradio 

  Se ejecuta mediante:
```bash
python gradio-app.py
```
La interfaz da la posibilidad de cargar imágenes y entrenar un modelo con las imágenes OK:
![image](https://github.com/user-attachments/assets/03dbbe25-a9a3-4c39-8a88-3f86b8c74bc3)

Además, se puede realizar inferencia y devolver predicciones sobre imágenes KO con sus mapas de calor asociadas:
![image](https://github.com/user-attachments/assets/d877be12-bc47-4f9f-9383-3b3c3423ef39)
</details>

