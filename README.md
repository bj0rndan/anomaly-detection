
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
### Perfilado avanzado del modelo usando Pytorch.profiler con performance.ipynb
```python
# MODIFY THE BELOW PARAMS
fe_model = resnet_feature_extractor(layer2=True) # Load the feature extraction model 
cae_model = FeatCAE(in_channels=512, latent_dim=100)
torch_state = torch.load('/home/jovyan/work/anomaly_detection_demo/modelsave/model.pth')
cae_model.load_state_dict(torch_state['model_state_dict'])
```
- Dentro de FE model es posible definir las capas con las que se van a extraer las características (layer2, layer3 y layer4) para ello es necesario poner la capa que se necesite a True.
- De la configuración elegida anteriormente depende el tamaño de entrada al modelo FeatCAE(in_channels=X)
- Para cambiar de un modelo entrenado a otro solo hace falta cambiar la ruta de torch_state

<details>
<summary> Tabla de in_channels basada en elección de capas: </summary>
  
| Layer2 | Layer3 | Layer4 | in_channels |
| :---: | :---: | :---: | :---: |
| True | False | False |  512 |
| False | True | False | 1024 |
| False | False | True | 2048 |
| True | True | False | 1536 |
| True | False | True | 2560 |
| False | True | True | 3072 |
| True | True | True | 3584 |

</details>


torch.profiler es una herramienta de perfilado avanzada que permite analizar el rendimiento de modelos PyTorch, proporcionando información detallada sobre:

ProfilerActivity.CPU: Operaciones realizadas en CPU
ProfilerActivity.CUDA: Operaciones realizadas en GPU
Tiempos de ejecución
Uso de memoria
Trazas de operadores

Tanto para el extractor de características como para el modelo de extracción de características como del modelo de tipo Autoencoder los resultados se guardan en formato .csv en el directorio de los outputs:
<div align="center">
  <img src="https://github.com/user-attachments/assets/cec2d05c-6016-4164-9f05-aad0cffb619f" alt="Description" width="600">
</div>

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
