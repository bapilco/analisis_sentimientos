# Detección de emociones en mensajes de estudiantes para un tutor virtual

Clasificador supervisado que identifica el estado emocional de un estudiante a partir del mensaje que escribe a un tutor virtual. Es un componente pensado para que, en trabajos futuros, el tutor adapte su estrategia pedagógica, por ejemplo simplificando la explicación cuando detecta frustración o confusión.

## Objetivo y clases

A partir de un mensaje en español, predecir una de cinco clases relevantes para el contexto educativo:

| Clase | Ejemplo |
|---|---|
| `frustracion` | "Me cuesta muchísimo seguir el ritmo de la lectura de historia." |
| `confusion` | "Me perdí en esa última parte, ¿podría repetirlo por favor?" |
| `enojo` | "¡Estoy harto de esta plataforma que siempre falla al subir el taller!" |
| `alegria` | "¡Pasé física, todavía no me lo creo!" |
| `neutral` | "Es una asignatura obligatoria del plan de estudios." |

## Fuentes de datos

- **GoEmotions traducido al español** ([`mrm8488/go_emotions-es-mt`](https://huggingface.co/datasets/mrm8488/go_emotions-es-mt)): comentarios de Reddit etiquetados por personas con 27 emociones más `neutral`, agrupadas en las cinco clases del proyecto.
- **Mensajes educativos sintéticos** generados con ChatGPT y Gemini (`data/dataset_sentimientos_educativo_gpt.csv` y `data/dataset_sentimientos_educativo_gemini.csv`), que aportan el dominio educativo.

Tras la limpieza y el balanceo, el dataset final tiene 5.354 mensajes: 5.155 de GoEmotions y 199 sintéticos.

## Metodología

1. **EDA y limpieza:** limpieza del ruido de la traducción automática, eliminación de duplicados, casi duplicados y variantes de plantilla en los datos sintéticos, filtrado de textos muy cortos y balanceo por submuestreo estratificado.
2. **Extracción de características:** embeddings de oraciones con el modelo Sentence-BERT multilingüe `paraphrase-multilingual-MiniLM-L12-v2` (transfer learning, 384 dimensiones), con TF-IDF como línea base.
3. **Pipelines de scikit-learn** con escalado, selección de características de tipo *filter* (`SelectKBest` con ANOVA F) y clasificador, para evitar el *data leakage*.
4. **Optimización de hiperparámetros** con Optuna (TPE) y *repeated stratified k-fold*, con curva de validación y curva de aprendizaje.
5. **Comparación estadística** de regresión logística y SVM con 10 × 10 *repeated k-fold*, test de Wilcoxon y t-test corregido de Nadeau y Bengio, y evaluación final en un conjunto de prueba separado.
6. **Experimentos complementarios:** TF-IDF frente a Sentence-BERT y aporte de los datos sintéticos (modelo entrenado solo con GoEmotions frente a GoEmotions + sintéticos).
7. **Visualización con t-SNE** del espacio de características, coloreado por clase y por fuente.

## Resultados principales

- Modelo elegido: [COMPLETAR]
- F1-macro en el conjunto de prueba final: [COMPLETAR]
- Sentence-BERT frente a TF-IDF: [COMPLETAR]
- Aporte de los datos sintéticos: [COMPLETAR]

## Estructura del repositorio

```
.
├── analisis.ipynb      # Notebook con todo el proyecto (Secciones 1 a 10)
├── data/               # CSV sintéticos y splits de GoEmotions
├── instructions/       # Instrucciones usadas para construir cada sección de la notebook
└── README.md
```

## Cómo ejecutarlo

1. Abrir `analisis.ipynb` en Google Colab.
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4* (recomendado para calcular los embeddings; el resto funciona en CPU).
3. Copiar los dos CSV sintéticos a la carpeta de Google Drive `MyDrive/analisis_sentimientos/data` (o cambiar `CARPETA_DRIVE` en la celda de configuración general).
4. Ejecutar todo: *Entorno de ejecución → Ejecutar todo*. La notebook monta Drive al inicio y pedirá autorización.

Los cálculos costosos (embeddings, estudios de Optuna, validación cruzada 10 × 10 y t-SNE) se guardan en la misma carpeta de Drive y se reutilizan en ejecuciones posteriores mientras el dataset no cambie.

## Requisitos

Python 3 con: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `datasets` (Hugging Face), `spacy`, `sentence-transformers`, `torch` y `optuna`. En Colab, las bibliotecas que faltan se instalan desde la propia notebook.

## Autores y curso

- Autores: [COMPLETAR]
- Curso: [COMPLETAR: nombre del curso de Aprendizaje Supervisado], Universidad San Francisco de Quito (USFQ)
