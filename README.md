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
5. **Comparación estadística** de regresión logística (lineal) y SVM con kernel RBF (no lineal) con 10 × 10 *repeated k-fold*, test de Wilcoxon y t-test corregido de Nadeau y Bengio, y evaluación final en un conjunto de prueba separado.
6. **Experimentos complementarios:** TF-IDF frente a Sentence-BERT y aporte de los datos sintéticos (modelo entrenado solo con GoEmotions frente a GoEmotions + sintéticos).
7. **Visualización con t-SNE** del espacio de características, coloreado por clase y por fuente.

## Resultados principales

- Modelo elegido: regresión logística sobre embeddings de Sentence-BERT (`C` ≈ 0,0027, sin selección de características). La SVM con kernel RBF obtuvo un F1-macro algo mayor en el 10 × 10 (0,562 frente a 0,553), pero la diferencia no es significativa según el t-test corregido de Nadeau y Bengio (p = 0,162), así que se eligió el modelo más simple y rápido.
- F1-macro en el conjunto de prueba final: 0,565, coherente con el 0,553 de la validación cruzada 10 × 10. Alegría es la clase mejor reconocida (F1 = 0,689) y neutral la peor (0,443); la confusión más frecuente es entre frustración y enojo.
- Sentence-BERT frente a TF-IDF: +0,059 de F1-macro en el 10 × 10 (0,553 frente a 0,494; IC 95 % [0,039; 0,080]), significativo en ambos tests (p < 0,001, también con el t-test corregido).
- Aporte de los datos sintéticos: mejoran el F1-macro en los mensajes educativos (0,652 frente a 0,495 en el CV; 0,564 frente a 0,472 en la prueba final), pero la mejora global no es significativa (+0,005; t-test corregido p = 0,064). Como los mensajes educativos de prueba los generaron los mismos modelos (GPT y Gemini), esa mejora no demuestra una adaptación a mensajes de estudiantes reales.

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

- Autores: [COMPLETAR: nombres de los autores]
- Curso: MMIA 6012 Aprendizaje Supervisado (profesor Felipe Grijalva Arévalo), Maestría en Inteligencia Artificial, Universidad San Francisco de Quito (USFQ)
