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

Los ejemplos están tomados de los mensajes educativos sintéticos del proyecto.

## Fuentes de datos

- **GoEmotions traducido al español** ([`mrm8488/go_emotions-es-mt`](https://huggingface.co/datasets/mrm8488/go_emotions-es-mt)): comentarios de Reddit etiquetados por personas con 27 emociones más `neutral`, agrupadas en las cinco clases del proyecto. Se descarga automáticamente desde Hugging Face al ejecutar la notebook.
- **Mensajes educativos sintéticos** generados con ChatGPT y Gemini (`data/dataset_sentimientos_educativo_gpt.csv` y `data/dataset_sentimientos_educativo_gemini.csv`), que aportan el dominio educativo. Se pidieron 200 mensajes por clase a cada modelo; ChatGPT entregó 1.000 y Gemini 500.
- **Mensajes escritos por personas** (`data/mensajes_reales.csv`): 15 mensajes (3 por clase) usados solo para una evaluación piloto.

Tras la limpieza y el balanceo, el dataset final tiene 5.354 mensajes: 5.155 de GoEmotions y 199 sintéticos.

<details>
<summary>Prompt usado para generar los mensajes sintéticos (el mismo para ChatGPT y Gemini)</summary>

```text
Quiero que me ayudes a generar un dataset en un csv, esto quiero que sea en un contexto educativo para poder realizar un analisis de sentimientos
Las categorias son: frustracion, confusion, enojo, alegria y neutral
Quiero unos 200 ejemplos por clase te voy a proporcionar ejemplos de como mas o menos deberian quedar los registros
Frustracion:

* "No entiendo nada."
* "Esto es muy difícil para mí / No me sale."
* "Soy malo para esto (o para esta materia)."
* "No soy capaz de hacerlo."
* "Mi mente se quedó en blanco."

Confusion

* "Me perdí en esa parte, ¿podría repetirlo?"
* "Entiendo las palabras por separado, pero no la idea general."
* "¿De dónde salió ese número / ese dato?"
* "Me suena lógico cuando lo explica, pero no sabría cómo hacerlo solo."
* "Siento que me falta una pieza para entender esto."

Enojo

* "¡No entiendo nada, esto no sirve para nada!"
* "¡Ya me cansé, no lo voy a hacer!"
* "Es demasiado difícil, es imposible hacerlo."
* "¿Para qué me va a servir esto en la vida?"

Alegria

* "¡Pasé, no me lo creo!"
* "¡Valió la pena desvelarse estudiando!"
* "¡Saqué diez!".
* "¡Ya la libré!"

Neutral

* "Me da igual sacar buena o mala nota, con aprobar me basta."
* "Esta materia ni me gusta ni me disgusta, simplemente es un requisito más."
* "Vengo a clases porque toca, no porque me interese el tema."
* "No me siento ni estresado ni motivado, solo estoy aquí cumpliendo."
* "Es un día más, nada nuevo."
```

</details>

## Metodología

1. **EDA y limpieza:** limpieza del ruido de la traducción automática, eliminación de duplicados, casi duplicados y variantes de plantilla en los datos sintéticos, filtrado de textos muy cortos y balanceo por submuestreo estratificado.
2. **Extracción de características:** embeddings de oraciones con el modelo Sentence-BERT multilingüe `paraphrase-multilingual-MiniLM-L12-v2` (transfer learning, 384 dimensiones), con TF-IDF como línea base.
3. **Pipelines de scikit-learn** con escalado, selección de características de tipo *filter* (`SelectKBest` con ANOVA F) y clasificador, para evitar el *data leakage*. Un 20 % de los datos se reserva como conjunto de prueba final, que no se usa en la optimización ni en la comparación.
4. **Optimización de hiperparámetros** con Optuna (TPE) y *repeated stratified k-fold*, con curva de validación y curva de aprendizaje.
5. **Comparación estadística** de regresión logística (lineal) y SVM con kernel RBF (no lineal) con 10 × 10 *repeated k-fold*, test de Wilcoxon de rangos con signo y, como análisis complementario, el t-test corregido de Nadeau y Bengio. Evaluación final en el conjunto de prueba.
6. **Experimentos complementarios:** TF-IDF frente a Sentence-BERT, aporte de los datos sintéticos (modelo entrenado solo con GoEmotions frente a GoEmotions + sintéticos) y evaluación piloto con mensajes escritos por personas.
7. **Visualización con t-SNE** del espacio de características, coloreado por clase y por fuente.

## Resultados principales

- **Modelo elegido:** regresión logística sobre embeddings de Sentence-BERT (`C` ≈ 0,0027; la selección de características no mejoró el resultado, por lo que se conservan las 384 dimensiones). En el 10 × 10, la SVM con kernel RBF obtuvo un F1-macro algo mayor (0,562 frente a 0,553). El test de Wilcoxon indica que la diferencia es significativa (p = 0,00001), pero es pequeña (0,9 puntos), y el t-test corregido de Nadeau y Bengio, que tiene en cuenta que las particiones comparten datos de entrenamiento, no la encuentra significativa (p = 0,162). Se eligió la regresión logística por ser más simple, más rápida y más estable: su brecha entre entrenamiento y validación es de 0,07, frente a 0,34 en la SVM.
- **F1-macro en el conjunto de prueba final:** 0,565, coherente con el 0,553 de la validación cruzada 10 × 10. Alegría es la clase mejor reconocida (F1 = 0,689) y neutral la peor (0,443); la confusión más frecuente es entre frustración y enojo.
- **Sentence-BERT frente a TF-IDF:** +0,059 de F1-macro en el 10 × 10 (0,553 frente a 0,494; IC 95 % [0,039; 0,080]), significativo en ambos tests (p < 0,001).
- **Aporte de los datos sintéticos:** mejoran el F1-macro en los mensajes educativos (0,652 frente a 0,495 en el CV; 0,564 frente a 0,472 en la prueba final), mientras que la mejora global es pequeña (+0,005). Como los mensajes educativos de prueba los generaron los mismos modelos (GPT y Gemini), esa mejora no demuestra una adaptación a mensajes de estudiantes reales.
- **Evaluación piloto con mensajes escritos por personas:** con 15 mensajes (3 por clase), el modelo acierta 6 (F1-macro = 0,30; IC 95 % de la exactitud de aproximadamente [0,20; 0,64]). La muestra es demasiado pequeña para conclusiones, pero el modelo no reconoció ningún mensaje de frustración ni de enojo.
- **Experimento complementario (`analisis_balanced.ipynb`):** con un tope de 3.000 ejemplos por clase en lugar de 1.031, el F1-macro de la regresión logística en la misma prueba final sube de 0,565 a 0,595 (diferencia significativa: bootstrap pareado, IC 95 % [0,010; 0,050], y McNemar, p = 0,015).

## Hallazgos metodológicos

- **Ruido de la traducción automática:** cerca del 4 % de los comentarios traducidos tenía cadenas de símbolos (`♪ ♪ ♪`, `* * *`, `¡¡¡¡`) generadas al traducir emojis y signos repetidos; se limpiaron antes de buscar duplicados.
- **Datos sintéticos generados con plantillas:** ChatGPT produjo 1.000 mensajes a partir de solo 100 frases distintas (cada una repetida 10 veces cambiando la materia), y Gemini 500 a partir de otras 100. Muchas de esas frases derivan de los ejemplos incluidos en el prompt. Un filtro de casi duplicados por similitud TF-IDF no las detectaba, porque en mensajes cortos la parte que cambia domina la comparación; se detectaron analizando la estructura (inicios repetidos y enmascaramiento de la materia) y se conservó un mensaje por plantilla.

## Limitaciones

- Los datos etiquetados por personas son comentarios de Reddit traducidos automáticamente, no mensajes de estudiantes.
- Los datos educativos son pocos (199 mensajes tras eliminar las plantillas) y fueron generados por modelos de lenguaje.
- Frustración y enojo son difíciles de separar, tanto en las etiquetas de origen como en el espacio de embeddings.
- El conjunto de desarrollo se usa para optimizar y para comparar los modelos (sin *nested CV*, por su costo); el sesgo se mitiga con el conjunto de prueba final reservado.
- La evaluación con mensajes escritos por personas es solo una prueba piloto de 15 mensajes.

## Estructura del repositorio

```
.
├── analisis.ipynb           # Notebook con todo el proyecto (Secciones 1 a 10)
├── analisis_balanced.ipynb  # Variante experimental con un submuestreo menos agresivo (tope de 3.000 por clase)
├── data/                    # CSV sintéticos (ChatGPT y Gemini) y mensajes escritos por personas
└── README.md
```

## Cómo ejecutarlo

1. Abrir `analisis.ipynb` en Google Colab.
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4* (recomendado para calcular los embeddings; el resto funciona en CPU).
3. Copiar los archivos de `data/` a la carpeta de Google Drive `MyDrive/analisis_sentimientos/data` (o cambiar `CARPETA_DRIVE` en la celda de configuración general). Los dos CSV sintéticos son obligatorios; `mensajes_reales.csv` es opcional: si no está, la evaluación con mensajes escritos por personas (7.3) se omite sin error.
4. Ejecutar todo: *Entorno de ejecución → Ejecutar todo*. La notebook monta Drive al inicio y pedirá autorización.

Los cálculos costosos (embeddings, estudios de Optuna, validación cruzada 10 × 10 y t-SNE) se guardan en la misma carpeta de Drive y se reutilizan en ejecuciones posteriores mientras el dataset no cambie.

**Notebook experimental (`analisis_balanced.ipynb`).** Necesita su propia carpeta de Drive, `MyDrive/analisis_sentimientos_2/data`, con los dos CSV sintéticos, donde guarda todas sus salidas. Además lee, sin modificarla, la carpeta de datos de la notebook principal (dataset final, embeddings y estudio de Optuna de la regresión logística), así que **la notebook principal debe ejecutarse primero**.

## Requisitos

Python 3 con: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `datasets` (Hugging Face), `spacy`, `sentence-transformers`, `torch`, `optuna` y, para la notebook experimental, `statsmodels`. En Colab, las bibliotecas que faltan se instalan desde la propia notebook.
