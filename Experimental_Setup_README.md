

````markdown
# Experimental Setup

## Propósito

Este documento describe la etapa inicial del pipeline `main_v3`, cuyo objetivo es organizar, normalizar y auditar la estructura experimental antes de cualquier análisis posterior.

La finalidad de esta fase es establecer una base reproducible para experimentos con múltiples laboratorios, múltiples fechas y múltiples mediciones, manteniendo trazabilidad desde el archivo crudo hasta el registro formal del pipeline.

---

## Estructura experimental soportada

El pipeline actual asume una organización de datos con la forma:

```text
FECHA/
   LAB/
      Raw Data/
         *.csv
````

Ejemplo:

```text
02Mar26/
   Betta/
      Raw Data/
         1medcolor.csv
         2medcolor.csv
         ...
   Epsilon/
      Raw Data/
         1medcolor.csv
         2medcolor.csv
         ...
```

En esta versión del diseño:

* `fecha` es el primer eje organizativo,
* `lab` es el segundo eje organizativo,
* `turno` deja de ser un eje estructural activo,
* las mediciones se tratan como secuencias extensas dentro de cada laboratorio.

---

## Objetivo de esta etapa

La etapa **Experimental Setup** tiene como propósito responder, en orden, las siguientes preguntas:

1. ¿Qué fechas y laboratorios existen realmente en la estructura del experimento?
2. ¿Qué archivos crudos están disponibles?
3. ¿Cómo se traducen esos archivos a una identidad canónica del pipeline?
4. ¿Cómo queda documentada esa traducción?
5. ¿Cómo puede verificarse de forma simple la estructura resultante?

Esta etapa no realiza todavía análisis de calidad, análisis de estabilidad, análisis informacional ni inferencia. Su papel es construir la base estructural y documental del pipeline.

---

## Arquitectura general

Hasta este momento, el flujo implementado en `main_v3.py` corresponde al **Level 0** del pipeline:

```text
Level 0A -> descubrimiento de archivos crudos
Level 0B -> planificación de normalización
Level 0C -> generación de manifest
Level 0D -> resumen estructural
```

El archivo `main_v3.py` funciona como orquestador y coordina los módulos especializados.

---

## Módulos involucrados

### 1. `main_v3.py`

Es el orquestador principal del pipeline en su versión v3.

### Responsabilidades

* detectar fechas válidas,
* detectar laboratorios válidos,
* construir la configuración del experimento,
* ejecutar los módulos de preparación estructural,
* imprimir diagnósticos de control en consola.

### Justificación

Su función es coordinar el flujo completo sin mezclar en un solo lugar toda la lógica especializada de descubrimiento, normalización, auditoría y resumen.

---

### 2. `Modulos/config.py`

Define la clase `ExperimentConfig`, que actúa como contrato formal del experimento.

### Responsabilidades

* almacenar la raíz de origen y la raíz de destino,
* registrar fechas y laboratorios detectados,
* configurar modos de etiquetado (`blind`, `declared`, `external`),
* configurar modos de agrupación,
* validar que la configuración sea consistente,
* mantener `default_shift` como campo legacy opcional.

### Justificación

Centraliza toda la configuración del experimento en una sola estructura validable. Esto evita que el pipeline dependa de variables dispersas y facilita la trazabilidad y la reproducibilidad.

---

### 3. `Modulos/io_dataset.py`

Se encarga del descubrimiento físico de los archivos crudos.

### Responsabilidades

* recorrer la estructura `fecha -> lab -> Raw Data`,
* detectar archivos CSV disponibles,
* construir objetos que representan cada archivo encontrado,
* devolver la colección completa de archivos crudos detectados.

### Justificación

Separa la pregunta “qué archivos existen realmente en disco” de cualquier lógica semántica o analítica posterior. Esta capa permite validar el inventario físico del experimento antes de transformarlo.

---

### 4. `Modulos/normalizer.py`

Construye el plan de normalización del pipeline.

### Responsabilidades

* tomar los archivos crudos descubiertos,
* asignar a cada uno un identificador canónico,
* definir rutas de destino,
* mantener el pipeline en modo blind cuando corresponde,
* preparar la traducción entre nombres físicos y nombres formales.

### MID canónico actual

El identificador formal de cada medición sigue el formato:

```text
{date}_{lab}_{index:03d}
```

Ejemplo:

```text
02Mar26_Betta_010
02Mar26_Epsilon_017
```

### Justificación

Los nombres originales de los archivos no son necesariamente adecuados como identificadores universales del pipeline. El normalizador introduce una nomenclatura estable, consistente y trazable.

---

### 5. `Modulos/manifest.py`

Escribe el archivo `manifest_all.csv`.

### Responsabilidades

* registrar formalmente el resultado del plan de normalización,
* vincular archivo fuente con archivo destino,
* registrar el `MID`,
* conservar metadatos estructurales como fecha y laboratorio,
* mantener compatibilidad con campos legacy cuando sea necesario.

### Justificación

El manifest es la bitácora formal del pipeline. Permite auditar exactamente qué decisión tomó el sistema para cada archivo.

---

### 6. `Modulos/summary_table.py`

Resume la estructura resultante a partir del manifest.

### Responsabilidades

* cargar `manifest_all.csv`,
* agrupar por `(fecha, lab)`,
* imprimir una tabla estructural con el número de archivos por combinación.

### Ejemplo de salida esperada

```text
Fecha   | Laboratorio | Archivos
02Mar26 | Betta       | 27
02Mar26 | Epsilon     | 27
03Mar26 | Betta       | 27
03Mar26 | Epsilon     | 27
04Mar26 | Betta       | 27
04Mar26 | Epsilon     | 27
```

### Justificación

Ofrece una validación humana rápida de la estructura del experimento ya preparada. No analiza las señales, pero sí verifica que la organización del dataset sea consistente.

---

## Flujo implementado

### Level 0A — Raw Dataset Discovery

En esta etapa el pipeline detecta:

* fechas válidas,
* laboratorios válidos,
* archivos CSV crudos disponibles.

### Resultado esperado

Una lista de archivos descubiertos, con fecha, laboratorio y ruta fuente.

---

### Level 0B — Normalization Planning

En esta etapa el pipeline transforma la lista de archivos descubiertos en un plan formal de normalización.

### Resultado esperado

Una lista planificada donde cada archivo tiene:

* ruta fuente,
* ruta destino,
* `MID`,
* `group`,
* `label`.

---

### Level 0C — Manifest Generation

En esta etapa el pipeline escribe el `manifest_all.csv`.

### Resultado esperado

Un registro formal de trazabilidad entre el estado crudo y el estado canónico.

---

### Level 0D — Structural Summary

En esta etapa el pipeline carga el manifest y produce un resumen tabular por fecha y laboratorio.

### Resultado esperado

Una verificación estructural simple del número de archivos organizados por bloque experimental.

---

## Cambios conceptuales de `main_v3`

La transición de `main_v2.py` a `main_v3.py` introduce un cambio importante en la lógica estructural:

### Antes

La organización heredada arrastraba una semántica basada en `turno`.

### Ahora

La nueva organización toma como ejes principales:

* `fecha`
* `lab`

y trata `turno` como un campo legacy opcional, no como un componente activo de la identidad del experimento.

### Justificación del cambio

Dado que ahora las mediciones son más largas y más numerosas, el eje `turno` dejó de ser una unidad útil de organización y añadía fricción al soporte multi-laboratorio.

---

## Estado actual del pipeline

Hasta este punto, el pipeline ya:

* detecta correctamente fechas experimentales,
* detecta correctamente laboratorios,
* descubre archivos crudos,
* genera identificadores canónicos `MID`,
* escribe un manifest de trazabilidad,
* produce un resumen estructural consistente,
* soporta múltiples laboratorios,
* elimina la dependencia estructural de `turno`.

---

## Alcance actual

Esta etapa cubre exclusivamente la preparación estructural del experimento.

Todavía no incluye:

* análisis de columnas,
* tablas analysis-ready,
* conversión a parquet,
* métricas de calidad,
* análisis de estabilidad,
* bins informacionales,
* estados informacionales,
* inferencia secuencial.

---

## Próximos pasos

Las siguientes iteraciones del pipeline extenderán `main_v3.py` hacia nuevas capas del sistema, incluyendo:

1. preparación analysis-ready,
2. detección de roles de columnas,
3. tablas parquet,
4. validación de consistencia,
5. calidad,
6. estabilidad,
7. análisis informacional,
8. inferencia secuencial.

---

## Nota final

Este documento describe únicamente la fase inicial de **Experimental Setup** y su función dentro de la arquitectura general de `main_v3`.

Su propósito es fijar la base estructural, nominal y documental del pipeline antes de cualquier análisis científico posterior.

```

