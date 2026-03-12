
````md
# Analysis Ready Data Preparation

## Propósito

Este documento describe la etapa **Analysis Ready Data Preparation** del pipeline experimental `main_v3`.

Su objetivo es transformar el conjunto de archivos crudos descubiertos en **tablas estructuradas, consistentes, auditables y listas para análisis**, sin modificar la fuente original de datos.

Esta etapa trabaja **después de Experimental Setup / Level 0** y constituye la primera capa formal de preparación tabular del experimento.

---

## Lugar dentro del pipeline

La secuencia general actual del pipeline es:

```text
Level 0A  Raw dataset discovery
Level 0B  Normalization planning
Level 0C  Manifest generation
Level 0D  Structural summary

Level 1A  Analysis-ready catalog build
Level 1B  CSV dialect inspection
Level 1C  Column role detection
Level 1D  Measurement time bounds
Level 1E  Analysis table build
Level 1F  Analysis-ready schema validation
````

---

## Contrato estructural activo

La arquitectura vigente de `main_v3` está basada en:

```text
fecha -> lab -> medición
```

Los ejes estructurales activos son:

* `fecha`
* `lab`
* `archivo` / índice de medición

El identificador canónico actual es:

```text
{fecha}_{lab}_{index:03d}
```

Ejemplos:

```text
02Mar26_Betta_010
02Mar26_Epsilon_017
04Mar26_Betta_023
```

Importante:

* `turno` ya **no** es eje estructural activo
* `jornada` ya **no** es eje estructural activo
* ambos pueden conservarse solo como campos **legacy** o informativos cuando sea necesario

---

## Filosofía de diseño

La etapa Analysis Ready sigue los mismos principios del resto de `main_v3`:

### 1. Separación estricta de responsabilidades

Cada módulo responde una sola pregunta.

### 2. No modificar datos fuente

Los CSV crudos en `source_root` se consideran la referencia física del experimento y no deben alterarse durante esta etapa.

### 3. Trazabilidad completa

Cada decisión relevante del pipeline debe quedar registrada mediante artefactos intermedios.

### 4. Reproducibilidad

La misma entrada estructural debe generar los mismos resultados tabulares.

### 5. Compatibilidad controlada con legado

Campos como `turno`, `jornada`, `color` o `etiqueta` pueden mantenerse cuando aportan compatibilidad, pero no deben gobernar el diseño principal del pipeline.

---

## Objetivo de la etapa Analysis Ready

Esta etapa convierte cada medición CSV en una tabla con un esquema explícito y uniforme, donde ya estén identificados:

* archivo físico de origen
* dialecto CSV
* columnas temporales
* columnas de canales
* límites temporales útiles
* tabla final lista para análisis
* consistencia de esquema entre archivos

En otras palabras, esta etapa responde:

```text
¿Qué archivo real se lee?
¿Cómo debe interpretarse?
¿Qué columnas significan tiempo y qué columnas significan señal?
¿Cuál es la forma tabular estable para el análisis posterior?
```

---

# Módulos de Analysis Ready

## 1. `analysis_ready_prep.py`

### Función

Construye el catálogo base de archivos para análisis y realiza una inspección superficial del formato CSV.

### Responsabilidades

* leer `manifest_all.csv`
* resolver la ruta física real del CSV en `source_root`
* generar `analysis_catalog.csv`
* inspeccionar delimitador, codificación y consistencia de columnas
* generar `dialect_report.json`

### Artefactos generados

* `analysis_catalog.csv`
* `dialect_report.json`

### Pregunta que resuelve

```text
¿Qué archivo físico corresponde a cada medición y se puede leer correctamente?
```

---

## 2. `column_role_detection.py`

### Función

Detecta los roles semánticos principales de las columnas de cada CSV.

### Responsabilidades

* detectar columna de tiempo del sistema (`t_sys`)
* detectar columna de tiempo relativo (`t_rel`)
* detectar columnas numéricas de señal
* asignar nombres canónicos a los canales

### Artefacto generado

* `column_roles.json`

### Pregunta que resuelve

```text
¿Qué significa cada columna del CSV?
```

---

## 3. `measurement_time_bounds.py`

### Función

Calcula límites temporales útiles para cada medición a partir de las columnas detectadas.

### Responsabilidades

* leer `analysis_catalog.csv`
* leer `column_roles.json`
* localizar tiempo inicial y final
* resumir ventana temporal utilizable por medición

### Artefacto generado

* `measurement_time_bounds.json`

### Pregunta que resuelve

```text
¿Cuál es el intervalo temporal efectivo de cada medición?
```

---

## 4. `analysis_table_io.py`

### Función

Es la capa de lectura y transformación tabular de bajo nivel.

### Responsabilidades

* leer CSV con el dialecto correcto
* seleccionar columnas por índice
* renombrar a nombres canónicos
* construir DataFrames listos para escritura
* manejar valores centinela y detalles de parsing

### Artefacto

No necesariamente genera un artefacto final por sí solo, pero es una dependencia operacional clave del builder tabular.

### Pregunta que resuelve

```text
¿Cómo se convierte un CSV crudo en una tabla estructurada y consistente?
```

---

## 5. `analysis_table_builder.py`

### Función

Construye las tablas finales **Analysis Ready** para cada medición.

### Responsabilidades

* leer `analysis_catalog.csv`
* usar `column_roles.json`
* usar `analysis_table_io.py`
* generar tablas limpias por medición
* escribir archivos Parquet
* registrar acciones del proceso

### Artefactos generados

* árbol `Analysis Ready/`
* `table_actions.jsonl`

### Layout actual por defecto

```text
Analysis Ready/<fecha>/<lab>/<mid>.parquet
```

Layout legacy opcional:

```text
Analysis Ready/<fecha>/<lab>/<jornada>/<mid>.parquet
```

### Pregunta que resuelve

```text
¿Cuál es la tabla final estandarizada que se usará en análisis posteriores?
```

---

## 6. `analysis_ready_schema_table.py`

### Función

Valida la consistencia estructural de los archivos Analysis Ready ya construidos.

### Responsabilidades

* leer los Parquet generados
* resumir esquema por archivo
* comparar consistencia por grupo estructural
* detectar variaciones inesperadas de columnas o tipos

### Artefactos generados

* `analysis_ready_schema_by_file.csv`
* `analysis_ready_schema_table.csv`

### Pregunta que resuelve

```text
¿Todas las tablas Analysis Ready tienen un esquema consistente?
```

---

# Estado actual en `main_v3.py`

Hasta este momento, la capa Analysis Ready ya quedó integrada en `main_v3.py` con los siguientes niveles:

* Level 1A — analysis-ready catalog build
* Level 1B — CSV dialect inspection
* Level 1C — column role detection
* Level 1D — measurement time bounds
* Level 1E — analysis table build
* Level 1F — analysis-ready schema validation

Esto significa que `main_v3.py` ya puede:

1. descubrir el dataset
2. generar manifest
3. construir catálogo de análisis
4. validar legibilidad estructural de CSV
5. detectar roles de columnas
6. calcular límites temporales
7. construir tablas Parquet listas para análisis
8. verificar consistencia de esquema final

---

# Artefactos esperados de salida

Al completar esta etapa, deben existir artefactos como:

```text
target_root/
│
├── manifest_all.csv
├── analysis_catalog.csv
├── dialect_report.json
├── column_roles.json
├── measurement_time_bounds.json
├── table_actions.jsonl
├── analysis_ready_schema_by_file.csv
├── analysis_ready_schema_table.csv
│
└── Analysis Ready/
    ├── 02Mar26/
    │   ├── Betta/
    │   │   ├── 02Mar26_Betta_010.parquet
    │   │   ├── 02Mar26_Betta_011.parquet
    │   │   └── ...
    │   └── Epsilon/
    │       └── ...
    ├── 03Mar26/
    └── 04Mar26/
```

---

# Decisiones de diseño importantes

## 1. `source_root` y `target_root` tienen papeles distintos

* `source_root` contiene los CSV físicos originales
* `target_root` contiene artefactos producidos por el pipeline

Analysis Ready debe leer físicamente desde `source_root`, no desde rutas planeadas en `target_root`.

---

## 2. El catálogo de análisis no es todavía la tabla final

`analysis_catalog.csv` es un índice operacional del pipeline.
No reemplaza a los Parquet de Analysis Ready.

---

## 3. `column_roles.json` es un puente semántico

Es el artefacto que permite pasar de una estructura CSV genérica a una tabla con significado experimental explícito.

---

## 4. La validación de esquema es obligatoria

No basta con construir Parquet; además debe verificarse que el conjunto resultante sea estructuralmente consistente.

---

# Resultado esperado

Si esta capa funciona correctamente, el pipeline debe garantizar que:

* todos los CSV relevantes existen y se leen
* el dialecto CSV es consistente o al menos detectable
* las columnas de tiempo y señal quedan identificadas
* las tablas finales tienen nombres y estructura uniformes
* los archivos Analysis Ready quedan listos para capas posteriores como:

  * quality
  * stability
  * informational analysis
  * inferencia secuencial

---

# Qué queda después de esta etapa

Una vez terminada Analysis Ready Data Preparation, el siguiente bloque natural del pipeline ya no es estructural sino analítico.

Los siguientes pasos del proyecto pueden incluir:

* quality metrics
* stability analysis
* informational transforms
* comparison layers
* inferencia secuencial

---

# Resumen ejecutivo

La etapa **Analysis Ready Data Preparation** convierte datos experimentales crudos en una base tabular estable, auditable y lista para análisis.

Esta etapa ya quedó alineada con el contrato v3:

```text
fecha + lab + medición
```

y elimina la dependencia estructural previa de:

* `turno`
* `jornada`
* MID heredado de `main_v2.py`

Con ello, `main_v3.py` ya dispone de una base limpia para las siguientes capas analíticas del pipeline.

```

