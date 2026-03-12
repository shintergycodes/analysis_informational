# quality_metrics.py
from __future__ import annotations

"""
quality_metrics.py (Nivel 1A: Calidad) - MÉTRICAS NÚCLEO

Contiene las funciones "core" del análisis de calidad descrito en la documentación:
- calcular_metricas_luz(df_luz, nombre_archivo, columna)
- validar_valores_luz(df_luz, columna)

Este módulo NO hace IO ni genera reportes. Eso vive en:
- quality_io.py / quality_runner.py (lectura + orquestación)
- quality_compare.py (comparativas + figuras)
- quality_gate.py (decisión PASS/FAIL)

Integración esperada:
- main.py crea QualityConfig (quality_config.py) y lo pasa como `cfg=...` a estas funciones.
- Si `cfg` no se pasa, se usan defaults equivalentes a los del script de referencia.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:  # pragma: no cover
    stats = None
    _SCIPY_IMPORT_ERROR = e


# ---------------------------------------------------------------------
# Tipos/errores internos
# ---------------------------------------------------------------------

class QualityMetricsError(RuntimeError):
    """Error específico de cálculo/validación de métricas."""


# ---------------------------------------------------------------------
# Helpers (sin IO)
# ---------------------------------------------------------------------

def _require_scipy() -> None:
    if stats is None:
        raise ImportError(
            "Falta dependencia: scipy. Instala scipy para usar regresión lineal (stats.linregress). "
            f"Detalle: {_SCIPY_IMPORT_ERROR}"
        )


def _as_numeric_array(
    df: pd.DataFrame,
    columna: str,
    sentinels: Tuple[float, ...] = (-111.0,),
) -> np.ndarray:
    """
    Convierte df[columna] a np.ndarray float, reemplazando sentinels por NaN.
    La limpieza fina (dropna/min samples) se decide en el Runner; aquí solo normalizamos.
    """
    if df is None or columna not in df.columns:
        return np.array([], dtype=float)

    s = pd.to_numeric(df[columna], errors="coerce")

    # Reemplazar sentinels por NaN
    for x in sentinels:
        s = s.mask(s == x, np.nan)

    arr = s.to_numpy(dtype=float)
    return arr


def _finite(arr: np.ndarray) -> np.ndarray:
    """Filtra a valores finitos (quita NaN/Inf)."""
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ---------------------------------------------------------------------
# API pública (core)
# ---------------------------------------------------------------------

def calcular_metricas_luz(
    df_luz: pd.DataFrame,
    nombre_archivo: str,
    columna: str,
    *,
    cfg: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calcula métricas de calidad para una columna (láser).

    Retorna dict con métricas numéricas + calidad_general (1-10).
    Devuelve None si no hay datos suficientes o la columna no existe.

    Parámetros:
    - df_luz: DataFrame con columnas canónicas (ej. Luz1..LuzN).
    - nombre_archivo: etiqueta del archivo/medición (para trazabilidad).
    - columna: nombre de la columna del láser.
    - cfg: QualityConfig (opcional). Si viene, se usan:
        cfg.voltage.v_min / v_max,
        cfg.score.noise_scale / snr_divisor / trend_scale / weights,
        cfg.cleaning.sentinels.
    """
    _require_scipy()

    # Defaults compatibles con script de referencia
    v_min = 0.0
    v_max = 5.0
    sentinels = (-111.0,)
    noise_scale = 20.0
    snr_divisor = 5.0
    trend_scale = 1000.0
    weights = (1/3, 1/3, 1/3)
    clamp_min, clamp_max = 0.0, 10.0

    if cfg is not None:
        # Se asume QualityConfig de quality_config.py
        v_min = float(cfg.voltage.v_min)
        v_max = float(cfg.voltage.v_max)
        sentinels = tuple(getattr(cfg.cleaning, "sentinels", sentinels))
        noise_scale = float(cfg.score.noise_scale)
        snr_divisor = float(cfg.score.snr_divisor)
        trend_scale = float(cfg.score.trend_scale)
        weights = tuple(cfg.score.weights)
        clamp_min = float(cfg.score.clamp_min)
        clamp_max = float(cfg.score.clamp_max)

    if df_luz is None or len(df_luz) == 0 or columna not in df_luz.columns:
        return None

    arr = _as_numeric_array(df_luz, columna, sentinels=sentinels)
    arr = _finite(arr)

    min_samples = 5
    if cfg is not None:
        min_samples = int(getattr(cfg.cleaning, "min_samples_per_channel", min_samples))

    if arr.size < min_samples:
        return None

    # Métricas básicas
    media = float(np.mean(arr))
    desviacion = float(np.std(arr))
    minimo = float(np.min(arr))
    maximo = float(np.max(arr))
    rango = float(maximo - minimo)
    mediana = float(np.median(arr))

    # Ruido RMS (std de la señal centrada)
    ruido_rms = float(np.std(arr - media))

    # SNR (dB)
    if desviacion > 0:
        snr_db = float(20.0 * np.log10(np.abs(media) / desviacion))
    else:
        snr_db = float(np.inf)

    # Coeficiente de variación (%)
    if media != 0:
        coef_variacion = float((desviacion / np.abs(media)) * 100.0)
    else:
        coef_variacion = float(np.inf)

    # Tendencia: pendiente de regresión lineal (V/muestra)
    x = np.arange(arr.size, dtype=float)
    if x.size > 1:
        pendiente, _, r_value, _, _ = stats.linregress(x, arr)
        tendencia_por_muestra = float(pendiente)
        correlacion_tendencia = float(r_value)
    else:
        tendencia_por_muestra = 0.0
        correlacion_tendencia = 0.0

    # Validación de valores fuera de rango (0-5V por default)
    fuera_de_rango = int(np.sum((arr < v_min) | (arr > v_max)))
    pct_fuera_rango = float((fuera_de_rango / arr.size) * 100.0) if arr.size else 0.0

    # Calidad general en escala configurable (según clamp_min/clamp_max)
    # basada en 3 factores: ruido, SNR y estabilidad.
    factor_ruido = _clamp(10.0 - (ruido_rms * noise_scale), clamp_min, clamp_max)
    factor_snr = _clamp(snr_db / snr_divisor, clamp_min, clamp_max)
    factor_estabilidad = _clamp(10.0 - (np.abs(tendencia_por_muestra) * trend_scale), clamp_min, clamp_max)

    w1, w2, w3 = weights
    calidad_general = float(w1 * factor_ruido + w2 * factor_snr + w3 * factor_estabilidad)
    calidad_general = _clamp(calidad_general, clamp_min, clamp_max)

    return {
        "archivo": nombre_archivo,
        "columna": columna,
        "media": media,
        "desviacion": desviacion,
        "minimo": minimo,
        "maximo": maximo,
        "rango": rango,
        "mediana": mediana,
        "ruido_rms": ruido_rms,
        "snr_db": snr_db,
        "coef_variacion": coef_variacion,
        "tendencia_por_muestra": tendencia_por_muestra,
        "correlacion_tendencia": correlacion_tendencia,
        "valores_fuera_rango": fuera_de_rango,
        "porcentaje_fuera_rango": pct_fuera_rango,
        "calidad_general": calidad_general,
        "num_muestras": int(arr.size),
        "muestras_suficientes": True,
        "factor_ruido": factor_ruido,
        "factor_snr": factor_snr,
        "factor_estabilidad": factor_estabilidad,
    }


def validar_valores_luz(
    df_luz: pd.DataFrame,
    columna: str,
    *,
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Valida rangos y consistencia para una columna de láser.

    Retorna:
      {"valido": bool, "razon": str, "pct_fuera_rango": float, "fuera_rango": int}

    Regla:
    - Si hay NaN/Inf en la serie cruda -> invalida (valido=False)
    - Si hay fuera de rango:
        - válido solo si el porcentaje <= max_out_of_range_pct (configurable)
        - razón siempre documenta el porcentaje
    """
    v_min = 0.0
    v_max = 5.0
    max_out_pct = 0.1
    sentinels = (-111.0,)

    if cfg is not None:
        v_min = float(cfg.voltage.v_min)
        v_max = float(cfg.voltage.v_max)
        max_out_pct = float(cfg.voltage.max_out_of_range_pct)
        sentinels = tuple(getattr(cfg.cleaning, "sentinels", sentinels))

    if df_luz is None or columna not in df_luz.columns:
        return {"valido": False, "razon": f"Columna {columna} no disponible", "pct_fuera_rango": None, "fuera_rango": None}

    raw = pd.to_numeric(df_luz[columna], errors="coerce")

    # Sentinels -> NaN para conteo de finitud
    for x in sentinels:
        raw = raw.mask(raw == x, np.nan)

    values = raw.to_numpy(dtype=float)

    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return {"valido": False, "razon": "Valores NaN o infinitos", "pct_fuera_rango": None, "fuera_rango": None}

    min_samples = 5
    if cfg is not None:
        min_samples = int(getattr(cfg.cleaning, "min_samples_per_channel", min_samples))

    if values.size < min_samples:
        return {
            "valido": False,
            "razon": f"Muestras insuficientes: {values.size} < {min_samples}",
            "pct_fuera_rango": None,
            "fuera_rango": None,
        }

    fuera = int(np.sum((values < v_min) | (values > v_max)))
    pct = float((fuera / values.size) * 100.0) if values.size else 0.0

    if fuera > 0:
        if pct > max_out_pct:
            return {
                "valido": False,
                "razon": f"Fuera de rango {v_min}-{v_max}V: {pct:.4f}% excede umbral {max_out_pct}%",
                "pct_fuera_rango": pct,
                "fuera_rango": fuera,
            }
        return {
            "valido": True,
            "razon": f"Algunos valores fuera de rango {v_min}-{v_max}V ({pct:.4f}%)",
            "pct_fuera_rango": pct,
            "fuera_rango": fuera,
        }

    return {"valido": True, "razon": "Valores dentro de rango aceptable", "pct_fuera_rango": pct, "fuera_rango": fuera}


__all__ = [
    "QualityMetricsError",
    "calcular_metricas_luz",
    "validar_valores_luz",
]
