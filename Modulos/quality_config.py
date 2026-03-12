# quality_config.py
from __future__ import annotations

"""
QualityConfig (Nivel 1A: Calidad)

Este módulo centraliza la configuración del análisis de calidad descrito en la documentación
(calidad_datos.py) y actúa como contrato estable entre:

- quality_runner.py  (orquestación por medición/archivo)
- quality_metrics.py (cálculo de métricas por láser)
- quality_compare.py (comparativas + reportes)
- quality_gate.py    (decisión PASS/FAIL para pasar a Nivel 2)

Diseño:
- Este archivo NO ejecuta análisis.
- Solo declara parámetros, defaults, validación y utilidades de serialización.
- Los defaults se basan en la sección "Parámetros de Configuración" de la documentación.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import json


# -----------------------------
# Dataclasses de configuración
# -----------------------------

@dataclass(frozen=True)
class QualityThresholds:
    """
    Umbrales de clasificación (escala 1-10).

    - excellent/good/bad provienen de la documentación.
    - pass_to_informational define el gate que abre puerta al Nivel 2.
    """
    excellent: float = 8.5
    good: float = 7.0
    bad: float = 6.0
    pass_to_informational: float = 7.0  # gate >= 7.0 pasa a informacionales


@dataclass(frozen=True)
class VoltageValidation:
    """
    Rangos de validación para valores de voltaje.
    """
    v_min: float = 0.0
    v_max: float = 5.0
    max_out_of_range_pct: float = 0.1  # % máximo permitido (por columna / medición)


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Umbrales de estabilidad para clasificación/penalización.
    """
    max_abs_trend_v_per_sample: float = 0.001  # V/muestra
    max_cv_pct: float = 10.0                  # % máximo permitido


@dataclass(frozen=True)
class ScoreModel:
    """
    Parámetros para el cálculo de 'calidad_general' (1-10).

    Nota: el cálculo exacto vive en quality_metrics.py (calcular_metricas_luz).
    Estos parámetros permiten ajustar penalizaciones sin tocar la lógica.
    """
    # Penalización por ruido: factor_ruido = max(0, 10 - ruido_rms * noise_scale)
    noise_scale: float = 20.0

    # Factor SNR: factor_snr = min(10, snr_db / snr_divisor)
    snr_divisor: float = 5.0

    # Penalización por deriva: factor_estabilidad = max(0, 10 - abs(trend)*trend_scale)
    trend_scale: float = 1000.0

    # Pesos (ruido, snr, estabilidad). Deben sumar 1.0
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)

    clamp_min: float = 0.0
    clamp_max: float = 10.0


@dataclass(frozen=True)
class QualityArtifacts:
    """
    Nombres de archivos esperados en outputs (Nivel 1A / Quality).

    Incluye:
    - artefactos base por medición/láser
    - artefactos diagnósticos visuales
    - artefactos agregados multi-lab para etapas posteriores
    """
    resultados_luces_csv: str = "resultados_luces.csv"
    datos_completos_luces_csv: str = "datos_completos_luces.csv"
    quality_scores_by_file_csv: str = "quality_scores_by_file.csv"
    quality_summary_by_lab_csv: str = "quality_summary_by_lab.csv"
    quality_summary_by_date_lab_csv: str = "quality_summary_by_date_lab.csv"
    quality_summary_by_laser_csv: str = "quality_summary_by_laser.csv"
    resumen_ejecutivo_txt: str = "resumen_ejecutivo.txt"

    reporte_comparativo_template_png: str = "reporte_comparativo_{col}.png"
    comparacion_columnas_archivos_png: str = "comparacion_columnas_archivos.png"
    evolucion_temporal_columnas_png: str = "evolucion_temporal_columnas.png"


@dataclass(frozen=True)
class ChannelPolicy:
    """
    Channel selection policy.

    - If channels is None, channels are auto-detected using allowed_prefixes.
    - If channels is provided, that subset is enforced (legacy or controlled runs).
    """
    channels: Optional[Tuple[str, ...]] = None
    min_channels_required: int = 4
    allowed_prefixes: Tuple[str, ...] = ("Ch", "Luz", "Laser", "Canal")


@dataclass(frozen=True)
class DataCleaningPolicy:
    """
    Política mínima de limpieza para evitar falsos fallos por sentinels o NaNs.

    - sentinels: valores especiales a tratar como NaN (ej: -111.0)
    - min_valid_samples: número mínimo de muestras válidas por canal para evaluar métricas
    """
    sentinels: Tuple[float, ...] = (-111.0,)
    min_valid_samples: int = 10


@dataclass(frozen=True)
class QualityConfig:
    """
    Configuración completa de Nivel 1A (Calidad).

    Esta clase se instancia en main.py y se pasa al QualityRunner.
    """
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    voltage: VoltageValidation = field(default_factory=VoltageValidation)
    stability: StabilityThresholds = field(default_factory=StabilityThresholds)
    score: ScoreModel = field(default_factory=ScoreModel)
    channels: ChannelPolicy = field(default_factory=ChannelPolicy)
    cleaning: DataCleaningPolicy = field(default_factory=DataCleaningPolicy)
    artifacts: QualityArtifacts = field(default_factory=QualityArtifacts)

    version: str = "1.1"
    name: str = "Level1_Quality"

    # ---------
    # Factory
    # ---------
    @classmethod
    def from_defaults(cls) -> "QualityConfig":
        """
        Defaults alineados con la documentación.
        """
        return cls()

    # ------------
    # Validación
    # ------------
    def validate(self) -> None:
        """
        Valida coherencia interna. Lanza ValueError con mensajes accionables.
        """
        t = self.thresholds
        if not (0 <= t.bad <= t.good <= t.excellent <= 10):
            raise ValueError(
                f"Umbrales inválidos: se espera 0<=bad<=good<=excellent<=10; "
                f"recibido bad={t.bad}, good={t.good}, excellent={t.excellent}"
            )

        if not (0 <= t.pass_to_informational <= 10):
            raise ValueError(f"pass_to_informational fuera de [0,10]: {t.pass_to_informational}")

        v = self.voltage
        if not (v.v_min < v.v_max):
            raise ValueError(f"Rango voltaje inválido: v_min={v.v_min} debe ser < v_max={v.v_max}")
        if v.max_out_of_range_pct < 0:
            raise ValueError(f"max_out_of_range_pct no puede ser negativo: {v.max_out_of_range_pct}")

        s = self.stability
        if s.max_abs_trend_v_per_sample < 0:
            raise ValueError("max_abs_trend_v_per_sample no puede ser negativo.")
        if s.max_cv_pct < 0:
            raise ValueError("max_cv_pct no puede ser negativo.")

        sc = self.score
        if sc.noise_scale <= 0:
            raise ValueError("score.noise_scale debe ser > 0")
        if sc.snr_divisor <= 0:
            raise ValueError("score.snr_divisor debe ser > 0")
        if sc.trend_scale <= 0:
            raise ValueError("score.trend_scale debe ser > 0")

        w = sc.weights
        if len(w) != 3:
            raise ValueError("score.weights debe tener exactamente 3 pesos (ruido, snr, estabilidad).")
        wsum = float(sum(w))
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(f"score.weights debe sumar 1.0; suma actual={wsum}")

        ch = self.channels
        if ch.min_channels_required < 1:
            raise ValueError("channels.min_channels_required debe ser >= 1")
        if ch.channels is not None and len(ch.channels) < ch.min_channels_required:
            raise ValueError(
                f"channels.channels tiene {len(ch.channels)} elementos, "
                f"pero min_channels_required={ch.min_channels_required}"
            )

        if ch.channels is None:
            if not ch.allowed_prefixes:
                raise ValueError("channels.allowed_prefixes cannot be empty when channels is None.")
            if any((not isinstance(p, str) or not p.strip()) for p in ch.allowed_prefixes):
                raise ValueError("channels.allowed_prefixes must contain non-empty strings.")

        cl = self.cleaning
        if cl.min_valid_samples < 1:
            raise ValueError("cleaning.min_valid_samples debe ser >= 1")


    # -----------------
    # Serialización
    # -----------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: Path, indent: int = 2) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_str(indent=indent), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityConfig":
        """
        Reconstruye config desde dict (ej. config guardada en JSON).
        """
        return cls(
            thresholds=QualityThresholds(**data.get("thresholds", {})),
            voltage=VoltageValidation(**data.get("voltage", {})),
            stability=StabilityThresholds(**data.get("stability", {})),
            score=ScoreModel(**data.get("score", {})),
            channels=ChannelPolicy(**data.get("channels", {})),
            cleaning=DataCleaningPolicy(**data.get("cleaning", {})),
            artifacts=QualityArtifacts(**data.get("artifacts", {})),
            version=data.get("version", "1.0"),
            name=data.get("name", "Level1_Quality"),
        )

    @classmethod
    def read_json(cls, path: Path) -> "QualityConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
