import io
import re
import zipfile
from pathlib import Path
from datetime import datetime

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = Path(r"E:\2027\Marzo")

LAB_MAP = {
    "coyoacan": "Epsilon",
    "lindavista": "Betta",
}

# ==========================================================
# UTILIDADES
# ==========================================================

def detectar_laboratorio_desde_ruta(ruta: str) -> str:
    ruta_low = ruta.lower()
    for clave, lab in LAB_MAP.items():
        if clave in ruta_low:
            return lab
    return "Unknown"


def extraer_fecha_desde_nombre(nombre: str) -> str | None:
    """
    Busca patrones tipo:
      2026-03-2_3.zip
      2026-03-3_4.zip
      2026-03-04.zip
    y toma la primera fecha encontrada.
    """
    m = re.search(r"(\d{4})-(\d{2})-(\d{1,2})", nombre)
    if not m:
        return None

    yyyy = int(m.group(1))
    mm = int(m.group(2))
    dd = int(m.group(3))

    try:
        dt = datetime(yyyy, mm, dd)
        return dt.strftime("%d%b%y")
    except ValueError:
        return None


def detectar_mes_root(fecha_token: str) -> str:
    dt = datetime.strptime(fecha_token, "%d%b%y")
    return dt.strftime("%b%y")


def asegurar_directorio_destino(base_dir: Path, fecha_token: str, laboratorio: str) -> Path:
    mes_root = detectar_mes_root(fecha_token)
    destino = base_dir / mes_root / fecha_token / laboratorio / "Raw Data"
    destino.mkdir(parents=True, exist_ok=True)
    return destino


# ==========================================================
# EXPLORACIÓN / EXTRACCIÓN RECURSIVA
# ==========================================================

def extraer_csv_raw_desde_zip(
    zip_file_obj,
    ruta_virtual_actual: str,
    base_dir: Path,
    laboratorio_actual: str | None = None,
    fecha_actual: str | None = None,
    resumen: list | None = None,
):
    """
    Recorre recursivamente un ZIP (o ZIP en memoria), entra a ZIPs internos
    y extrae archivos que estén dentro de csv_raw/.
    """
    if resumen is None:
        resumen = []

    with zipfile.ZipFile(zip_file_obj, "r") as z:
        for item in z.namelist():
            ruta_virtual_item = f"{ruta_virtual_actual}/{item}"

            # Actualizar laboratorio si aparece en la ruta
            lab_detectado = detectar_laboratorio_desde_ruta(ruta_virtual_item)
            if lab_detectado != "Unknown":
                laboratorio_item = lab_detectado
            else:
                laboratorio_item = laboratorio_actual

            # Actualizar fecha si el nombre actual contiene fecha
            fecha_detectada = extraer_fecha_desde_nombre(item)
            fecha_item = fecha_detectada if fecha_detectada is not None else fecha_actual

            # 1) Si el item es otro ZIP, entrar recursivamente
            if item.lower().endswith(".zip"):
                try:
                    data = z.read(item)
                    extraer_csv_raw_desde_zip(
                        io.BytesIO(data),
                        ruta_virtual_item,
                        base_dir,
                        laboratorio_actual=laboratorio_item,
                        fecha_actual=fecha_item,
                        resumen=resumen,
                    )
                except zipfile.BadZipFile:
                    print(f"[AVISO] No se pudo abrir como ZIP: {ruta_virtual_item}")
                continue

            # 2) Si está dentro de csv_raw y es CSV, extraer
            item_low = item.lower()
            if "csv_raw/" in item_low and item_low.endswith(".csv"):
                if laboratorio_item is None:
                    laboratorio_item = "Unknown"

                if fecha_item is None:
                    print(f"[AVISO] No se pudo detectar fecha para: {ruta_virtual_item}")
                    continue

                destino_dir = asegurar_directorio_destino(base_dir, fecha_item, laboratorio_item)
                filename = Path(item).name
                destino_archivo = destino_dir / filename

                with z.open(item) as source, open(destino_archivo, "wb") as target:
                    target.write(source.read())

                resumen.append({
                    "ruta_virtual": ruta_virtual_item,
                    "laboratorio": laboratorio_item,
                    "fecha": fecha_item,
                    "destino": str(destino_archivo),
                })

                print(f"[OK] {ruta_virtual_item}")
                print(f"     -> {destino_archivo}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"No existe BASE_DIR: {BASE_DIR}")

    zip_raiz = sorted(BASE_DIR.glob("*.zip"))

    if not zip_raiz:
        print("No se encontraron ZIP raíz en:", BASE_DIR)
        return

    print("\nIniciando organización multinivel...\n")
    print("Directorio base:", BASE_DIR)
    print("ZIPs raíz encontrados:")
    for z in zip_raiz:
        print("  -", z.name)
    print()

    resumen = []

    for zip_path in zip_raiz:
        print("=" * 80)
        print("Explorando ZIP raíz:", zip_path.name)
        print("=" * 80)

        laboratorio_raiz = detectar_laboratorio_desde_ruta(zip_path.name)
        if laboratorio_raiz == "Unknown":
            laboratorio_raiz = None

        with open(zip_path, "rb") as f:
            extraer_csv_raw_desde_zip(
                f,
                ruta_virtual_actual=zip_path.name,
                base_dir=BASE_DIR,
                laboratorio_actual=laboratorio_raiz,
                fecha_actual=None,
                resumen=resumen,
            )

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print("Total CSV extraídos:", len(resumen))

    if resumen:
        por_fecha_lab = {}
        for r in resumen:
            key = (r["fecha"], r["laboratorio"])
            por_fecha_lab[key] = por_fecha_lab.get(key, 0) + 1

        for (fecha, lab), n in sorted(por_fecha_lab.items()):
            print(f"{fecha} | {lab}: {n} archivos")

    print("\nOrganización completada.\n")


if __name__ == "__main__":
    main()