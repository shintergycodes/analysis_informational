from pathlib import Path
import pandas as pd
import re
import json

# ============================================================
# (Compat) display para show() en .py (sin tocar show())
# ============================================================
try:
    from IPython.display import display  # noqa: F401
except Exception:
    def display(x):  # type: ignore
        print(x)

# ============================================================
# RAÍZ DEL PROYECTO (AUTO: donde esté este .py, o algún padre)
# ============================================================

def _find_project_root(start_dir: Path) -> Path:
    """
    Busca hacia arriba (start_dir, parent, parent...) hasta encontrar
    la estructura esperada de DATA_PATH. Si no la encuentra, regresa start_dir.
    """
    expected_rel = Path("Mar26_clean") / "Reports" / "Level4_Change" / "BlindSequential"
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / expected_rel).is_dir():
            return candidate
    return start_dir

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)

# ============================================================
# RUTA REAL DE LOS DATOS (Nivel4_Change / BlindSequential)
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "Mar26_clean" # MODIFICAR EN CASO NECESARIO
    / "Reports"
    / "Level4_Change"
    / "BlindSequential"
)

# ============================================================
# UTILIDADES: DETECCIÓN / LECTURA / LIMPIEZA
# ============================================================

FEATURES = ["forma", "mov", "ene", "fft"]



#================================== LABS

# ============================================================
# LABS: detección de laboratorio
# ============================================================

LAB_ALIASES = {
    "lab1": "Lab_1",
    "lab_1": "Lab_1",
    "laboratorio1": "Lab_1",
    "laboratorio_1": "Lab_1",
    "lab2": "Lab_2",
    "lab_2": "Lab_2",
    "laboratorio2": "Lab_2",
    "laboratorio_2": "Lab_2",
}

def _normalize_lab_token(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def extract_lab_from_path(path: Path) -> str:
    """
    Detecta el laboratorio desde la ruta o nombre del archivo.
    Ajusta LAB_ALIASES según tu estructura real.
    """
    parts = list(path.parts) + [path.name]
    for part in reversed(parts):
        key = _normalize_lab_token(part)
        if key in LAB_ALIASES: 
            return LAB_ALIASES[key]

    # intento extra por regex sobre toda la ruta
    whole = _normalize_lab_token(str(path))
    m = re.search(r"lab(\d+)", whole)
    if m:
        return f"Lab_{m.group(1)}"

    m = re.search(r"laboratorio(\d+)", whole)
    if m:
        return f"Lab_{m.group(1)}"

    return "LAB_UNK"

def make_lab_date_key(lab: str, date: str) -> str:
    return f"{lab}__{date}"

#====================================

def extract_date_from_filename(fname: str) -> str | None:
    m = re.search(r"(\d{2}[A-Za-z]{3}\d{2})", fname)
    return m.group(1) if m else None

def clean_colname(c) -> str:
    return str(c).strip().replace(" ", "_")

def read_sheet_raw(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Lee hoja donde:
      - fila 0 = encabezados reales
      - filas 1.. = datos
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    header = [clean_colname(c) for c in raw.iloc[0].tolist()]
    df = raw.iloc[1:].copy()
    df.columns = header
    df = df.reset_index(drop=True)

    # eliminar columnas basura
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    return df

# ============================================================
# CÁLCULOS MATEMÁTICOS
# ============================================================

def to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def recompute_from_directional_kl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruye completamente, desde KL direccionales:
      - s_k
      - pct_k (por característica, suma=100 para cada k)
      - g_raw (suma de pct_k)
      - g_norm (normalización global, suma=100)

    Y además llena columnas compatibles con Excel:
      - %forma_col, %mov_col, %ene_col, %fft_col
    """
    df = df.copy()

    # 1) Tipado numérico seguro
    base_cols = (
        ["n", "N_from", "N_to"]
        + [f"DKL_{k}_fwd" for k in FEATURES]
        + [f"DKL_{k}_bwd" for k in FEATURES]
    )
    df = to_numeric_cols(df, base_cols)

    if "n" in df.columns:
        # Int64 admite NA
        df["n"] = df["n"].astype("Int64")

    # 2) s_k = KL_fwd + KL_bwd
    for k in FEATURES:
        fwd = f"DKL_{k}_fwd"
        bwd = f"DKL_{k}_bwd"
        if fwd not in df.columns or bwd not in df.columns:
            raise KeyError(f"Faltan columnas requeridas: {fwd}, {bwd}")
        df[f"s_{k}"] = df[fwd] + df[bwd]

    # 3) pct_k: normalización interna por característica
    for k in FEATURES:
        s_col = f"s_{k}"
        denom = df[s_col].sum(skipna=True)

        if denom == 0 or pd.isna(denom):
            df[f"pct_{k}"] = 0.0
        else:
            df[f"pct_{k}"] = 100.0 * df[s_col] / denom

        # Compatibilidad Excel
        df[f"%{k}_col"] = df[f"pct_{k}"]

    # 4) g_raw = sum_k pct_k
    pct_cols = [f"pct_{k}" for k in FEATURES]
    df["g_raw"] = df[pct_cols].sum(axis=1)

    # 5) g_norm = 100*g_raw / sum_m g_raw(m)
    denom_g = df["g_raw"].sum(skipna=True)
    if denom_g == 0 or pd.isna(denom_g):
        df["g_norm"] = 0.0
    else:
        df["g_norm"] = 100.0 * df["g_raw"] / denom_g

    return df

# ============================================================
# CARGA DE LIBROS
# ============================================================

def load_all_books(data_path: Path) -> dict:
    """
    Devuelve estructura:
        data[lab][date][laser] = df_con_g_norm

    Busca recursivamente para soportar múltiples laboratorios.
    """
    data: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}

    xlsx_candidates = sorted(data_path.rglob("transitions_per_laser_*.xlsx"))

    for xlsx in xlsx_candidates:
        date = extract_date_from_filename(xlsx.name)
        if date is None:
            continue

        lab = extract_lab_from_path(xlsx)

        print(f"\nProcesando lab={lab} | fecha={date} | archivo={xlsx.name}")

        xls = pd.ExcelFile(xlsx)

        data.setdefault(lab, {})
        data[lab].setdefault(date, {})

        for sheet in xls.sheet_names:
            df = read_sheet_raw(xlsx, sheet)
            df = recompute_from_directional_kl(df)
            data[lab][date][sheet] = df

    return data

# ============================================================
# VALIDACIÓN
# ============================================================

def validate_internal_sums(data: dict) -> pd.DataFrame:
    rows = []
    for lab, by_date in data.items():
        for date, lasers in by_date.items():
            for laser, df in lasers.items():
                rows.append({
                    "lab": lab,
                    "date": date,
                    "laser": laser,
                    "sum_g_norm": float(pd.to_numeric(df["g_norm"], errors="coerce").sum()),
                    "sum_pct_forma": float(pd.to_numeric(df.get("pct_forma"), errors="coerce").sum()),
                    "sum_pct_mov": float(pd.to_numeric(df.get("pct_mov"), errors="coerce").sum()),
                    "sum_pct_ene": float(pd.to_numeric(df.get("pct_ene"), errors="coerce").sum()),
                    "sum_pct_fft": float(pd.to_numeric(df.get("pct_fft"), errors="coerce").sum()),
                })
    return pd.DataFrame(rows).sort_values(["lab", "date", "laser"], ignore_index=True)

# ============================================================
# FUNCIÓN EXCLUSIVA: DISEÑO (BÁSICO) DEL .TEX
# ============================================================

def _df_to_latex_longtable(df: pd.DataFrame, caption: str, label: str, max_rows: int | None = None) -> str:
    dfx = df.copy()
    if max_rows is not None:
        dfx = dfx.head(max_rows)
    if len(dfx.columns) == 0:
        return ""
    return dfx.to_latex(
        index=False,
        longtable=True,
        escape=True,
        caption=caption,
        label=label
    )

def latex_escape_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "/")        # rutas Windows → estilo LaTeX
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    s = s.replace("$", r"\$")
    return s

def build_basic_report_tex(ctx: dict) -> str:
    """
    Documento LaTeX básico (provisional) con toda la información que
    el programa imprime/expone.
    """

    script_dir = ctx.get("SCRIPT_DIR", "")
    project_root = ctx.get("PROJECT_ROOT", "")
    data_path = ctx.get("DATA_PATH", "")

    xlsx_files = ctx.get("xlsx_files", [])
    json_files = ctx.get("json_files", [])
    dates = ctx.get("DATES", [])

    fechas_cargadas = ctx.get("fechas_cargadas", [])
    hojas_ejemplo = ctx.get("hojas_ejemplo", [])


    validation_df = ctx.get("validation_df", None)

    date_example = ctx.get("date_example", None)
    laser_example = ctx.get("laser_example", None)
    df_example = ctx.get("df_example", None)
    df_example_cols = ctx.get("df_example_cols", None)

    df_long = ctx.get("df_long", None)
    trans_counts = ctx.get("trans_counts", None)
    N_min = ctx.get("N_min", None)
    df_aggs = ctx.get("df_aggs", None)
    loc_wide = ctx.get("loc_wide", None)
    dom_min = ctx.get("dom_min", None)
    dom_max = ctx.get("dom_max", None)
    df_p = ctx.get("df_p", None)
    sig = ctx.get("sig", None)
    by_group = ctx.get("by_group", None)
    by_lab = ctx.get("by_lab", None)
    conc = ctx.get("conc", None)
    top3 = ctx.get("top3", None)
    resumen = ctx.get("resumen", None)

    def itemize_list(items):
        if not items:
            return "\\emph{(vacío)}\n"
        out = "\\begin{itemize}\n"
        for it in items:
            out += f"  \\item {latex_escape_text(it)}\n"
        out += "\\end{itemize}\n"
        return out

    tex = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{array}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{placeins}")
    tex.append(r"\usepackage{caption}")
    tex.append(r"\captionsetup{labelfont=bf}")
    tex.append(r"\begin{document}")

    tex.append(r"\section*{Reporte automático (básico) del pipeline}")

    # ---------------- RUTAS ----------------
    tex.append(r"\subsection*{Rutas detectadas}")
    tex.append(r"\begin{itemize}")
    tex.append(rf"  \item SCRIPT\_DIR: \texttt{{{latex_escape_text(script_dir)}}}")
    tex.append(rf"  \item PROJECT\_ROOT: \texttt{{{latex_escape_text(project_root)}}}")
    tex.append(rf"  \item DATA\_PATH: \texttt{{{latex_escape_text(data_path)}}}")
    tex.append(r"\end{itemize}")

    # ---------------- ARCHIVOS ----------------
    tex.append(r"\subsection*{Archivos detectados}")
    tex.append(r"\paragraph{Excel encontrados:}")
    tex.append(itemize_list([getattr(f, "name", str(f)) for f in xlsx_files]))
    tex.append(r"\paragraph{JSON encontrados:}")
    tex.append(itemize_list([getattr(f, "name", str(f)) for f in json_files]))
    tex.append(r"\paragraph{Jornadas detectadas:}")
    tex.append(itemize_list(dates))

    if fechas_cargadas:
        tex.append(r"\subsection*{Carga}")
        tex.append(r"\paragraph{Fechas cargadas:}")
        tex.append(itemize_list(fechas_cargadas))

    if hojas_ejemplo:
        tex.append(r"\paragraph{Hojas ejemplo (primera fecha):}")
        tex.append(itemize_list(hojas_ejemplo))

    # ---------------- VALIDACIÓN ----------------
    if isinstance(validation_df, pd.DataFrame) and not validation_df.empty:
        tex.append(r"\subsection*{Validación interna}")
        tex.append(_df_to_latex_longtable(
            validation_df.head(12),
            caption="Validación (primeras filas).",
            label="tab:validacion"
        ))

    # ---------------- EJEMPLO ----------------
    if date_example and laser_example:
        tex.append(r"\subsection*{Ejemplo}")
        tex.append(rf"\noindent \textbf{{Fecha:}} {latex_escape_text(date_example)}\\")
        tex.append(rf"\noindent \textbf{{Láser:}} {latex_escape_text(laser_example)}\\")

        if isinstance(df_example_cols, list):
            cols_str = ", ".join([latex_escape_text(c) for c in df_example_cols])
            tex.append(rf"\noindent \textbf{{Columnas:}} {cols_str}\\")

        if isinstance(df_example, pd.DataFrame) and not df_example.empty:
            tex.append(_df_to_latex_longtable(
                df_example.head(10),
                caption="Head(10) del ejemplo.",
                label="tab:ejemplo_head"
            ))

    # ---------------- PIPELINE 2 ----------------
    if isinstance(df_long, pd.DataFrame):
        tex.append(r"\section*{Pipeline limpio}")
        tex.append(_df_to_latex_longtable(
            df_long.head(12),
            caption="df_long (head).",
            label="tab:dflong"
        ))

    if isinstance(trans_counts, pd.DataFrame) and N_min is not None:
        tex.append(r"\subsection*{Rango comparable}")
        tex.append(rf"\noindent \textbf{{N\_min}} = {int(N_min)}\\")
        tex.append(_df_to_latex_longtable(
            trans_counts.head(48),
            caption="Conteos por (fecha, láser).",
            label="tab:trans_counts"
        ))

    if isinstance(df_aggs, pd.DataFrame):
        tex.append(_df_to_latex_longtable(
            df_aggs.head(12),
            caption=r"df\_aggs (head).",
            label="tab:dfaggs"
        ))

    if isinstance(df_p, pd.DataFrame):
        tex.append(_df_to_latex_longtable(
            df_p.head(30),
            caption=r"df\_p (head).",
            label="tab:dfp"
        ))

    if isinstance(resumen, dict) and resumen:
        tex.append(r"\subsection*{Resumen inferencial}")
        tex.append(r"\begin{itemize}")
        tex.append(rf"  \item Transiciones analizadas: {int(resumen.get('transiciones_analizadas',0))}")
        tex.append(rf"  \item Significativas (q$\le$0.05): {int(resumen.get('significativas_q<=0.05',0))}")

        grp_dom, cnt = resumen.get("grupo_mas_domina", (None, 0))
        lab_dom, cnt_lab = resumen.get("lab_mas_domina", (None, 0))
        share_topK = float(resumen.get("share_topK", 0.0))

        tex.append(rf"  \item Grupo con mayor dominancia: {latex_escape_text(grp_dom)} ({int(cnt)})")
        tex.append(rf"  \item Laboratorio con mayor dominancia: {latex_escape_text(lab_dom)} ({int(cnt_lab)})")
        tex.append(rf"  \item Concentración Top-K: {100*share_topK:0.2f}\%")
        tex.append(r"\end{itemize}")

    tex.append(r"\end{document}")
    return "\n".join(tex)


def write_tex_report(tex_path: Path, tex_content: str) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_content, encoding="utf-8")


# ============================================================
# PIPELINE LIMPIO: df_long -> df_aggs -> inferencia por transición
# (salida organizada + tablas clave)
# ============================================================

import numpy as np
# (pd y re ya importados arriba)

# ----------------------------
# 0) Utilidades de display
# ----------------------------
def show(title, obj=None, max_rows=12):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    if obj is None:
        return
    if isinstance(obj, pd.DataFrame):
        display(obj.head(max_rows))
    else:
        display(obj)

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:0.4f}")


# ============================================================
# 1) Construcción df_long
#    columnas mínimas: d, ell, laser, n, g_norm, A, pct_*
# ============================================================
def build_df_long(data: dict, FEATURES: list[str]) -> pd.DataFrame:
    rows = []
    pct_cols = [f"pct_{k}" for k in FEATURES]
    s_cols   = [f"s_{k}"   for k in FEATURES]

    for lab, by_date in data.items():
        for d, lasers_dict in by_date.items():
            for laser_name, df in lasers_dict.items():

                m = re.search(r"(\d+)", str(laser_name))
                ell = int(m.group(1)) if m else np.nan

                tmp = df.copy()
                tmp["lab"] = str(lab)
                tmp["d"] = d
                tmp["lab_date"] = make_lab_date_key(lab, d)
                tmp["laser"] = str(laser_name)
                tmp["ell"] = ell

                tmp["n"] = pd.to_numeric(tmp["n"], errors="coerce").astype("Int64")

                needed = ["g_norm"] + s_cols + pct_cols
                missing = [c for c in needed if c not in tmp.columns]
                if missing:
                    raise KeyError(f"Faltan columnas en {lab} / {d} / {laser_name}: {missing}")

                tmp["A"] = tmp[s_cols].sum(axis=1, skipna=True)

                keep = ["lab", "d", "lab_date", "ell", "laser", "n", "g_norm", "A"] + pct_cols
                rows.append(tmp[keep])

    df_long = pd.concat(rows, ignore_index=True)
    df_long = df_long.sort_values(["lab", "d", "ell", "n"], ignore_index=True)
    df_long["ell"] = pd.to_numeric(df_long["ell"], errors="coerce").astype("Int64")

    return df_long

# ============================================================
# 2) Rango comparable N_min y agregados por (d,n)
# ============================================================
def compute_N_min(df_long: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    trans_counts = (
        df_long.groupby(["lab", "d", "ell"], as_index=False)["n"]
              .max()
              .rename(columns={"n": "N_dl"})
    )
    N_min = int(trans_counts["N_dl"].min())
    return N_min, trans_counts


def build_df_aggs(df_long: pd.DataFrame, N_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfN = df_long[df_long["n"].between(1, N_min)].copy()

    # Loc_lab,d(n) = mediana de g_norm sobre lasers
    df_loc = (
        dfN.groupby(["lab", "d", "n"], as_index=False)["g_norm"]
           .median()
           .rename(columns={"g_norm": "Loc"})
    )

    # Abs_lab,d(n) = mediana de A sobre lasers
    df_abs = (
        dfN.groupby(["lab", "d", "n"], as_index=False)["A"]
           .median()
           .rename(columns={"A": "Abs"})
    )

    # IQR_lab,d(n)
    q75 = dfN.groupby(["lab", "d", "n"])["g_norm"].quantile(0.75).reset_index(name="Q75")
    q25 = dfN.groupby(["lab", "d", "n"])["g_norm"].quantile(0.25).reset_index(name="Q25")
    df_iqr = q75.merge(q25, on=["lab", "d", "n"], how="left")
    df_iqr["IQR"] = df_iqr["Q75"] - df_iqr["Q25"]
    df_iqr = df_iqr[["lab", "d", "n", "IQR"]]

    # Dominancia: comparar ganadores por (ell, n) entre TODOS los grupos lab+fecha
    grp = dfN.groupby(["ell", "n"])
    max_g = grp["g_norm"].transform("max")
    is_win = (dfN["g_norm"] == max_g)

    n_winners = grp["g_norm"].transform(lambda s: (s == s.max()).sum())
    dfN["v_vote"] = is_win.astype(float) / n_winners.astype(float)

    df_dom = (
        dfN.groupby(["lab", "d", "n"], as_index=False)["v_vote"]
           .sum()
           .rename(columns={"v_vote": "Dom"})
    )

    df_aggs = (
        df_loc.merge(df_abs, on=["lab", "d", "n"], how="left")
              .merge(df_iqr, on=["lab", "d", "n"], how="left")
              .merge(df_dom, on=["lab", "d", "n"], how="left")
              .sort_values(["lab", "n", "d"], ignore_index=True)
    )

    return df_aggs, dfN


# ============================================================
# 3) Inferencia por transición: permutación sobre etiquetas
#    T(n)=max(median)-min(median), BH para q
# ============================================================
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    qs = ps * m / (np.arange(1, m+1))
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    out = np.empty_like(qs)
    out[order] = np.clip(qs, 0, 1)
    return out


def perm_pvalue_T(values: np.ndarray, labels: np.ndarray, dates: list, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)

    def T(lbls):
        meds = []
        for d in dates:
            v = values[lbls == d]
            meds.append(np.median(v) if len(v) else np.nan)
        meds = np.asarray(meds, dtype=float)
        meds = meds[~np.isnan(meds)]
        if len(meds) < 2:
            return 0.0
        return float(np.max(meds) - np.min(meds))

    T_obs = T(labels)

    ge = 0
    for _ in range(n_perm):
        perm = labels.copy()
        rng.shuffle(perm)
        if T(perm) >= T_obs - 1e-12:
            ge += 1

    return (1 + ge) / (1 + n_perm), T_obs


def infer_by_transition(df_long: pd.DataFrame, N_min: int, n_perm=5000, seed0=123) -> pd.DataFrame:
    group_col = "lab_date"
    groups = sorted(df_long[group_col].dropna().unique().tolist())

    rows = []
    for n in range(1, N_min + 1):
        sub = df_long[df_long["n"] == n][["lab", "d", group_col, "laser", "g_norm"]].dropna()
        values = sub["g_norm"].to_numpy()
        labels = sub[group_col].to_numpy()

        p, Tobs = perm_pvalue_T(values, labels, groups, n_perm=n_perm, seed=seed0 + int(n))

        med_by_group = sub.groupby(group_col)["g_norm"].median().sort_values(ascending=False)

        grp_top = med_by_group.index[0] if len(med_by_group) else None
        grp_2nd = med_by_group.index[1] if len(med_by_group) > 1 else None
        grp_bot = med_by_group.index[-1] if len(med_by_group) else None

        def split_lab_date(x):
            if x is None or "__" not in str(x):
                return None, None
            a, b = str(x).split("__", 1)
            return a, b

        lab_top, d_top = split_lab_date(grp_top)
        lab_2nd, d_2nd = split_lab_date(grp_2nd)
        lab_bot, d_bot = split_lab_date(grp_bot)

        rows.append({
            "n": n,
            "grp_top": grp_top,
            "grp_2nd": grp_2nd,
            "grp_bot": grp_bot,
            "lab_top": lab_top,
            "d_top": d_top,
            "lab_2nd": lab_2nd,
            "d_2nd": d_2nd,
            "lab_bot": lab_bot,
            "d_bot": d_bot,
            "T_maxgap": Tobs,
            "p": p
        })

    df_p = pd.DataFrame(rows).sort_values("n", ignore_index=True)
    df_p["q"] = bh_fdr(df_p["p"].to_numpy())
    return df_p

# ============================================================
# 4) Resúmenes listos para reporte (tablas B y C)
# ============================================================
def summarize_inference(df_p: pd.DataFrame, alpha=0.05, K=5):
    sig = df_p[df_p["q"] <= alpha].copy()

    by_group = (
        df_p.groupby(["lab_top", "d_top", "grp_top"], dropna=False)
            .agg(
                wins=("n", "count"),
                sum_T=("T_maxgap", "sum"),
                mean_T=("T_maxgap", "mean"),
                max_T=("T_maxgap", "max"),
            )
            .sort_values(["wins", "sum_T"], ascending=False)
            .reset_index()
    )

    by_lab = (
        df_p.groupby("lab_top", dropna=False)
            .agg(
                wins=("n", "count"),
                sum_T=("T_maxgap", "sum"),
                mean_T=("T_maxgap", "mean"),
                max_T=("T_maxgap", "max"),
            )
            .sort_values(["wins", "sum_T"], ascending=False)
            .reset_index()
            .rename(columns={"lab_top": "lab"})
    )

    df_sorted = df_p.sort_values("T_maxgap", ascending=False).copy()
    topK = df_sorted.head(K)
    share_topK = float(topK["T_maxgap"].sum() / df_sorted["T_maxgap"].sum()) if df_sorted["T_maxgap"].sum() > 0 else 0.0

    conc = pd.DataFrame([{
        "K": K,
        "share_topK": share_topK,
        "topK_transitions": ", ".join(topK["n"].astype(str).tolist())
    }])

    top3 = df_sorted.head(3).copy()

    resumen = {
        "transiciones_analizadas": int(len(df_p)),
        "significativas_q<=0.05": int(len(sig)),
        "grupo_mas_domina": (
            by_group.iloc[0]["grp_top"],
            int(by_group.iloc[0]["wins"])
        ) if len(by_group) else (None, 0),
        "lab_mas_domina": (
            by_lab.iloc[0]["lab"],
            int(by_lab.iloc[0]["wins"])
        ) if len(by_lab) else (None, 0),
        "share_topK": share_topK,
    }

    return sig, by_group, by_lab, conc, top3, resumen

# ============================================================
# FUNCIONES NUEVAS: Guardado de gráficas de barras (g_norm)
# (para integrar al .py anterior sin romper estructura)
# ============================================================

def _date_token_sort_key(token: str):
    """
    Orden robusto para tokens tipo 05Feb26, 19Feb26, etc.
    Si no puede parsear, cae a orden lexicográfico.
    """
    import re

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    m = re.match(r"^(\d{2})([A-Za-z]{3})(\d{2})$", str(token))
    if not m:
        return (9999, 99, 99, str(token))

    dd = int(m.group(1))
    mon = m.group(2).title()
    yy = int(m.group(3))
    mm = month_map.get(mon, 99)

    # asumimos 20yy
    return (2000 + yy, mm, dd, str(token))


def _sorted_lasers_for_plot(lasers):
    """
    Ordena Laser_1, Laser_2, ... Laser_10 de forma natural.
    """
    import re

    def key_fn(x):
        s = str(x)
        m = re.search(r"(\d+)", s)
        return (0, int(m.group(1)), s) if m else (1, 10**9, s)

    return sorted(list(lasers), key=key_fn)


#######=====
def save_gnorm_bar_charts(
    df_long: pd.DataFrame,
    png_dir: Path | str,
    pdf_dir: Path | str,
    metric_slug: str,
    metric_title: str,
    lab_col: str = "lab",
    date_col: str = "d",
    laser_col: str = "laser",
    n_col: str = "n",
    value_col: str = "g_norm",
    date_order: list[str] | None = None,
    bar_width: float = 0.18,
    png_dpi: int = 150,
    pdf_name: str | None = None,
    png_name_suffix: str = "",
    transition_tick_map: dict[int, str] | None = None,
    xtick_rotation: int = 45,
) -> dict:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pathlib import Path

    if df_long is None or len(df_long) == 0:
        raise ValueError("df_long está vacío; no hay datos para graficar.")

    png_dir = Path(png_dir)
    pdf_dir = Path(pdf_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    required = {lab_col, date_col, laser_col, n_col, value_col}
    missing = [c for c in required if c not in df_long.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas en df_long: {missing}")

    dfx = df_long.copy()
    dfx[n_col] = pd.to_numeric(dfx[n_col], errors="coerce")
    dfx[value_col] = pd.to_numeric(dfx[value_col], errors="coerce")
    dfx = dfx.dropna(subset=[lab_col, date_col, laser_col, n_col, value_col]).copy()

    if dfx.empty:
        raise ValueError("df_long quedó vacío tras limpieza de columnas numéricas.")

    dfx[n_col] = dfx[n_col].astype(int)
    dfx[lab_col] = dfx[lab_col].astype(str)
    dfx[date_col] = dfx[date_col].astype(str)

    if date_order is None:
        unique_dates = dfx[date_col].dropna().unique().tolist()
        date_order = sorted(unique_dates, key=_date_token_sort_key)

    present_dates = set(dfx[date_col].unique().tolist())
    date_order = [d for d in date_order if d in present_dates]

    dfx_plot = (
        dfx.groupby([lab_col, date_col, laser_col, n_col], as_index=False)[value_col]
           .median()
           .sort_values([lab_col, laser_col, n_col, date_col], ignore_index=True)
    )

    combos = (
        dfx_plot[[lab_col, laser_col]]
        .drop_duplicates()
        .sort_values([lab_col, laser_col], ignore_index=True)
        .to_dict("records")
    )

    if pdf_name is None:
        pdf_name = f"{metric_slug}_barras_multilab.pdf"

    pdf_path = pdf_dir / pdf_name
    png_paths = []

    with PdfPages(pdf_path) as pdf:
        for rec in combos:
            lab = rec[lab_col]
            laser = rec[laser_col]

            df_one = dfx_plot[
                (dfx_plot[lab_col] == lab) &
                (dfx_plot[laser_col] == laser)
            ].copy()

            if df_one.empty:
                continue

            transitions = sorted(df_one[n_col].dropna().astype(int).unique().tolist())
            x = np.arange(len(transitions))

            fig, ax = plt.subplots(figsize=(12, 6))

            for i, date in enumerate(date_order):
                df_date = df_one[df_one[date_col] == date]
                s_map = dict(zip(df_date[n_col].astype(int).tolist(),
                                 df_date[value_col].astype(float).tolist()))

                values = [float(s_map.get(n, 0.0)) for n in transitions]

                ax.bar(
                    x + i * bar_width,
                    values,
                    width=bar_width,
                    label=str(date)
                )

            ax.set_title(f"{lab} | {laser} — {metric_title}")

            if transition_tick_map is not None:
                tick_labels = [str(transition_tick_map.get(int(n), str(n))) for n in transitions]
                ax.set_xlabel("Inicio de transición (estampa de tiempo)")
            else:
                tick_labels = [str(n) for n in transitions]
                ax.set_xlabel("Transición")

            ax.set_ylabel(f"{value_col} (%)")
            ax.set_xticks(x + bar_width * (len(date_order) - 1) / 2)
            ax.set_xticklabels(tick_labels, rotation=xtick_rotation, ha="right")
            ax.legend(title="Jornada")
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

            safe_lab = str(lab).replace(" ", "_")
            safe_laser = str(laser).replace(" ", "_")
            if png_name_suffix:
                png_file = f"{safe_lab}_{safe_laser}_{metric_slug}_barras{png_name_suffix}.png"
            else:
                png_file = f"{safe_lab}_{safe_laser}_{metric_slug}_barras.png"

            png_path = png_dir / png_file
            fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
            png_paths.append(png_path)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return {
        "pdf": pdf_path,
        "pngs": png_paths,
        "date_order": date_order,
        "combos": combos,
    }

#=================


def save_gnorm_bar_charts_from_pipeline(
    df_long: pd.DataFrame,
    data_path: Path,
    date_order: list[str] | None = None,
    trans_slot_df: pd.DataFrame | None = None,
    value_col: str = "g_norm",
    metric_slug: str = "global",
    metric_title: str = "Distribución de cambio global (g_norm)",
) -> dict:

    
    # Construir mapa n -> estampa temporal de referencia (si se proporciona)
    transition_tick_map = None
    if trans_slot_df is not None and not trans_slot_df.empty:
        if {"n", "transition_label_start_ref"}.issubset(set(trans_slot_df.columns)):
            _tmp = trans_slot_df[["n", "transition_label_start_ref"]].copy()
            _tmp["n"] = pd.to_numeric(_tmp["n"], errors="coerce").astype("Int64")
            _tmp = _tmp.dropna(subset=["n"]).drop_duplicates(subset=["n"], keep="first")

            # Mapa por n; etiqueta = estampa de inicio
            transition_tick_map = {
                int(row["n"]): str(row["transition_label_start_ref"])
                for _, row in _tmp.iterrows()
                if pd.notna(row["transition_label_start_ref"])
            }
    fig_root = Path(data_path) / "figures" / metric_slug
    png_dir = fig_root / "png"
    pdf_dir = fig_root / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    #
    
    return save_gnorm_bar_charts(
        df_long=df_long,
        png_dir=png_dir,
        pdf_dir=pdf_dir,
        metric_slug=metric_slug,
        metric_title=metric_title,
        date_col="d",
        laser_col="laser",
        n_col="n",
        value_col=value_col,
        date_order=date_order,
        png_name_suffix="",
        transition_tick_map=transition_tick_map,
        xtick_rotation=45,
    )

# ============================================================
# FUNCIONES NUEVAS: mapeo de transiciones (n) a estampas de tiempo
# SOLO PARA REPORTE (NO TOCAR CALCULOS)
# ============================================================

def load_measurement_time_bounds_json(project_root: Path) -> dict:
    """
    Carga measurement_time_bounds.json desde la raíz de Feb26_clean:
      PROJECT_ROOT / "Feb26_clean" / "measurement_time_bounds.json"
    """
    mt_path = project_root / "Mar26_clean" / "measurement_time_bounds.json"
    if not mt_path.is_file():
        raise FileNotFoundError(
            "No se encontró measurement_time_bounds.json en:\n"
            f"  {mt_path}\n\n"
            "Ubicación esperada (según tu estructura): PROJECT_ROOT/Feb26_clean/"
        )

    import json
    with mt_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("measurement_time_bounds.json debe ser un objeto JSON (dict).")

    return obj


def _parse_mid_date_and_index(mid: str) -> tuple[str | None, int | None]:
    """
    Extrae:
      - fecha token (ej. 19Feb26)
      - índice de medición (ej. 001 -> 1)
    desde mid tipo: 19Feb26_M_Epsilon_001
    """
    m = re.match(r"^(\d{2}[A-Za-z]{3}\d{2})_M_[^_]+_(\d{3})$", str(mid))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def measurement_time_bounds_to_df(mt_obj: dict, lab: str = "LAB_UNK") -> pd.DataFrame:
    rows = []
    for key, rec in mt_obj.items():
        if not isinstance(rec, dict):
            continue

        mid = rec.get("mid", key)
        d, m_idx = _parse_mid_date_and_index(mid)

        rows.append({
            "lab": lab,
            "mid_key": key,
            "mid": mid,
            "date": d,
            "m_idx": m_idx,
            "raw_path": rec.get("raw_path"),
            "raw_exists": rec.get("raw_exists"),
            "status": rec.get("status"),
            "notes": rec.get("notes"),
            "t_sys_start": rec.get("t_sys_start"),
            "t_sys_end": rec.get("t_sys_end"),
            "t_rel_start": rec.get("t_rel_start"),
            "t_rel_end": rec.get("t_rel_end"),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if "m_idx" in df.columns:
        df["m_idx"] = pd.to_numeric(df["m_idx"], errors="coerce").astype("Int64")

    df = df.sort_values(["lab", "date", "m_idx", "mid"], ignore_index=True)
    return df

def build_transition_time_lookup(mt_df: pd.DataFrame) -> pd.DataFrame:
    if mt_df is None or mt_df.empty:
        return pd.DataFrame(columns=[
            "lab", "d", "n", "mid_from", "mid_to",
            "t_start_transition", "t_end_transition",
            "t_from_end", "t_to_start",
            "transition_label_start", "transition_label_range", "transition_label_full"
        ])

    required = {"lab", "date", "m_idx", "mid", "t_sys_start", "t_sys_end"}
    missing = [c for c in required if c not in mt_df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en mt_df: {missing}")

    rows = []

    for (lab, d), g in mt_df.groupby(["lab", "date"], dropna=True):
        g = g.copy().sort_values("m_idx", ignore_index=True)
        recs = g.to_dict("records")

        for i in range(len(recs) - 1):
            a = recs[i]
            b = recs[i + 1]

            if pd.isna(a.get("m_idx")) or pd.isna(b.get("m_idx")):
                continue

            m_from = int(a["m_idx"])
            m_to = int(b["m_idx"])

            if m_to != m_from + 1:
                continue

            n = m_from
            t_start = a.get("t_sys_start")
            t_end   = b.get("t_sys_end")
            t_from_end = a.get("t_sys_end")
            t_to_start = b.get("t_sys_start")

            rows.append({
                "lab": lab,
                "d": d,
                "n": n,
                "mid_from": a.get("mid"),
                "mid_to": b.get("mid"),
                "t_start_transition": t_start,
                "t_end_transition": t_end,
                "t_from_end": t_from_end,
                "t_to_start": t_to_start,
                "transition_label_start": str(t_start) if t_start is not None else None,
                "transition_label_range": f"{t_start} → {t_end}" if (t_start is not None and t_end is not None) else None,
                "transition_label_full": (
                    f"{t_start} → {t_end}  ({a.get('mid')} -> {b.get('mid')})"
                    if (t_start is not None and t_end is not None) else None
                ),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")
        out = out.sort_values(["lab", "d", "n"], ignore_index=True)

    return out

def build_transition_slot_lookup_for_global_reports(
    trans_time_df: pd.DataFrame,
    prefer_date: str | None = None
) -> pd.DataFrame:
    """
    Construye un mapeo por n (sin fecha) para tablas globales (ej. df_p),
    donde no existe columna de fecha y por tanto no puede ponerse una estampa
    exacta por jornada.

    Estrategia:
      - si prefer_date se indica y existe, usa esa fecha como referencia;
      - si no, usa la primera fecha disponible.
    """
    if trans_time_df is None or trans_time_df.empty:
        return pd.DataFrame(columns=["n", "transition_label_start_ref", "transition_label_range_ref"])

    required = {"d", "n", "transition_label_start", "transition_label_range"}
    missing = [c for c in required if c not in trans_time_df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en trans_time_df: {missing}")

    dfx = trans_time_df.copy()

    dates = [x for x in dfx["d"].dropna().astype(str).unique().tolist()]
    if not dates:
        return pd.DataFrame(columns=["n", "transition_label_start_ref", "transition_label_range_ref"])

    if (prefer_date is None) or (prefer_date not in dates):
        # usar la primera fecha ordenada
        dates_sorted = sorted(dates)
        prefer_date = dates_sorted[0]

    ref = dfx[dfx["d"].astype(str) == str(prefer_date)].copy()
    ref = ref.sort_values("n", ignore_index=True)

    out = ref[["n", "transition_label_start", "transition_label_range"]].copy()
    out = out.rename(columns={
        "transition_label_start": "transition_label_start_ref",
        "transition_label_range": "transition_label_range_ref",
    })

    return out


def attach_transition_labels_for_report(
    df_report: pd.DataFrame,
    trans_time_df: pd.DataFrame,
    *,
    lab_col: str = "lab",
    date_col: str = "d",
    n_col: str = "n",
    replace_n_with_start_label: bool = True,
    keep_n_original: bool = True,
    label_col_name: str = "transicion",
    range_col_name: str = "transicion_rango",
) -> pd.DataFrame:
    if df_report is None:
        return df_report

    dfx = df_report.copy()
    if dfx.empty:
        return dfx

    if lab_col not in dfx.columns or date_col not in dfx.columns or n_col not in dfx.columns:
        raise KeyError(
            f"Se requieren columnas '{lab_col}', '{date_col}' y '{n_col}'."
        )

    need_cols = ["lab", "d", "n", "transition_label_start", "transition_label_range"]
    missing = [c for c in need_cols if c not in trans_time_df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en trans_time_df: {missing}")

    dfx[n_col] = pd.to_numeric(dfx[n_col], errors="coerce").astype("Int64")
    dfx[lab_col] = dfx[lab_col].astype(str)
    dfx[date_col] = dfx[date_col].astype(str)

    lut = trans_time_df[["lab", "d", "n", "transition_label_start", "transition_label_range"]].copy()
    lut["n"] = pd.to_numeric(lut["n"], errors="coerce").astype("Int64")
    lut["lab"] = lut["lab"].astype(str)
    lut["d"] = lut["d"].astype(str)

    dfx = dfx.merge(
        lut,
        left_on=[lab_col, date_col, n_col],
        right_on=["lab", "d", "n"],
        how="left",
        suffixes=("", "_lut")
    )

    if keep_n_original and n_col in dfx.columns and "n_original" not in dfx.columns:
        dfx["n_original"] = dfx[n_col]

    dfx[label_col_name] = dfx["transition_label_start"]
    dfx[range_col_name] = dfx["transition_label_range"]

    if replace_n_with_start_label:
        dfx[n_col] = dfx[label_col_name]

    return dfx


def attach_transition_labels_global_report(
    df_report: pd.DataFrame,
    trans_slot_df: pd.DataFrame,
    *,
    n_col: str = "n",
    replace_n_with_start_label: bool = True,
    keep_n_original: bool = True,
    label_col_name: str = "transicion",
    range_col_name: str = "transicion_rango",
) -> pd.DataFrame:
    """
    Versión para tablas globales SIN columna de fecha (ej. df_p, top3, sig),
    usando un mapeo de referencia por n (slot temporal de una fecha de referencia).
    """
    if df_report is None:
        return df_report

    dfx = df_report.copy()
    if dfx.empty:
        return dfx

    if n_col not in dfx.columns:
        raise KeyError(f"Se requiere columna '{n_col}' en df_report.")

    need_cols = {"n", "transition_label_start_ref", "transition_label_range_ref"}
    missing = [c for c in need_cols if c not in trans_slot_df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en trans_slot_df: {missing}")

    dfx[n_col] = pd.to_numeric(dfx[n_col], errors="coerce").astype("Int64")

    lut = trans_slot_df.copy()
    lut["n"] = pd.to_numeric(lut["n"], errors="coerce").astype("Int64")

    dfx = dfx.merge(lut, on="n", how="left")

    if keep_n_original and "n_original" not in dfx.columns:
        dfx["n_original"] = dfx[n_col]

    dfx[label_col_name] = dfx["transition_label_start_ref"]
    dfx[range_col_name] = dfx["transition_label_range_ref"]

    if replace_n_with_start_label:
        dfx[n_col] = dfx[label_col_name]

    return dfx

def prepare_all_outputs_with_time_labels(
    *,
    df_long: pd.DataFrame,
    df_aggs: pd.DataFrame,
    df_p: pd.DataFrame,
    sig: pd.DataFrame,
    top3: pd.DataFrame,
    trans_time_df: pd.DataFrame,
    trans_slot_df: pd.DataFrame,
) -> dict:
    """
    Convierte TODOS los artefactos de salida a versiones con estampas
    temporales en lugar de n. No altera los originales.

    Devuelve dict con:
      df_long, df_aggs, df_p, sig, top3  (versiones *_out)
    """
    # Tablas con fecha + n
    df_long_out = attach_transition_labels_for_report(
        df_long,
        trans_time_df,
        date_col="d",
        n_col="n",
        replace_n_with_start_label=True,
        keep_n_original=True,
        label_col_name="transicion",
        range_col_name="transicion_rango",
    )

    df_aggs_out = attach_transition_labels_for_report(
        df_aggs,
        trans_time_df,
        date_col="d",
        n_col="n",
        replace_n_with_start_label=True,
        keep_n_original=True,
        label_col_name="transicion",
        range_col_name="transicion_rango",
    )

    # Tablas globales sin fecha (usa referencia por n)
    df_p_out = attach_transition_labels_global_report(
        df_p,
        trans_slot_df,
        n_col="n",
        replace_n_with_start_label=True,
        keep_n_original=True,
        label_col_name="transicion",
        range_col_name="transicion_rango",
    )

    sig_out = attach_transition_labels_global_report(
        sig,
        trans_slot_df,
        n_col="n",
        replace_n_with_start_label=True,
        keep_n_original=True,
        label_col_name="transicion",
        range_col_name="transicion_rango",
    )

    top3_out = attach_transition_labels_global_report(
        top3,
        trans_slot_df,
        n_col="n",
        replace_n_with_start_label=True,
        keep_n_original=True,
        label_col_name="transicion",
        range_col_name="transicion_rango",
    )

    return {
        "df_long": df_long_out,
        "df_aggs": df_aggs_out,
        "df_p": df_p_out,
        "sig": sig_out,
        "top3": top3_out,
    }

def write_json_sidecar(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        
# ============================================================
# MAIN (MODIFICADO: TODO PRODUCTO SALE CON ESTAMPAS TEMPORALES)
# ============================================================

if __name__ == "__main__":
    print("SCRIPT_DIR   :", SCRIPT_DIR)
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATA_PATH    :", DATA_PATH)

    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            "No se encontró la carpeta esperada:\n"
            f"  {DATA_PATH}\n\n"
            "Coloca este .py en el PROJECT_ROOT (o dentro de una subcarpeta) "
            "que contenga la estructura:\n"
            "  Feb26_clean/Reports/Level4_Change/BlindSequential"
        )

    # Detectar archivos
    xlsx_files = sorted(DATA_PATH.rglob("transitions_per_laser_*.xlsx"))
    json_files = sorted(DATA_PATH.rglob("blind_seq_meta_*.json"))

    print("\nExcel encontrados:")
    for f in xlsx_files:
        print("  -", f.name)

    print("\nJSON encontrados:")
    for f in json_files:
        print("  -", f.name)

    DATES = sorted({extract_date_from_filename(f.name) for f in xlsx_files if extract_date_from_filename(f.name)})
    print("\nJornadas detectadas:", DATES)

    # ------------------------------------------------------------
    # (A) .TEX preliminar: detección xlsx/json + jornadas
    # ------------------------------------------------------------
    FIG_ROOT = DATA_PATH / "figures"
    TEX_ROOT = DATA_PATH / "tex"
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TEX_ROOT.mkdir(parents=True, exist_ok=True)

    tex_path = TEX_ROOT / "pipeline_report.tex"
    ctx0 = {
        "SCRIPT_DIR": SCRIPT_DIR,
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_PATH": DATA_PATH,
        "xlsx_files": xlsx_files,
        "json_files": json_files,
        "DATES": DATES,
    }
    write_tex_report(tex_path, build_basic_report_tex(ctx0))

    tex_meta_prelim = {
        "tex_path": str(tex_path),
        "SCRIPT_DIR": str(SCRIPT_DIR),
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_PATH": str(DATA_PATH),
        "xlsx_files": [str(p) for p in xlsx_files],
        "json_files": [str(p) for p in json_files],
        "DATES": DATES,
    }
    write_json_sidecar(TEX_ROOT / "pipeline_report_prelim_meta.json", tex_meta_prelim)


#=================================    
    # Cargar todo
    data = load_all_books(DATA_PATH)
    validation_df = validate_internal_sums(data)

    labs_loaded = sorted(data.keys())
    print("\nLaboratorios cargados:", labs_loaded)

    if not data:
        raise RuntimeError("No se cargó ningún archivo. Verifica que existan transitions_per_laser_*.xlsx en DATA_PATH.")

    lab_example = labs_loaded[0]
    dates_loaded = sorted(data[lab_example].keys())
    print(f"Fechas cargadas en {lab_example}:", dates_loaded)

    first_date = dates_loaded[0]
    print(f"Hojas ejemplo ({lab_example}, primera fecha):", list(data[lab_example][first_date].keys()))

    date_example = dates_loaded[min(3, len(dates_loaded) - 1)]
    laser_keys = sorted(data[lab_example][date_example].keys())
    laser_example = laser_keys[min(3, len(laser_keys) - 1)]
    df_example = data[lab_example][date_example][laser_example]

#=========================================
    print("\nEjemplo:")
    print("Laboratorio:", lab_example)
    print("Fecha:", date_example)
    print("Laser:", laser_example)
    print("\nHead(10):")
    print(df_example.head(10).to_string(index=False))

    # ============================================================
    # (B) Pipeline limpio (CALCULO INTERNO con n)
    # ============================================================
    df_long = build_df_long(data=data, FEATURES=FEATURES)

    # ------------------------------------------------------------
    # (B.0) Cargar estampas de tiempo (capa de salida)
    # ------------------------------------------------------------
    mt_obj = load_measurement_time_bounds_json(PROJECT_ROOT)
    mt_df = measurement_time_bounds_to_df(mt_obj, lab=lab_example)
    
    trans_time_df = build_transition_time_lookup(mt_df)

    # Para tablas globales sin fecha (df_p, sig, top3) usamos referencia por n
    trans_slot_df = build_transition_slot_lookup_for_global_reports(
        trans_time_df,
        prefer_date=None  # o fija "19Feb26" si quieres una referencia explícita
    )

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # (B.1) Guardar gráficas de barras (global + componentes)
    #       IMPORTANTE: usar df_long ORIGINAL (n numérico)
    # ------------------------------------------------------------
    metric_specs = [
        {
            "metric_slug": "global",
            "value_col": "g_norm",
            "metric_title": "Distribución de cambio global (g_norm)",
        },
        {
            "metric_slug": "forma",
            "value_col": "pct_forma",
            "metric_title": "Distribución porcentual (forma)",
        },
        {
            "metric_slug": "mov",
            "value_col": "pct_mov",
            "metric_title": "Distribución porcentual (movimiento)",
        },
        {
            "metric_slug": "ene",
            "value_col": "pct_ene",
            "metric_title": "Distribución porcentual (energía)",
        },
        {
            "metric_slug": "fft",
            "value_col": "pct_fft",
            "metric_title": "Distribución porcentual (fft)",
        },
    ]

    all_bar_artifacts = {}

    for spec in metric_specs:
        artifacts = save_gnorm_bar_charts_from_pipeline(
            df_long=df_long,       # ORIGINAL, no etiquetado (cálculo)
            data_path=DATA_PATH,
            # date_order=["19Feb26", "20Feb26", "21Feb26", "22Feb26"],  # opcional
            trans_slot_df=trans_slot_df,
            value_col=spec["value_col"],
            metric_slug=spec["metric_slug"],
            metric_title=spec["metric_title"],
        )
        all_bar_artifacts[spec["metric_slug"]] = artifacts

    print("\nGráficas de barras guardadas por métrica:")
    for metric_slug, artifacts in all_bar_artifacts.items():
        print(f"\n[{metric_slug}]")
        print("PDF :", artifacts["pdf"])
        print("PNGs:")
        for p in artifacts["pngs"]:
            print("  -", p.name)

    # ------------------------------------------------------------
    # Pipeline inferencial (cálculo con n original)
    # ------------------------------------------------------------
    N_min, trans_counts = compute_N_min(df_long)
    show(f"N_min (rango comparable) = {N_min}", trans_counts.sort_values(["d", "ell"]))

    df_aggs, dfN = build_df_aggs(df_long, N_min)

    # vista rápida opcional (de cálculo puro, se queda con n)
    loc_wide = (
        df_aggs.assign(col_key=df_aggs["lab"].astype(str) + "__" + df_aggs["d"].astype(str))
            .pivot(index="n", columns="col_key", values="Loc")
            .sort_index()
    )
    show("Loc por fecha (wide) para inspección", loc_wide)

    print("\nSanity Dom min/max:", float(df_aggs["Dom"].min()), float(df_aggs["Dom"].max()))

    # ---- inferencia mínima por transición (cálculo con n) ----
    df_p = infer_by_transition(df_long, N_min=N_min, n_perm=5000, seed0=123)
    sig, by_group, by_lab, conc, top3, resumen = summarize_inference(df_p, alpha=0.05, K=5)
    # ------------------------------------------------------------
    # (B.2) Capa de SALIDA UNIFICADA:
    #       TODO producto/tablas/exportables sale con estampas
    # ------------------------------------------------------------
    outputs = prepare_all_outputs_with_time_labels(
        df_long=df_long,
        df_aggs=df_aggs,
        df_p=df_p,
        sig=sig,
        top3=top3,
        trans_time_df=trans_time_df,
        trans_slot_df=trans_slot_df,
    )

    df_long_out = outputs["df_long"]
    df_aggs_out = outputs["df_aggs"]
    df_p_out = outputs["df_p"]
    sig_out = outputs["sig"]
    top3_out = outputs["top3"]

    # Mostrar SIEMPRE outputs etiquetados (productos)
    show("df_long (producto con estampa de inicio)", df_long_out)
    show("df_aggs (Loc/Abs/IQR/Dom) con estampas de tiempo", df_aggs_out)
    show("Tabla inferencial por transición (df_p) con estampa de inicio", df_p_out, max_rows=30)
    show(
        "Transiciones significativas (q<=0.05) con estampa de inicio",
        sig_out.sort_values("T_maxgap", ascending=False),
        max_rows=30
    )

    show("Resumen por grupo lab+fecha (wins e intensidad)", by_group, max_rows=20)
    show("Resumen por laboratorio", by_lab, max_rows=20)
    show("Concentración del efecto (Top-K)", conc)
    show("Top 3 transiciones por intensidad con estampa de inicio", top3_out)

    # ---- bloque de salida tipo consola (ordenado) ----
    print("\n" + "="*70)
    print("RESUMEN INFERENCIAL (MINIMO REPORTABLE)")
    print("="*70)
    print(f"Transiciones analizadas: {resumen['transiciones_analizadas']}")
    print(f"Significativas (q<=0.05): {resumen['significativas_q<=0.05']}")

    grp_dom, cnt = resumen["grupo_mas_domina"]
    print(f"\nGrupo con mayor frecuencia de dominancia: {grp_dom} ({cnt} transiciones)")

    lab_dom, cnt_lab = resumen["lab_mas_domina"]
    print(f"\nLaboratorio con mayor frecuencia de dominancia: {lab_dom} ({cnt_lab} transiciones)")
    
    print(f"\nConcentración del efecto en Top-{int(conc.loc[0,'K'])}: {100*resumen['share_topK']:0.2f}%")
    print("="*70)

    # ------------------------------------------------------------
    # (C) Ejemplo de hoja -> versión de producto con estampas
    
    # ------------------------------------------------------------
    df_example_out = df_example.copy()
    df_example_out["lab"] = str(lab_example)
    df_example_out["d"] = str(date_example)
    df_example_out = attach_transition_labels_for_report(
        df_example_out,
        trans_time_df,
        lab_col="lab",
        date_col="d",
        n_col="n",
        replace_n_with_start_label=True,
        keep_n_original=True,
        label_col_name="transicion",
        range_col_name="transicion_rango",
    )
    # ------------------------------------------------------------
    # (D) Re-escritura del .TEX final
    #     PASAR SOLO PRODUCTOS *_out (capa de salida)
    # ------------------------------------------------------------
    ctxF = dict(ctx0)
    ctxF.update({
        "fechas_cargadas": dates_loaded,
        "hojas_ejemplo": list(data[lab_example][first_date].keys()),
        
        "validation_df": validation_df,

        "date_example": date_example,
        "laser_example": laser_example,
        "df_example": df_example_out,
        "df_example_cols": df_example_out.columns.tolist(),

        # Productos etiquetados con estampas (salida)
        "df_long": df_long_out,
        "trans_counts": trans_counts,      # no depende de n de transición
        "N_min": N_min,
        "df_aggs": df_aggs_out,
        "loc_wide": loc_wide,              # vista técnica interna (puede quedarse con n)
        "dom_min": float(df_aggs["Dom"].min()),
        "dom_max": float(df_aggs["Dom"].max()),
        "df_p": df_p_out,
        "sig": sig_out,
        "by_group": by_group,
        "by_lab": by_lab,
        "conc": conc,
        "top3": top3_out,
        "resumen": resumen,

        # Metadatos de estampas
        "measurement_time_bounds_rows": len(mt_df),
        "transition_time_lookup_rows": len(trans_time_df),
        "transition_slot_lookup_rows": len(trans_slot_df),

        # Artefactos de gráficas (rutas)
        "figures_root": str(FIG_ROOT),
        "tex_root": str(TEX_ROOT),
        "bar_artifacts_by_metric": {
            k: {
                "pdf": str(v["pdf"]),
                "pngs": [str(p) for p in v["pngs"]],
                "date_order": v.get("date_order"),
                "combos": v.get("combos", []),
            }
            for k, v in all_bar_artifacts.items()
        },
    })
    write_tex_report(tex_path, build_basic_report_tex(ctxF))
tex_meta_final = {
    "tex_path": str(tex_path),
    "figures_root": str(FIG_ROOT),
    "tex_root": str(TEX_ROOT),
    "fechas_cargadas": dates_loaded,
    "labs_cargados": labs_loaded,
    "N_min": int(N_min) if N_min is not None else None,
    "resumen": resumen,
    "measurement_time_bounds_rows": int(len(mt_df)),
    "transition_time_lookup_rows": int(len(trans_time_df)),
    "transition_slot_lookup_rows": int(len(trans_slot_df)),
    "rows": {
        "validation_df": int(len(validation_df)),
        "df_long_out": int(len(df_long_out)),
        "df_aggs_out": int(len(df_aggs_out)),
        "df_p_out": int(len(df_p_out)),
        "sig_out": int(len(sig_out)),
        "by_group": int(len(by_group)),
        "by_lab": int(len(by_lab)),
        "conc": int(len(conc)),
        "top3_out": int(len(top3_out)),
    },
    "bar_artifacts_by_metric": {
        k: {
            "pdf": str(v["pdf"]),
            "pngs": [str(p) for p in v["pngs"]],
            "date_order": v.get("date_order"),
            "combos": v.get("combos", []),
        }
        for k, v in all_bar_artifacts.items()
    },
}
write_json_sidecar(TEX_ROOT / "pipeline_report_final_meta.json", tex_meta_final)

pipeline_artifacts_manifest = {
    "tex": {
        "report": str(tex_path),
        "prelim_meta": str(TEX_ROOT / "pipeline_report_prelim_meta.json"),
        "final_meta": str(TEX_ROOT / "pipeline_report_final_meta.json"),
    },
    "figures": {
        k: {
            "pdf": str(v["pdf"]),
            "pngs": [str(p) for p in v["pngs"]],
        }
        for k, v in all_bar_artifacts.items()
    },
}
write_json_sidecar(TEX_ROOT / "pipeline_artifacts_manifest.json", pipeline_artifacts_manifest)    