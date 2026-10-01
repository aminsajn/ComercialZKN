import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import unicodedata
import re
import json

st.set_page_config(page_title="Zukan PIC", page_icon="chart_with_upwards_trend", layout="wide")

ZUKAN_BLACK = "#2E2A25"
ZUKAN_BLUE  = "#0082CA"
ZUKAN_LBLUE = "#57C9E8"
ZUKAN_RED   = "#CC003D"
ZUKAN_GOLD  = "#CEA052"
ZUKAN_GREEN = "#00AD68"

st.markdown(f"""<style>
  [data-testid="stSidebar"] {{ background-color: #E8F5EE; }}
  [data-testid="stSidebar"] * {{ color: {ZUKAN_BLACK} !important; }}
  .block-container {{ padding-top: 2rem; }}
  div.stButton > button {{
      width: 100%;
      padding: 0.75rem 1rem;
      font-size: 0.95rem;
      font-weight: 400;
      border-radius: 6px;
      border: 1px solid transparent;
      background-color: transparent;
      color: {ZUKAN_BLACK};
      transition: all 0.15s;
  }}
  div.stButton > button:hover {{
      border-color: #d0d0d0;
      background-color: #f5f5f5;
  }}
</style>""", unsafe_allow_html=True)

ANOS = [2021, 2022, 2023, 2024, 2025, 2026]

def eu(val, d=3):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "-"
    return f"{val:,.{d}f}".replace(",","X").replace(".",",").replace("X",".")

def eu_s(val, d=2):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "-"
    return f"{val:+,.{d}f}".replace(",","X").replace(".",",").replace("X",".")

def eupct(val, d=1):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "-"
    return f"{val:+.{d}f}%".replace(".",",")

def eutn(val, signed=False):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "-"
    fmt = f"{val:+,.0f}" if signed else f"{val:,.0f}"
    return fmt.replace(",",".")

def _norm(s):
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[̀-ͯ]", "", s)
    return s

_SECTOR_ALIAS = {
    "barritas y bebidas deportivas":      "barritas y bebidas deport",
    "especias, aromas y colorantes":      "especias, aromas,colorant",
    "lacteos, yogures y postres lacteos": "lacteos, yogur, postr lac",
    "especias aromas y colorantes":       "especias, aromas,colorant",
}

def _norm_sector(s):
    if pd.isna(s):
        return None
    k = _norm(s)
    return _SECTOR_ALIAS.get(k, k)

if "pagina" not in st.session_state:
    st.session_state["pagina"] = "home"

@st.cache_data(show_spinner="Cargando datos...")
def load_data():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    fac = (df.groupby(["cod_cliente","nombre_comercial","ejercicio"], as_index=False)
             .agg(fac_M=("total_base_neto","sum")))
    fac["fac_M"] = (fac["fac_M"] / 1e6).round(3)
    return fac

@st.cache_data(show_spinner="Calculando detalle de productos...")
def load_bridge_data():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    cpx = (df.groupby(
               ["cod_cliente","nombre_comercial","cod_producto","descripcion_producto","ejercicio"],
               as_index=False)
             .agg(facturacion=("total_base_neto","sum"), toneladas=("Cantidad_TN","sum")))
    # pvp_medio: media aritmética del PvP de cada línea de pedido (eur_por_tn)
    # distinto de fac/tn que es una media ponderada por volumen
    pvp_linea = (df[df["eur_por_tn"].notna() & (df["Cantidad_TN"] > 0)]
                 .groupby(["cod_cliente","cod_producto","ejercicio"])["eur_por_tn"]
                 .mean().round(2).reset_index()
                 .rename(columns={"eur_por_tn": "pvp_medio"}))
    cpx = cpx.merge(pvp_linea, on=["cod_cliente","cod_producto","ejercicio"], how="left")
    # pvp ponderado por volumen (se mantiene para compatibilidad)
    cpx["pvp"] = (cpx["facturacion"] / cpx["toneladas"].replace(0, np.nan)).round(2)
    return cpx

def compute_bridge(cpx, cod_cliente, ano_ant, ano_act):
    ant = cpx[(cpx["cod_cliente"] == cod_cliente) & (cpx["ejercicio"] == ano_ant)].copy()
    act = cpx[(cpx["cod_cliente"] == cod_cliente) & (cpx["ejercicio"] == ano_act)].copy()

    fac_ant_total = ant["facturacion"].sum()
    fac_act_total = act["facturacion"].sum()
    delta_fac     = fac_act_total - fac_ant_total
    pct_fac       = (delta_fac / fac_ant_total * 100) if fac_ant_total > 0 else np.nan

    merged = ant.merge(act, on=["cod_cliente","cod_producto"],
                       suffixes=("_ant","_act"), how="outer")
    merged["descripcion"] = merged["descripcion_producto_ant"].combine_first(
                                merged["descripcion_producto_act"])

    mask_both   = merged["facturacion_ant"].notna() & merged["facturacion_act"].notna()
    mask_gained = merged["facturacion_ant"].isna()  & merged["facturacion_act"].notna()
    mask_lost   = merged["facturacion_ant"].notna()  & merged["facturacion_act"].isna()

    both   = merged[mask_both].copy()
    gained = merged[mask_gained].copy()
    lost   = merged[mask_lost].copy()

    tn_ant = both["toneladas_ant"].fillna(0).sum()
    tn_act = both["toneladas_act"].fillna(0).sum()

    # PvP medio: media aritmética de los PvP por línea de pedido (no ponderada por volumen)
    pvp_ant = both["pvp_medio_ant"].mean() if both["pvp_medio_ant"].notna().any() else np.nan
    pvp_act = both["pvp_medio_act"].mean() if both["pvp_medio_act"].notna().any() else np.nan

    both["delta_tn"]  = both["toneladas_act"]  - both["toneladas_ant"]
    both["delta_pvp"] = both["pvp_medio_act"].fillna(both["pvp_act"]) - both["pvp_medio_ant"].fillna(both["pvp_ant"])
    both["delta_fac"] = both["facturacion_act"] - both["facturacion_ant"]

    e3 = both[(both["delta_tn"].abs() > 0.001) & (both["delta_pvp"].abs() > 0.01)].copy()

    return {
        "fac_ant": fac_ant_total, "fac_act": fac_act_total,
        "delta_fac": delta_fac,   "pct_fac": pct_fac,
        "tn_ant": tn_ant,         "tn_act": tn_act,
        "pvp_ant": pvp_ant,       "pvp_act": pvp_act,
        "e3": e3, "e4": gained,   "e5": lost,
    }

@st.cache_data(show_spinner="Calculando sectores...")
def load_sector_data():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    cli_sector = (df.groupby("cod_cliente")[["Sector_1","Sector_2","Sector_3"]]
                  .agg(lambda x: x.mode()[0] if x.notna().any() else None)
                  .reset_index())
    cli_fac = (df.groupby(["cod_cliente","ejercicio"])
                 .agg(facturacion=("total_base_neto","sum"))
                 .reset_index())
    merged = cli_fac.merge(cli_sector, on="cod_cliente", how="left")
    sector_avg = (merged.groupby(["Sector_3","ejercicio"])
                  .agg(fac_media=("facturacion","mean"),
                       n_clientes=("cod_cliente","nunique"))
                  .reset_index())
    return cli_sector, sector_avg

@st.cache_data(show_spinner=False)
def load_sector_pvp_evol():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    df = df[df["Sector_3"].notna()].copy()
    pvp = (df[df["eur_por_tn"].notna() & (df["Cantidad_TN"] > 0)]
           .groupby(["Sector_3","ejercicio"])["eur_por_tn"]
           .mean().round(2).reset_index()
           .rename(columns={"eur_por_tn":"pvp_medio"}))
    vol = (df.groupby(["Sector_3","ejercicio"])
             .agg(toneladas=("Cantidad_TN","sum"))
             .reset_index())
    return pvp.merge(vol, on=["Sector_3","ejercicio"], how="outer")

@st.cache_data(show_spinner=False)
def load_familia_pvp_evol():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026) & df["familia"].notna()].copy()
    pvp = (df[df["eur_por_tn"].notna() & (df["Cantidad_TN"] > 0)]
           .groupby(["familia","ejercicio"])["eur_por_tn"]
           .mean().round(2).reset_index().rename(columns={"eur_por_tn":"pvp_medio"}))
    vol = df.groupby(["familia","ejercicio"]).agg(toneladas=("Cantidad_TN","sum")).reset_index()
    return pvp.merge(vol, on=["familia","ejercicio"], how="outer")

@st.cache_data(show_spinner=False)
def load_pais_pvp_evol():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026) & df["cod_pais"].notna()].copy()
    df["pais"] = df["cod_pais"].str.strip().str.upper()
    pvp = (df[df["eur_por_tn"].notna() & (df["Cantidad_TN"] > 0)]
           .groupby(["pais","ejercicio"])["eur_por_tn"]
           .mean().round(2).reset_index().rename(columns={"eur_por_tn":"pvp_medio"}))
    vol = df.groupby(["pais","ejercicio"]).agg(toneladas=("Cantidad_TN","sum")).reset_index()
    return pvp.merge(vol, on=["pais","ejercicio"], how="outer")

@st.cache_data(show_spinner=False)
def load_provincia_pvp_evol():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[(df["ejercicio"].between(2021, 2026)) &
            (df["cod_pais"] == "ES") &
            df["provincia"].notna()].copy()
    df["provincia"] = df["provincia"].str.strip().str.title()
    pvp = (df[df["eur_por_tn"].notna() & (df["Cantidad_TN"] > 0)]
           .groupby(["provincia","ejercicio"])["eur_por_tn"]
           .mean().round(2).reset_index().rename(columns={"eur_por_tn":"pvp_medio"}))
    vol = df.groupby(["provincia","ejercicio"]).agg(toneladas=("Cantidad_TN","sum")).reset_index()
    return pvp.merge(vol, on=["provincia","ejercicio"], how="outer")

@st.cache_data(show_spinner=False)
def load_diagnostico_data():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    df["fac_M"] = df["total_base_neto"] / 1e6
    df = df[df["fac_M"] > 0]

    tot_yr = df.groupby("ejercicio")["fac_M"].sum().round(2)
    cli_yr = df.groupby("ejercicio")["cod_cliente"].nunique()

    top20_pct = {}
    for y in ANOS:
        _d = df[df["ejercicio"] == y].groupby("cod_cliente")["fac_M"].sum()
        _t = _d.sum()
        top20_pct[y] = float(_d.nlargest(20).sum() / _t * 100) if _t > 0 else 0.0

    cli_sets   = {y: set(df[df["ejercicio"] == y]["cod_cliente"]) for y in ANOS}
    churn_new  = {ANOS[i]: len(cli_sets[ANOS[i]] - cli_sets[ANOS[i-1]]) for i in range(1, len(ANOS))}
    churn_lost = {ANOS[i]: len(cli_sets[ANOS[i-1]] - cli_sets[ANOS[i]]) for i in range(1, len(ANOS))}

    fam_yr = df.groupby(["ejercicio", "familia"])["fac_M"].sum().reset_index()

    xls = pd.read_excel("data/Productos_ValorAnadido_Zukan_Actualizado.xlsx")
    xls = xls[xls["Cód. Producto"].notna()].copy()
    xls["cod"] = pd.to_numeric(xls["Cód. Producto"], errors="coerce")
    xls = xls[xls["cod"].notna() & ~xls["Descripción Comercial"].astype(str).str.startswith("TOTAL")]
    lqfb_codes = set(xls["cod"].astype(int))
    lqfb_yr = df[df["cod_producto"].isin(lqfb_codes)].groupby("ejercicio")["fac_M"].sum()
    lqfb_pct = {y: float(lqfb_yr.get(y, 0) / tot_yr.get(y, 1) * 100) for y in ANOS}
    lqfb_val = {y: float(lqfb_yr.get(y, 0)) for y in ANOS}

    def _bridge(yr_a, yr_b, n=6):
        ca = df[df["ejercicio"] == yr_a].groupby(["cod_cliente", "nombre_comercial"])["fac_M"].sum().reset_index().set_index("cod_cliente")
        cb = df[df["ejercicio"] == yr_b].groupby(["cod_cliente", "nombre_comercial"])["fac_M"].sum().reset_index().set_index("cod_cliente")
        all_idx = ca.index.union(cb.index)
        ant = ca["fac_M"].reindex(all_idx).fillna(0)
        act = cb["fac_M"].reindex(all_idx).fillna(0)
        nombres = ca["nombre_comercial"].reindex(all_idx).combine_first(cb["nombre_comercial"].reindex(all_idx))
        delta = act - ant
        # Componentes agregados
        comp = {
            "tot_ant":      float(ant.sum()),
            "tot_act":      float(act.sum()),
            "bajan":        float(delta[(ant > 0) & (act > 0) & (delta < 0)].sum()),
            "suben":        float(delta[(ant > 0) & (act > 0) & (delta > 0)].sum()),
            "nuevos":       float(act[(ant == 0) & (act > 0)].sum()),
            "se_van":       float(-ant[(act == 0) & (ant > 0)].sum()),
        }
        # Top caídas y subidas de clientes existentes
        exist_mask = (ant > 0) & (act > 0)
        top_caidas = (delta[exist_mask & (delta < 0)]
                      .rename(nombres).nsmallest(n))
        top_subidas = (delta[exist_mask & (delta > 0)]
                       .rename(nombres).nlargest(n))
        return comp, top_caidas, top_subidas

    bridge_2425 = _bridge(2024, 2025)
    bridge_2526 = _bridge(2025, 2026)
    nutra_yr   = df[df["Sector_1"] == "nutraceuticos"].groupby("ejercicio")["fac_M"].sum()

    return {
        "tot_yr":     {int(k): float(v) for k, v in tot_yr.items()},
        "cli_yr":     {int(k): int(v)   for k, v in cli_yr.items()},
        "top20_pct":  top20_pct,
        "churn_new":  churn_new,
        "churn_lost": churn_lost,
        "fam_yr":     fam_yr,
        "lqfb_pct":   lqfb_pct,
        "lqfb_val":    lqfb_val,
        "bridge_2425": bridge_2425,
        "bridge_2526": bridge_2526,
        "nutra_yr":   {int(k): float(v) for k, v in nutra_yr.items()},
    }

@st.cache_data(show_spinner="Calculando scoring de leads...")
def load_leads_data():
    leads = pd.read_csv("data/Maestro Leads.csv", encoding="utf-8", low_memory=False)
    excl = {"No Longer Interested", "Canceled", "Cannot Contact"}
    leads = leads[~leads["statuscodename"].isin(excl)].copy()

    # ── Dimensión 5: calidad señal comercial (lead) ──────────────────────────
    status_map  = {"Qualified": 1.0, "Contacted": 0.65, "New": 0.40}
    job_map     = {"CEO": 1.0, "Manager": 0.75, "Technical": 0.45, "Assistant": 0.25}
    interes_map = {"Hot": 1.0, "Warm": 0.55, "Cold": 0.20}

    leads["_s_status"]  = leads["statuscodename"].map(status_map).fillna(0.35)
    leads["_s_job"]     = leads["bit_jobname"].map(job_map).fillna(0.35)
    leads["_s_interes"] = leads["nivel_interes"].map(interes_map).fillna(0.35)
    leads["createdon"]  = pd.to_datetime(leads["createdon"], errors="coerce", utc=True)
    cutoff = pd.Timestamp("now", tz="UTC") - pd.Timedelta(days=365)
    leads["_recent"]    = (leads["createdon"] >= cutoff).astype(float)

    leads["dim5"] = (leads["_s_status"]  * 0.45
                   + leads["_s_job"]     * 0.30
                   + leads["_s_interes"] * 0.15
                   + leads["_recent"]    * 0.10).clip(0, 1)

    # ── Datos históricos Zukán ────────────────────────────────────────────────
    df_hist = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df_hist["sector_key"] = df_hist["Sector_3"].map(lambda x: _norm(x) if pd.notna(x) else None)
    df25 = df_hist[df_hist["ejercicio"] == 2025].copy()

    # ── Dimensión 1: penetración de top clientes en el sector (2025, 25%) ────
    cli_fac = (df25.groupby("cod_cliente")
                   .agg(fac=("total_base_neto","sum"),
                        sector_key=("sector_key","first"))
                   .reset_index())
    umbral_top = cli_fac["fac"].quantile(0.75)
    cli_fac["es_top"] = (cli_fac["fac"] >= umbral_top).astype(int)
    penetracion = (cli_fac.groupby("sector_key")
                          .agg(n_top=("es_top","sum"), n_total=("cod_cliente","count"))
                          .reset_index())
    penetracion["pct_top"] = penetracion["n_top"] / penetracion["n_total"]
    penetracion["dim1"] = penetracion["pct_top"].rank(pct=True)

    # ── Dimensión 2: CAGR Zukán en el sector 2022-2025 (20%) ─────────────────
    fac_anual = (df_hist[df_hist["ejercicio"].isin([2022, 2025])]
                 .groupby(["sector_key","ejercicio"])["total_base_neto"].sum()
                 .reset_index())
    fac_22 = fac_anual[fac_anual["ejercicio"]==2022].set_index("sector_key")["total_base_neto"]
    fac_25 = fac_anual[fac_anual["ejercicio"]==2025].set_index("sector_key")["total_base_neto"]
    all_sk  = fac_22.index.union(fac_25.index)
    cagr_df = pd.DataFrame({"fac_22": fac_22.reindex(all_sk).fillna(0),
                             "fac_25": fac_25.reindex(all_sk).fillna(0)}, index=all_sk)
    def _cagr(r):
        if r["fac_22"] > 0 and r["fac_25"] > 0:
            return (r["fac_25"] / r["fac_22"]) ** (1/3) - 1
        elif r["fac_22"] == 0 and r["fac_25"] > 0:
            return np.nan   # sector nuevo → asignar máximo tras el rank
        return -1.0         # sector desaparecido → penalización máxima
    cagr_df["cagr"] = cagr_df.apply(_cagr, axis=1)
    cagr_df["cagr"] = cagr_df["cagr"].fillna(cagr_df["cagr"].max())
    cagr_df["dim2"] = cagr_df["cagr"].rank(pct=True)
    cagr_df = cagr_df.reset_index().rename(columns={"index": "sector_key"})

    # ── Dimensión 3: TAM Circana (15%) ────────────────────────────────────────
    try:
        xls = pd.read_excel("data/Sectores Circana.xlsx", header=None)
        xls.columns = xls.iloc[3]
        xls = xls.iloc[4:].reset_index(drop=True)
        xls.columns = ["DEPARTMENT","SECCION","FAMILIA","CATEGORIA","SEGMENTO","SUBSEGMENTO",
                       "TAM_YA_EUR_2025","TAM_EUR_2026","TAM_YA_VOL_2025","TAM_VOL_2026","ZUKAN"]
        xls = xls[xls["ZUKAN"] != "sin codificar"].copy()
        xls["TAM_YA_EUR_2025"] = pd.to_numeric(xls["TAM_YA_EUR_2025"], errors="coerce").fillna(0)
        xls["sector_key"] = xls["ZUKAN"].map(_norm)
        tam_s = xls.groupby("sector_key")["TAM_YA_EUR_2025"].sum().reset_index()
        tam_s.columns = ["sector_key", "tam_eur"]
        tam_s["dim3"] = tam_s["tam_eur"].rank(pct=True)
    except Exception:
        tam_s = pd.DataFrame(columns=["sector_key","tam_eur","dim3"])

    # ── Dimensión 4: ticket medio del sector (20%) ────────────────────────────
    por_sector = (df25.groupby("sector_key")
                     .agg(fac_total=("total_base_neto","sum"),
                          n_clientes=("cod_cliente","nunique"))
                     .reset_index())
    por_sector["ticket_medio"] = por_sector["fac_total"] / por_sector["n_clientes"]
    por_sector["dim4"] = por_sector["ticket_medio"].rank(pct=True)

    # ── Unir dimensiones al lead ──────────────────────────────────────────────
    leads["sector_key"] = leads["bit_sectorprincipalname"].map(_norm_sector)

    leads = (leads
             .merge(penetracion[["sector_key","dim1","pct_top"]], on="sector_key", how="left")
             .merge(cagr_df[["sector_key","dim2","cagr"]], on="sector_key", how="left")
             .merge(tam_s[["sector_key","dim3"]], on="sector_key", how="left")
             .merge(por_sector[["sector_key","dim4","ticket_medio","n_clientes"]],
                    on="sector_key", how="left"))

    for d in ["dim1","dim2","dim3","dim4"]:
        leads[d] = leads[d].fillna(0.10)

    # ── Score final 0-100 ─────────────────────────────────────────────────────
    # dim1 penetración top clientes 25% · dim2 CAGR sector 20% · dim3 TAM 15%
    # dim4 ticket medio 20% · dim5 señal comercial 20%
    leads["score_raw"] = (leads["dim1"] * 0.25
                        + leads["dim2"] * 0.20
                        + leads["dim3"] * 0.15
                        + leads["dim4"] * 0.20
                        + leads["dim5"] * 0.20)
    leads["score"] = (leads["score_raw"] * 100).round(1)

    q80 = leads["score"].quantile(0.80)
    q50 = leads["score"].quantile(0.50)
    leads["indice"] = pd.cut(leads["score"],
                             bins=[0, q50, q80, 101],
                             labels=["Índice 3", "Índice 2", "Índice 1"],
                             include_lowest=True)

    # ── Potencial estimado: azules (referencia) + verdes (tendencias catálogo) ──
    bench_ref   = load_sector_benchmarks()[["sector_key", "top_products"]]
    prod_meds   = load_sector_prod_medians()
    trends_data = load_trends()
    catalog_ref = load_product_catalog()

    # Normalizar nombre de producto del catálogo para lookup
    prod_med_norm = {}  # {(sector_key, _norm(prod)): fac_med}
    for (sk, prod), vals in prod_meds.items():
        prod_med_norm[(sk, _norm(prod))] = vals["fac_med"]

    def _potencial_sector(sk):
        if not sk or (isinstance(sk, float) and np.isnan(sk)):
            return np.nan
        total = 0.0
        found = False

        # Azules: top products de referencia del sector
        tp_row = bench_ref[bench_ref["sector_key"] == sk]
        top_prods_list = tp_row["top_products"].values[0] if len(tp_row) > 0 else []
        if isinstance(top_prods_list, list):
            for prod in top_prods_list:
                v = prod_med_norm.get((sk, _norm(prod)))
                if v and not (isinstance(v, float) and np.isnan(v)):
                    total += v
                    found = True

        # Verdes: productos catálogo que encajan con tendencias del sector
        if sk in trends_data:
            for pt in trends_data[sk].get("productos_teoricos", []):
                hits = _match_catalog(pt, catalog_ref)
                for h in hits[:2]:  # máx 2 matches por tendencia para no inflar
                    v = prod_med_norm.get((sk, _norm(h)))
                    if v and not (isinstance(v, float) and np.isnan(v)):
                        total += v
                        found = True
                        break  # solo el primer match de cada tendencia

        return total if found else np.nan

    leads["potencial_est"] = leads["sector_key"].map(_potencial_sector)
    leads["potencial_est"] = leads["potencial_est"].fillna(leads["ticket_medio"]).fillna(0)
    return leads


@st.cache_data(show_spinner="Calculando benchmarks sectoriales...")
def load_sector_benchmarks():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"] == 2025].copy()
    df["sector_key"] = df["Sector_3"].map(lambda x: _norm(x) if pd.notna(x) else None)

    per_client = (df.groupby(["sector_key", "cod_cliente"])
                    .agg(fac=("total_base_neto", "sum"), tn=("Cantidad_TN", "sum"))
                    .reset_index())

    bench = (per_client.groupby("sector_key")
                       .agg(
                           fac_mediana=("fac", "median"),
                           fac_max=("fac", "max"),
                           tn_mediana=("tn", "median"),
                           tn_max=("tn", "max"),
                           n_clientes=("cod_cliente", "nunique"),
                       )
                       .reset_index())

    pvp = (df.groupby("sector_key")
             .apply(lambda g: g["total_base_neto"].sum() / g["Cantidad_TN"].sum()
                    if g["Cantidad_TN"].sum() > 0 else np.nan)
             .reset_index(name="pvp_medio"))

    prod_freq = (df.groupby(["sector_key", "descripcion_producto"])["cod_cliente"]
                   .nunique()
                   .reset_index(name="n_cli_prod"))
    top_prods = (prod_freq.sort_values("n_cli_prod", ascending=False)
                          .groupby("sector_key")
                          .head(5))
    top_prods_agg = (top_prods.groupby("sector_key")["descripcion_producto"]
                               .apply(list)
                               .reset_index(name="top_products"))

    # Potencial basado en productos: mediana de facturación por producto y cliente en el sector
    prod_cli_fac = (df.groupby(["sector_key", "descripcion_producto", "cod_cliente"])
                      ["total_base_neto"].sum().reset_index(name="fac_prod_cli"))
    prod_med = (prod_cli_fac.groupby(["sector_key", "descripcion_producto"])
                             ["fac_prod_cli"].median()
                             .reset_index(name="fac_prod_med"))
    top_prods_med = top_prods.merge(prod_med, on=["sector_key", "descripcion_producto"], how="left")
    potencial_prod = (top_prods_med.groupby("sector_key")["fac_prod_med"]
                                   .sum()
                                   .reset_index(name="potencial_prod"))

    bench = bench.merge(pvp, on="sector_key", how="left")
    bench = bench.merge(top_prods_agg, on="sector_key", how="left")
    bench = bench.merge(potencial_prod, on="sector_key", how="left")
    return bench


@st.cache_data(show_spinner=False)
def load_ingrediente_sector_pvp():
    """
    Returns:
      pvp_lookup  : dict {(ing_key_norm, sector_key_norm): {pvp_medio, tn_media, n_clientes}}
      sector_ings : dict {sector_key_norm: [list of ing_key_norm]}
    """
    df = pd.read_csv("data/ingrediente_sector_pvp.csv", encoding="utf-8")
    pvp_lookup = {}
    sector_ings: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        ik = _norm(str(r["ingrediente"]))
        sk = str(r["sector_key"])
        pvp_lookup[(ik, sk)] = {
            "pvp_medio": float(r["pvp_medio"]),
            "tn_media":  float(r["tn_media"]),
            "n_clientes": int(r["n_clientes"]),
            "ingrediente_display": str(r["ingrediente"]),
        }
        sector_ings.setdefault(sk, []).append(ik)
    return pvp_lookup, sector_ings


_MKI_SECTOR_ALIAS = {
    # MKI usa nombres completos que se fragmentan al partir por coma;
    # los valores destino son exactamente _norm(Sector_3) del Maestro
    "lacteos":                              "lacteos, yogur, postr lac",
    "yogures y postres lacteos":            "lacteos, yogur, postr lac",
    "lacteos yogures y postres lacteos":    "lacteos, yogur, postr lac",
    "lacteos, yogures y postres lacteos":   "lacteos, yogur, postr lac",
    "barritas y bebidas deportivas":        "barritas y bebidas deport",
    "especias aromas y colorantes":         "especias, aromas,colorant",
    "especias  aromas colorant":            "especias, aromas,colorant",
}


def _resolve_sector_lead(sectores_zukan_raw, pvp_sectors):
    """Devuelve el primer sector del MKI que exista en la tabla PvP."""
    if not sectores_zukan_raw or (isinstance(sectores_zukan_raw, float)
                                   and np.isnan(sectores_zukan_raw)):
        return None
    for s in str(sectores_zukan_raw).split(","):
        s = s.strip()
        if not s:
            continue
        sk = _norm(s)
        sk = _MKI_SECTOR_ALIAS.get(sk, sk)
        if sk in pvp_sectors:
            return sk
    return None


_POT_NIVELES = [
    ("A — Premium", "> 200 k€",    200_000, float("inf"), "#6A0DAD"),
    ("B — Alto",    "50 – 200 k€",  50_000,     200_000, "#2E7D32"),
    ("C — Medio",   "10 – 50 k€",   10_000,      50_000, "#E76F51"),
    ("D — Bajo",    "< 10 k€",           0,      10_000, "#888888"),
]

def _nivel_potencial(v):
    if pd.isna(v) or v <= 0:
        return "Sin datos"
    if v > 200_000:
        return "A — Premium"
    if v >  50_000:
        return "B — Alto"
    if v >  10_000:
        return "C — Medio"
    return "D — Bajo"


def _resolve_all_sectors(sectores_zukan_raw, pvp_sectors):
    """Devuelve lista de sectores únicos del MKI que existen en la tabla PvP."""
    if not sectores_zukan_raw or (isinstance(sectores_zukan_raw, float)
                                   and np.isnan(sectores_zukan_raw)):
        return []
    seen, result = set(), []
    for s in str(sectores_zukan_raw).split(","):
        s = s.strip()
        if not s:
            continue
        sk = _norm(s)
        sk = _MKI_SECTOR_ALIAS.get(sk, sk)
        if sk in pvp_sectors and sk not in seen:
            seen.add(sk)
            result.append(sk)
    return result


@st.cache_data(show_spinner="Cargando leads por similitud...")
def load_leads_cluster_data():
    """Carga leads_clusters.csv y calcula potencial_est a partir de PvP historico.
    - Usa todos los sectores MKI (no solo el primero) para potencial multi-sector.
    - sector_lead = sector con mayor potencial estimado (referencia para la tabla).
    - Usa ingredientes_zukan_detectados de leads_fuentes_externas (todos los ingredientes)."""
    df = pd.read_csv("data/leads_clusters.csv", encoding="utf-8", low_memory=False)

    # Traer ingredientes completos desde leads_fuentes_externas (join por empresa+origen)
    ext = pd.read_csv("data/leads_fuentes_externas.csv", encoding="utf-8", low_memory=False,
                      usecols=["empresa", "origen", "ingredientes_zukan_detectados"])
    ext = ext.rename(columns={
        "empresa":                      "empresa_lead",
        "ingredientes_zukan_detectados": "ingredientes_detectados_full",
    })
    df = df.merge(ext[["empresa_lead", "origen", "ingredientes_detectados_full"]],
                  on=["empresa_lead", "origen"], how="left")

    pvp_lookup, sector_ings = load_ingrediente_sector_pvp()
    pvp_sectors = set(sk for _, sk in pvp_lookup.keys())

    df["ingredientes_lead"]   = df["ingredientes_detectados_full"].fillna(df["ingredientes_lead"])
    df["sector_leads_list"]   = df["sectores_zukan"].apply(
        lambda x: _resolve_all_sectors(x, pvp_sectors)
    )

    def _potencial_best_per_ing(row):
        """Cada ingrediente se cuenta UNA sola vez: el sector que da mayor potencial.
        Así evitamos duplicar el mismo ingrediente en N sectores del mismo lead."""
        ings_raw = row.get("ingredientes_lead")
        sectors  = row.get("sector_leads_list") or []
        if not ings_raw or pd.isna(ings_raw) or not sectors:
            return 0.0, None
        sector_contrib: dict[str, float] = {sk: 0.0 for sk in sectors}
        total = 0.0
        for ing in str(ings_raw).split(","):
            ing = ing.strip()
            if not ing:
                continue
            ik = _norm(ing)
            best_pot, best_sk = 0.0, None
            for sk in sectors:
                data = pvp_lookup.get((ik, sk))
                if data and data["tn_media"] > 0 and data["pvp_medio"] > 0:
                    pot = data["tn_media"] * data["pvp_medio"]
                    if pot > best_pot:
                        best_pot, best_sk = pot, sk
            if best_sk:
                total += best_pot
                sector_contrib[best_sk] += best_pot
        lead_sk = max(sector_contrib, key=sector_contrib.get) if total > 0 else (sectors[0] if sectors else None)
        return total, lead_sk

    _res             = df.apply(_potencial_best_per_ing, axis=1)
    df["potencial_est"] = _res.apply(lambda x: x[0])
    df["sector_lead"]   = _res.apply(lambda x: x[1])
    df["sector_key"]    = df["sector_lead"].fillna("")
    return df


@st.cache_data(show_spinner=False)
def load_sector_prod_medians():
    """Devuelve dict {(sector_key, nombre_normalizado): {fac_med, tn_med}}.
    Indexado por descripcion_producto (ventas) Y descripcion_comercial (catálogo)
    para que el matching por tendencias encuentre siempre la mediana correcta."""
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2025)].copy()
    df["sector_key"] = df["Sector_3"].map(lambda x: _norm(x) if pd.notna(x) else None)
    df = df.dropna(subset=["sector_key", "descripcion_producto"])

    # Unir con catálogo para obtener descripcion_comercial canónica
    try:
        cat = pd.read_csv("data/Maestro Producto.csv", encoding="utf-8-sig",
                          sep=None, engine="python",
                          usecols=["codigo_producto", "descripcion_comercial"])
        df = df.merge(cat, left_on="cod_producto", right_on="codigo_producto", how="left")
    except Exception:
        df["descripcion_comercial"] = np.nan

    per_prod_cli = (df.groupby(["sector_key", "descripcion_producto", "cod_cliente"])
                      .agg(fac=("total_base_neto", "sum"), tn=("Cantidad_TN", "sum"),
                           dc=("descripcion_comercial", "first"))
                      .reset_index())
    med = (per_prod_cli.groupby(["sector_key", "descripcion_producto"])
                       .agg(fac_med=("fac", "median"), tn_med=("tn", "median"),
                            dc=("dc", "first"))
                       .reset_index())
    result = {}
    for _, row in med.iterrows():
        val = {"fac_med": row["fac_med"], "tn_med": row["tn_med"]}
        # Indexar por descripcion_producto (nombre en ventas)
        result[(row["sector_key"], _norm(row["descripcion_producto"]))] = val
        # Indexar también por descripcion_comercial (nombre en catálogo) si difiere
        if pd.notna(row["dc"]):
            dc_key = (row["sector_key"], _norm(str(row["dc"])))
            if dc_key not in result:
                result[dc_key] = val
    return result


@st.cache_data
def load_trends():
    try:
        with open("data/tendencias_sector.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_data(show_spinner="Cargando catálogo de productos...")
def load_product_catalog():
    """Carga Maestro Producto con todos los campos de texto útiles para matching."""
    try:
        df = pd.read_csv("data/Maestro Producto.csv", encoding="utf-8-sig",
                         sep=None, engine="python")
        activos = df[df["es_producto_activo"] == True].dropna(subset=["descripcion_comercial"])
        result = []
        for _, r in activos.iterrows():
            result.append({
                "nombre":       str(r["descripcion_comercial"]),
                "norm":         _norm(str(r["descripcion_comercial"])),
                "tecnica_norm": _norm(str(r["descripcion_tecnica"])) if pd.notna(r.get("descripcion_tecnica")) else "",
                "desc_norm":    _norm(str(r["descripcion"])) if pd.notna(r.get("descripcion")) else "",
            })
        return result
    except FileNotFoundError:
        return []


_STOP_MATCH = {"para", "natural", "naturales", "funcional", "alto", "baja", "bajo",
               "rico", "rica", "base", "libre", "gran", "gran", "tipo", "puro", "pura"}

def _match_catalog(producto_teorico, catalog):
    """Busca en catálogo por descripcion_tecnica (tier 1), descripcion_comercial (tier 2)
    y descripcion (tier 3). Requiere al menos min(len(kws), 2) keywords coincidentes
    en el mismo campo para evitar falsos positivos por términos genéricos."""
    if not catalog:
        return []
    texto = re.sub(r"[(),]", " ", producto_teorico)
    kws = [w for w in _norm(texto).split()
           if len(w) > 3 and w not in _STOP_MATCH]
    if not kws:
        return []
    # Con 1 keyword basta 1; con 2+ se exigen al menos 2 (evita match por "jarabe" solo)
    min_match = min(len(kws), 2)
    tier1, tier2, tier3 = [], [], []
    seen = set()
    for entry in catalog:
        nombre = entry["nombre"]
        if nombre in seen:
            continue
        t1 = sum(1 for kw in kws if kw in entry["tecnica_norm"]) if entry["tecnica_norm"] else 0
        t2 = sum(1 for kw in kws if kw in entry["norm"])
        t3 = sum(1 for kw in kws if kw in entry["desc_norm"]) if entry["desc_norm"] else 0
        if t1 >= min_match:
            tier1.append(nombre); seen.add(nombre)
        elif t2 >= min_match:
            tier2.append(nombre); seen.add(nombre)
        elif t3 >= min_match:
            tier3.append(nombre); seen.add(nombre)
    return tier1 + tier2 + tier3


@st.cache_data(show_spinner="Cargando TAM Circana...")
def load_circana_data():
    xls = pd.read_excel("data/Sectores Circana.xlsx", header=None)
    xls.columns = xls.iloc[3]
    xls = xls.iloc[4:].reset_index(drop=True)
    xls.columns = ["DEPARTMENT","SECCION","FAMILIA","CATEGORIA","SEGMENTO","SUBSEGMENTO",
                   "TAM_YA_EUR_2025","TAM_EUR_2026","TAM_YA_VOL_2025","TAM_VOL_2026","ZUKAN"]
    xls = xls[xls["ZUKAN"] != "sin codificar"].copy()
    for c in ["TAM_YA_EUR_2025","TAM_EUR_2026","TAM_YA_VOL_2025","TAM_VOL_2026"]:
        xls[c] = pd.to_numeric(xls[c], errors="coerce").fillna(0)
    xls["sector_key"] = xls["ZUKAN"].map(_norm)
    tam = (xls.groupby(["ZUKAN","sector_key"])
              .agg(tam_eur_2025=("TAM_YA_EUR_2025","sum"),
                   tam_vol_2025=("TAM_YA_VOL_2025","sum"))
              .reset_index())

    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df["sector_key"] = df["Sector_3"].map(lambda x: _norm(x) if pd.notna(x) else None)

    def _zukan_yr(year):
        d = df[df["ejercicio"] == year].copy()
        return (d.groupby("sector_key")
                 .agg(fac=("total_base_neto","sum"),
                      tn=("Cantidad_TN","sum"),
                      n_cli=("cod_cliente","nunique"))
                 .reset_index())

    z25 = _zukan_yr(2025).rename(columns={"fac":"fac_2025","tn":"tn_2025","n_cli":"n_cli_2025"})
    z26 = _zukan_yr(2026).rename(columns={"fac":"fac_2026","tn":"tn_2026","n_cli":"n_cli_2026"})

    merged = tam.merge(z25, on="sector_key", how="left").merge(z26, on="sector_key", how="left")
    for c in ["fac_2025","tn_2025","n_cli_2025","fac_2026","tn_2026","n_cli_2026"]:
        merged[c] = merged[c].fillna(0)
    for c in ["n_cli_2025","n_cli_2026"]:
        merged[c] = merged[c].astype(int)
    # backward-compat aliases used in the rest of the code
    merged["fac_zukan"]  = merged["fac_2025"]
    merged["tn_zukan"]   = merged["tn_2025"]
    merged["n_clientes"] = merged["n_cli_2025"]
    merged["tam_tn_2025"] = merged["tam_vol_2025"] / 1000
    merged["pct_eur"] = (merged["fac_2025"] / merged["tam_eur_2025"] * 100).clip(0, 100)
    merged["pct_tn"]  = (merged["tn_2025"]  / merged["tam_tn_2025"]  * 100).clip(0, 100)
    merged = merged.sort_values("tam_eur_2025", ascending=False).reset_index(drop=True)
    return merged

@st.cache_data(show_spinner="Calculando pedidos...")
def load_orders_data():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    # E6: nivel albarán (unidad de entrega)
    alb = (df.groupby(["cod_cliente","ejercicio","id_albaran_vta"], as_index=False)
             .agg(
                 fac_alb=("total_base_neto","sum"),
                 tn_alb=("Cantidad_TN","sum"),
                 n_prods_alb=("cod_producto","nunique"),
             ))
    # E7: nivel pedido (distribución) y total productos por año
    ped = (df.groupby(["cod_cliente","ejercicio","id_pedido_vta"], as_index=False)
             .agg(n_prods_ped=("cod_producto","nunique")))
    prods_yr = (df.groupby(["cod_cliente","ejercicio"])
                  .agg(n_prods=("cod_producto","nunique"))
                  .reset_index())
    return alb, ped, prods_yr

def compute_e6_e7(alb, ped, prods_yr, cod_cliente, ano_ant, ano_act):
    alb_ant = alb[(alb["cod_cliente"]==cod_cliente) & (alb["ejercicio"]==ano_ant)]
    alb_act = alb[(alb["cod_cliente"]==cod_cliente) & (alb["ejercicio"]==ano_act)]
    ped_ant = ped[(ped["cod_cliente"]==cod_cliente) & (ped["ejercicio"]==ano_ant)]
    ped_act = ped[(ped["cod_cliente"]==cod_cliente) & (ped["ejercicio"]==ano_act)]
    py_ant  = prods_yr[(prods_yr["cod_cliente"]==cod_cliente) & (prods_yr["ejercicio"]==ano_ant)]
    py_act  = prods_yr[(prods_yr["cod_cliente"]==cod_cliente) & (prods_yr["ejercicio"]==ano_act)]

    def _safe(series, fn):
        return fn(series) if len(series) > 0 else np.nan

    e6 = {
        "n_ped_ant":  len(ped_ant),
        "n_ped_act":  len(ped_act),
        "n_ant":      len(alb_ant),
        "n_act":      len(alb_act),
        "ticket_ant": _safe(alb_ant["fac_alb"], lambda s: s.mean()),
        "ticket_act": _safe(alb_act["fac_alb"], lambda s: s.mean()),
        "tn_ant":     _safe(alb_ant["tn_alb"],  lambda s: s.mean()),
        "tn_act":     _safe(alb_act["tn_alb"],  lambda s: s.mean()),
    }

    n_prods_ant = int(py_ant["n_prods"].iloc[0]) if len(py_ant) > 0 else np.nan
    n_prods_act = int(py_act["n_prods"].iloc[0]) if len(py_act) > 0 else np.nan

    e7_avg_ant = _safe(ped_ant["n_prods_ped"], lambda s: s.mean())
    e7_avg_act = _safe(ped_act["n_prods_ped"], lambda s: s.mean())

    def _dist(series, label):
        if len(series) == 0:
            return pd.DataFrame(columns=[label, "Nº pedidos", "%"])
        vc = series.value_counts().sort_index().reset_index()
        vc.columns = [label, "Nº pedidos"]
        vc["%"] = (vc["Nº pedidos"] / vc["Nº pedidos"].sum() * 100).round(1)
        return vc

    return {
        "e6": e6,
        "n_prods_ant": n_prods_ant,
        "n_prods_act": n_prods_act,
        "e7_avg_ant": e7_avg_ant,
        "e7_avg_act": e7_avg_act,
        "dist_ant": _dist(ped_ant["n_prods_ped"], "Prods/pedido"),
        "dist_act": _dist(ped_act["n_prods_ped"], "Prods/pedido"),
    }

@st.cache_data(show_spinner="Calculando impacto potencial...")
def load_impacto_data():
    """Agrega los tres escenarios de crecimiento para la vista Impacto Zukán."""
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df["fac_M"] = df["total_base_neto"] / 1e6

    # 1. Baseline: facturación actual 2026
    fac_actual = df[df["ejercicio"] == 2026]["fac_M"].sum()

    # 2. Gap clientes actuales (2026 vs año de máximo histórico)
    fac_cur = (df[df["ejercicio"] == 2026]
               .groupby("cod_cliente", as_index=False)
               .agg(fac_act=("fac_M", "sum")))
    cli_act = set(fac_cur[fac_cur["fac_act"] > 0]["cod_cliente"])

    fac_hist_yr = (df[df["ejercicio"].between(2021, 2025)]
                   .groupby(["cod_cliente", "ejercicio"], as_index=False)
                   .agg(fac_anual=("fac_M", "sum")))
    idx_max = fac_hist_yr.groupby("cod_cliente")["fac_anual"].idxmax()
    fac_hist_max = (fac_hist_yr.loc[idx_max][["cod_cliente", "fac_anual"]]
                    .rename(columns={"fac_anual": "fac_max"})
                    .reset_index(drop=True))
    df_act = fac_cur.merge(fac_hist_max, on="cod_cliente", how="left")
    df_act["gap"] = (df_act["fac_max"] - df_act["fac_act"]).clip(lower=0)
    df_act = df_act[df_act["fac_act"] > 0]
    gap_actuales = df_act["gap"].sum()
    n_act_gap = int((df_act["gap"] > 0).sum())

    # 3. Recuperación clientes pasados (activos 2021-2025 pero sin 2026) → fac en año pico
    fac_hist_past = fac_hist_yr[~fac_hist_yr["cod_cliente"].isin(cli_act)]
    idx_max_past = fac_hist_past.groupby("cod_cliente")["fac_anual"].idxmax()
    df_past_peak = (fac_hist_past.loc[idx_max_past][["cod_cliente", "fac_anual"]]
                    .rename(columns={"fac_anual": "fac_max"})
                    .reset_index(drop=True))
    recuperacion_pasados = df_past_peak["fac_max"].sum()
    n_pasados = int(len(df_past_peak))

    # 4. Leads Cluster 1 (Alta similitud >= 75%)
    leads_df = load_leads_cluster_data()
    leads_1  = leads_df[leads_df["cluster"] == "1 - Alta"].copy()
    potencial_leads = leads_1["potencial_est"].sum() / 1e6
    n_leads_1 = int(len(leads_1))

    # Desglose por nivel de potencial estimado
    _NIV_ORDER = [n[0] for n in _POT_NIVELES]
    leads_1["_nivel"] = leads_1["potencial_est"].apply(_nivel_potencial)
    leads_nivel_df = (
        leads_1[leads_1["_nivel"] != "Sin datos"]
        .groupby("_nivel", as_index=False)
        .agg(potencial_M=("potencial_est", lambda x: x.sum() / 1e6),
             n_leads=("potencial_est", "count"))
        .set_index("_nivel").reindex(_NIV_ORDER)
        .fillna({"potencial_M": 0.0, "n_leads": 0})
        .reset_index()
    )
    leads_nivel_df["n_leads"] = leads_nivel_df["n_leads"].astype(int)

    potencial_total = fac_actual + gap_actuales + recuperacion_pasados + potencial_leads
    uplift_pct = (potencial_total / fac_actual - 1) * 100 if fac_actual > 0 else 0

    return {
        "fac_actual": fac_actual,
        "gap_actuales": gap_actuales,
        "n_act_gap": n_act_gap,
        "recuperacion_pasados": recuperacion_pasados,
        "n_pasados": n_pasados,
        "potencial_leads": potencial_leads,
        "n_leads_1": n_leads_1,
        "leads_nivel_df": leads_nivel_df,
        "potencial_total": potencial_total,
        "uplift_pct": uplift_pct,
    }


@st.cache_data(show_spinner="Calculando resumen ejecutivo...")
def load_resumen_ejecutivo():
    df = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    df = df[df["ejercicio"].between(2021, 2026)].copy()
    df["sector"] = df["Sector_3"].fillna("Sin sector").str.strip().str.title()
    df["sector_key"] = df["Sector_3"].map(lambda x: _norm(x) if pd.notna(x) else "sin sector")
    df["fac_M"] = df["total_base_neto"] / 1e6

    # Sector por cliente (más frecuente)
    cli_sector = (df.groupby("cod_cliente")
                  .agg(sector=("sector", lambda x: x.mode()[0] if x.notna().any() else "Sin sector"),
                       sector_key=("sector_key", lambda x: x.mode()[0] if x.notna().any() else "sin sector"))
                  .reset_index())

    # Facturación actual 2026
    fac_actual = float(df[df["ejercicio"] == 2026]["fac_M"].sum())

    # ── Clientes actuales: gap = peak_hist - fac_2026 ────────────────────────
    fac_hist_yr = (df[df["ejercicio"].between(2021, 2025)]
                   .groupby(["cod_cliente", "ejercicio"], as_index=False)
                   .agg(fac_anual=("fac_M", "sum")))
    idx_max = fac_hist_yr.groupby("cod_cliente")["fac_anual"].idxmax()
    fac_peak = fac_hist_yr.loc[idx_max].set_index("cod_cliente")["fac_anual"]

    fac_2026 = (df[df["ejercicio"] == 2026]
                .groupby("cod_cliente", as_index=False)
                .agg(fac_act=("fac_M", "sum")))
    fac_2026 = fac_2026.merge(cli_sector, on="cod_cliente", how="left")
    fac_2026["fac_max"] = fac_2026["cod_cliente"].map(fac_peak).fillna(0)
    fac_2026["gap"] = (fac_2026["fac_max"] - fac_2026["fac_act"]).clip(lower=0)
    fac_2026 = fac_2026[fac_2026["fac_act"] > 0]
    cli_act = set(fac_2026["cod_cliente"])

    gap_act_total = float(fac_2026["gap"].sum())
    gap_act_sec = (fac_2026.groupby("sector_key")
                   .agg(gap_act=("gap", "sum"), n_act=("cod_cliente", "nunique"))
                   .reset_index())

    # ── Clientes anteriores: peak de clientes sin ventas en 2026 ─────────────
    fac_past = fac_hist_yr[~fac_hist_yr["cod_cliente"].isin(cli_act)].copy()
    idx_past = fac_past.groupby("cod_cliente")["fac_anual"].idxmax()
    fac_past_peak = (fac_past.loc[idx_past][["cod_cliente", "fac_anual"]]
                    .rename(columns={"fac_anual": "fac_max"}).reset_index(drop=True))
    fac_past_peak = fac_past_peak.merge(cli_sector, on="cod_cliente", how="left")

    rec_total = float(fac_past_peak["fac_max"].sum())
    gap_ant_sec = (fac_past_peak.groupby("sector_key")
                   .agg(gap_ant=("fac_max", "sum"), n_ant=("cod_cliente", "nunique"))
                   .reset_index())

    # Máximos históricos = suma del mejor año individual de cada cliente (activos + anteriores)
    max_historico = float(fac_2026["fac_max"].sum()) + rec_total

    # ── Leads cluster 1 (Alta similitud) únicamente ───────────────────────────
    leads_df = load_leads_cluster_data()
    leads_df = leads_df[
        (leads_df["cluster"] == "1 - Alta") & (leads_df["potencial_est"] > 0)
    ].copy()
    leads_df["sector_key"] = leads_df["sector_key"].fillna("sin sector")
    leads_sec = (leads_df.groupby("sector_key")
                 .agg(pot_leads=("potencial_est", lambda x: x.sum() / 1e6),
                      n_leads=("potencial_est", "count"))
                 .reset_index())

    pot_leads_total = float(leads_sec["pot_leads"].sum())
    potencial_total = gap_act_total + rec_total + pot_leads_total

    # ── Tabla combinada por sector ────────────────────────────────────────────
    all_keys = sorted(
        set(gap_act_sec["sector_key"]) | set(gap_ant_sec["sector_key"]) | set(leads_sec["sector_key"])
    )
    sector_display = {
        _norm(row["sector"]): row["sector"]
        for _, row in cli_sector.iterrows()
        if pd.notna(row.get("sector"))
    }
    tabla = pd.DataFrame({"sector_key": all_keys})
    tabla = (tabla
             .merge(gap_act_sec, on="sector_key", how="left")
             .merge(gap_ant_sec, on="sector_key", how="left")
             .merge(leads_sec,   on="sector_key", how="left"))
    for c in ["gap_act", "gap_ant", "pot_leads", "n_act", "n_ant", "n_leads"]:
        tabla[c] = tabla[c].fillna(0)
    tabla["total"] = tabla["gap_act"] + tabla["gap_ant"] + tabla["pot_leads"]
    tabla["sector_label"] = tabla["sector_key"].map(
        lambda k: sector_display.get(k, k.title())
    )
    tabla = tabla.sort_values("total", ascending=False).reset_index(drop=True)

    return {
        "fac_actual":       fac_actual,
        "max_historico":    max_historico,
        "gap_act_total":    gap_act_total,
        "rec_total":        rec_total,
        "pot_leads_total":  pot_leads_total,
        "potencial_total":  potencial_total,
        "tabla":            tabla,
    }


@st.cache_data(show_spinner=False)
def load_lqfb_insights():
    xls = pd.read_excel("data/Productos_ValorAnadido_Zukan_Actualizado.xlsx")
    xls = xls[xls["Cód. Producto"].notna()].copy()
    xls["cod"] = pd.to_numeric(xls["Cód. Producto"], errors="coerce")
    xls = xls[
        xls["cod"].notna() &
        ~xls["Descripción Comercial"].astype(str).str.startswith("TOTAL")
    ].copy()
    lqfb_codes = set(xls["cod"].astype(int))

    mac = pd.read_csv("data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False)
    mac = mac[mac["ejercicio"].between(2021, 2026)].copy()
    mac["fac_M"] = mac["total_base_neto"] / 1e6
    lqfb = mac[mac["cod_producto"].isin(lqfb_codes)].copy()

    tot_yr  = mac.groupby("ejercicio")["fac_M"].sum()
    lqfb_yr = lqfb.groupby("ejercicio")["fac_M"].sum()
    mix_pct = {yr: float(lqfb_yr.get(yr, 0) / tot_yr.get(yr, 1) * 100) for yr in ANOS}

    cli_act_2026 = set(mac[(mac["ejercicio"] == 2026) & (mac["fac_M"] > 0)]["cod_cliente"])
    cli_2026_tot  = mac[mac["ejercicio"] == 2026].groupby("cod_cliente")["fac_M"].sum()
    cli_2026_tot  = cli_2026_tot[cli_2026_tot > 0]
    cli_2026_lqfb = lqfb[lqfb["ejercicio"] == 2026].groupby("cod_cliente")["fac_M"].sum()
    cli_2026_lqfb = cli_2026_lqfb.reindex(cli_2026_tot.index).fillna(0)
    pct_lqfb_2026 = cli_2026_lqfb / cli_2026_tot * 100
    n_cero_lqfb  = int((pct_lqfb_2026 == 0).sum())
    fac_cero_lqfb = float(cli_2026_tot[pct_lqfb_2026 == 0].sum())
    n_bajo_lqfb  = int(((pct_lqfb_2026 > 0) & (pct_lqfb_2026 < 20)).sum())

    lqfb_hist    = lqfb[lqfb["ejercicio"].between(2021, 2025)]
    lqfb_hist_yr = lqfb_hist.groupby(["cod_cliente", "ejercicio"])["fac_M"].sum().reset_index()
    lqfb_hist_act = lqfb_hist_yr[lqfb_hist_yr["cod_cliente"].isin(cli_act_2026)]
    gap_lqfb_act_total = 0.0
    n_gap_lqfb_act = 0
    if len(lqfb_hist_act) > 0:
        idx = lqfb_hist_act.groupby("cod_cliente")["fac_M"].idxmax()
        peak = lqfb_hist_act.loc[idx].set_index("cod_cliente")["fac_M"]
        lqfb_2026_cli = lqfb[lqfb["ejercicio"] == 2026].groupby("cod_cliente")["fac_M"].sum()
        lqfb_2026_cli = lqfb_2026_cli.reindex(peak.index).fillna(0)
        gap = (peak - lqfb_2026_cli).clip(lower=0)
        gap_lqfb_act_total = float(gap.sum())
        n_gap_lqfb_act = int((gap > 0).sum())

    lqfb_hist_past = lqfb_hist_yr[~lqfb_hist_yr["cod_cliente"].isin(cli_act_2026)]
    rec_lqfb_total = 0.0
    n_rec_lqfb = 0
    if len(lqfb_hist_past) > 0:
        idx2 = lqfb_hist_past.groupby("cod_cliente")["fac_M"].idxmax()
        peak2 = lqfb_hist_past.loc[idx2].set_index("cod_cliente")["fac_M"]
        rec_lqfb_total = float(peak2.sum())
        n_rec_lqfb = int(len(peak2))

    return {
        "mix_pct":             mix_pct,
        "n_cero_lqfb":         n_cero_lqfb,
        "fac_cero_lqfb":       fac_cero_lqfb,
        "n_bajo_lqfb":         n_bajo_lqfb,
        "gap_lqfb_act_total":  gap_lqfb_act_total,
        "n_gap_lqfb_act":      n_gap_lqfb_act,
        "rec_lqfb_total":      rec_lqfb_total,
        "n_rec_lqfb":          n_rec_lqfb,
    }


# ── Helpers donut (módulo) ────────────────────────────────────────────────────
_PIE_COLORS = [
    "#0082CA","#00AD68","#CC003D","#F4A261","#2A9D8F","#E9C46A",
    "#264653","#A8DADC","#E76F51","#457B9D","#1D3557","#8ECAE6",
    "#95D5B2","#74C69D","#52B788","#B7E4C7","#FFDDD2","#FFBA08",
    "#3A86FF","#FB5607","#FF006E","#8338EC","#06D6A0","#118AB2","#073B4C",
]

def _split_otros(labels, values, threshold=5.0):
    total = sum(v for v in values if v > 0)
    ml, mv, mc, ol, ov = [], [], [], [], []
    for i, (l, v) in enumerate(zip(labels, values)):
        if v <= 0:
            continue
        pct = v / total * 100 if total > 0 else 0
        color = _PIE_COLORS[i % len(_PIE_COLORS)]
        if pct < threshold:
            ol.append(l); ov.append(v)
        else:
            ml.append(l); mv.append(v); mc.append(color)
    if ol:
        ml.append("Otros"); mv.append(sum(ov)); mc.append("#CCCCCC")
    return ml, mv, mc, ol, ov

def _donut_fig(labels, values, colors, title, unit, height=380):
    fmt     = "M€" if unit == "€" else unit
    hover_v = "%{customdata:.3f} " + fmt if unit == "€" else "%{value:,.0f} " + unit
    custom  = [v / 1e6 for v in values] if unit == "€" else values
    fig = go.Figure(go.Pie(
        labels=labels, values=values, customdata=custom,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="white", width=1.5)),
        textinfo="percent", textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>" + hover_v + "<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=ZUKAN_BLACK), x=0.5),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(font=dict(size=9), orientation="v"),
        paper_bgcolor="white", height=height,
    )
    return fig


def _yellow_block(title, content_html):
    return (
        f'<div style="background:#FFFDE7;border-left:4px solid #F9A825;border-radius:8px;'
        f'padding:14px 18px;margin:14px 0;">'
        f'<div style="font-size:10px;color:#F57F17;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.6px;margin-bottom:7px;">&#9889; {title}</div>'
        f'{content_html}</div>'
    )


fac = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    logo = Path("data/logo_zukan.png")
    if logo.exists():
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.image(str(logo), use_container_width=True)
    st.markdown("---")
    if st.button("Inicio"):
        st.session_state["pagina"] = "home"
    if st.button("Detalle evolutivo general"):
        st.session_state["pagina"] = "evolutivo"
    if st.button("Detalle variacion"):
        st.session_state["pagina"] = "variacion"
    if st.button("Activación Comercial"):
        st.session_state["pagina"] = "activacion"
    if st.button("Impacto Zukán"):
        st.session_state["pagina"] = "impacto"
    st.markdown("---")
    st.caption("Datos: Zukan ERP 2021-2026")

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["pagina"] == "home":
    st.markdown("<br>", unsafe_allow_html=True)

    logo = Path("data/logo_zukan.png")
    if logo.exists():
        c1, c2, c3 = st.columns([2,1,2])
        with c2:
            st.image(str(logo), use_container_width=True)

    st.markdown(f"""
    <h1 style='color:{ZUKAN_BLACK}; text-align:center; margin-top:0.5rem; margin-bottom:0.2rem;'>
        Planificacion Inteligencia Comercial
    </h1>
    <p style='text-align:center; color:#666; font-size:1rem; margin-bottom:2.5rem;'>
        Selecciona una vista para comenzar
    </p>
    """, unsafe_allow_html=True)

    col1, col_gap, col2, col_gap2, col3 = st.columns([2, 0.3, 2, 0.3, 2])

    with col1:
        st.markdown(f"""
        <div style='border:2px solid {ZUKAN_BLUE}; border-radius:16px; padding:2rem;
                    text-align:center; background:white; margin-bottom:0.5rem;'>
          <div style='font-size:2.5rem;'>📊</div>
          <div style='font-size:1.2rem; font-weight:700; color:{ZUKAN_BLACK}; margin:0.5rem 0;'>
              Detalle evolutivo general
          </div>
          <div style='color:#666; font-size:0.9rem;'>
              Facturacion por cliente de 2021 a 2026<br>con variacion anual en porcentaje
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Abrir detalle evolutivo", key="btn_ev"):
            st.session_state["pagina"] = "evolutivo"
            st.rerun()

    with col2:
        st.markdown(f"""
        <div style='border:2px solid {ZUKAN_BLUE}; border-radius:16px; padding:2rem;
                    text-align:center; background:white; margin-bottom:0.5rem;'>
          <div style='font-size:2.5rem;'>🔍</div>
          <div style='font-size:1.2rem; font-weight:700; color:{ZUKAN_BLACK}; margin:0.5rem 0;'>
              Detalle variacion
          </div>
          <div style='color:#666; font-size:0.9rem;'>
              Analisis de variacion por cliente:<br>volumen, precio, productos y pedidos
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Abrir detalle variacion", key="btn_var"):
            st.session_state["pagina"] = "variacion"
            st.rerun()

    with col3:
        st.markdown(f"""
        <div style='border:2px solid {ZUKAN_BLUE}; border-radius:16px; padding:2rem;
                    text-align:center; background:white; margin-bottom:0.5rem;'>
          <div style='font-size:2.5rem;'>🎯</div>
          <div style='font-size:1.2rem; font-weight:700; color:{ZUKAN_BLACK}; margin:0.5rem 0;'>
              Activación Comercial
          </div>
          <div style='color:#666; font-size:0.9rem;'>
              Potencial por cliente: actuales,<br>pasados y no clientes
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Abrir activación comercial", key="btn_act"):
            st.session_state["pagina"] = "activacion"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c_imp_l, c_imp_m, c_imp_r = st.columns([1, 2, 1])
    with c_imp_m:
        st.markdown(f"""
        <div style='border:2px solid {ZUKAN_GREEN}; border-radius:16px; padding:1.6rem 2rem;
                    text-align:center; background:white; margin-bottom:0.5rem;'>
          <div style='font-size:1.1rem; font-weight:700; color:{ZUKAN_GREEN}; margin-bottom:0.3rem;
                      text-transform:uppercase; letter-spacing:0.06em;'>
              Impacto Zukán
          </div>
          <div style='color:#555; font-size:0.88rem;'>
              Vista ejecutiva del potencial total si se ejecutan todas las oportunidades:<br>
              recuperar clientes actuales, reactivar pasados y convertir leads prioritarios
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Abrir Impacto Zukán", key="btn_impacto"):
            st.session_state["pagina"] = "impacto"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA EVOLUTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["pagina"] == "evolutivo":

    if st.button("Volver al inicio"):
        st.session_state["pagina"] = "home"
        st.rerun()

    st.markdown(f"<h2 style='color:{ZUKAN_BLACK}'>Detalle evolutivo general</h2>", unsafe_allow_html=True)

    _tab_evo, _tab_diag = st.tabs(["📊  Análisis evolutivo", "🔍  Diagnóstico estratégico"])

    with _tab_evo:
        pivot = fac.pivot_table(index=["cod_cliente","nombre_comercial"],
                                columns="ejercicio", values="fac_M", aggfunc="sum").reset_index()
        pivot.columns.name = None

        for i in range(1, len(ANOS)):
            ant, act = ANOS[i-1], ANOS[i]
            tag = f"{str(act)[-2:]}vs{str(ant)[-2:]}pct"
            ca  = pivot[ant] if ant in pivot.columns else None
            cb  = pivot[act] if act in pivot.columns else None
            pivot[tag] = (np.where(ca > 0, ((cb.fillna(0) - ca) / ca * 100).round(1), np.nan)
                          if ca is not None and cb is not None else np.nan)

        col_ord = ["cod_cliente","nombre_comercial"]
        for i, ano in enumerate(ANOS):
            if ano in pivot.columns: col_ord.append(ano)
            if i > 0:
                tag = f"{str(ano)[-2:]}vs{str(ANOS[i-1])[-2:]}pct"
                if tag in pivot.columns: col_ord.append(tag)
        pivot = pivot[[c for c in col_ord if c in pivot.columns]]

        opciones = sorted(pivot["nombre_comercial"].dropna().unique().tolist())
        fc1, fc2 = st.columns([3, 1])
        with fc1:
            seleccion = st.multiselect("Filtrar clientes", opciones,
                                       placeholder="Todos los clientes (selecciona uno o varios)...")
        with fc2:
            ordenar = st.selectbox("Ordenar por", [
                "Facturación 2025", "Facturación 2026", "Nombre A-Z", "Variación 25vs24"])

        df_show = pivot.copy()
        if seleccion:
            df_show = df_show[df_show["nombre_comercial"].isin(seleccion)]

        if ordenar == "Facturación 2025" and 2025 in df_show.columns:
            df_show = df_show.sort_values(2025, ascending=False, na_position="last")
        elif ordenar == "Facturación 2026" and 2026 in df_show.columns:
            df_show = df_show.sort_values(2026, ascending=False, na_position="last")
        elif ordenar == "Nombre A-Z":
            df_show = df_show.sort_values("nombre_comercial")
        elif ordenar == "Variación 25vs24" and "25vs24pct" in df_show.columns:
            df_show = df_show.sort_values("25vs24pct", ascending=False, na_position="last")

        tot = {"cod_cliente":"","nombre_comercial":"TOTAL"}
        for ano in ANOS:
            tot[ano] = round(df_show[ano].sum(), 3) if ano in df_show.columns else np.nan
        for i in range(1, len(ANOS)):
            ant, act = ANOS[i-1], ANOS[i]
            tag = f"{str(act)[-2:]}vs{str(ant)[-2:]}pct"
            if tag in df_show.columns:
                ta, tb = tot.get(ant) or 0, tot.get(act) or 0
                tot[tag] = round((tb - ta) / ta * 100, 1) if ta > 0 else np.nan

        df_final = pd.concat([df_show, pd.DataFrame([tot])], ignore_index=True)
        ano_cols = [c for c in df_final.columns if c in ANOS]
        pct_cols = [c for c in df_final.columns if str(c).endswith("pct")]

        ren = {"cod_cliente":"Codigo","nombre_comercial":"Cliente"}
        ren.update({a: f"{a} (M€)" for a in ano_cols})
        pct_labels = {}
        for c in pct_cols:
            parts = c.replace("pct","")
            a2, a1 = parts.split("vs")
            pct_labels[c] = f"{a2}vs{a1}%"
        ren.update(pct_labels)

        df_r   = df_final.rename(columns=ren)
        ano_r  = [f"{a} (M€)" for a in ano_cols]
        pct_r  = [pct_labels[c] for c in pct_cols]

        def color_pct(col):
            out = []
            for v in col:
                if pd.isna(v): out.append("")
                elif v > 0:    out.append(f"color:{ZUKAN_GREEN};font-weight:600")
                elif v < 0:    out.append(f"color:{ZUKAN_RED};font-weight:600")
                else:          out.append(f"color:{ZUKAN_GOLD};font-weight:600")
            return out
        def hl_total(row):
            if row["Cliente"] == "TOTAL":
                return [f"background-color:{ZUKAN_BLACK};color:white;font-weight:700"] * len(row)
            return [""] * len(row)

        fmt = {c: eu for c in ano_r}
        fmt.update({c: eupct for c in pct_r})

        styled = (df_r.style
            .apply(hl_total, axis=1)
            .apply(color_pct, subset=pct_r)
            .format(fmt, na_rep="-")
            .set_properties(**{"font-size":"12.5px","text-align":"right"}, subset=ano_r+pct_r)
            .set_properties(**{"text-align":"left","font-weight":"500"}, subset=["Cliente"])
        )

        def delta_s(act, ant):
            if ant and ant > 0: return eupct((act - ant) / ant * 100)
            return None

        totales = {ano: (df_show[ano].sum() if ano in df_show.columns else 0.0) for ano in ANOS}
        kcols   = st.columns(6)
        for i, (kc, ano) in enumerate(zip(kcols, ANOS)):
            v   = totales.get(ano, 0.0)
            ant = totales.get(ANOS[i-1]) if i > 0 else None
            kc.metric(str(ano), f"{eu(v, 2)} M€",
                      delta=(delta_s(v, ant) if i > 0 and ano != 2026 else None))

        st.markdown("---")
        st.caption(f"{len(df_show)} clientes")
        st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

        # ── Evolutivo PvP medio y Volumen por sector ──────────────────────────
        st.markdown("---")
        st.markdown(f"<h3 style='color:{ZUKAN_BLACK}'>Evolutivo por sector · PvP medio y Volumen (TN)</h3>",
                    unsafe_allow_html=True)

        sect_evol = load_sector_pvp_evol()
        _anos_se  = sorted(int(a) for a in sect_evol["ejercicio"].dropna().unique())

        pvp_piv = (sect_evol.pivot_table(index="Sector_3", columns="ejercicio",
                                         values="pvp_medio", aggfunc="mean")
                   .rename(columns=lambda a: f"pvp_{a}").reset_index())
        tn_piv  = (sect_evol.pivot_table(index="Sector_3", columns="ejercicio",
                                         values="toneladas", aggfunc="sum")
                   .rename(columns=lambda a: f"tn_{a}").reset_index())
        comb = pvp_piv.merge(tn_piv, on="Sector_3", how="outer").rename(columns={"Sector_3":"Sector"})

        col_ord = ["Sector"]
        for i, ano in enumerate(_anos_se):
            col_ord.append(f"pvp_{ano}")
            col_ord.append(f"tn_{ano}")
            if i > 0:
                ant = _anos_se[i-1]
                pvp_tag = f"pvpΔ_{str(ano)[-2:]}vs{str(ant)[-2:]}"
                tn_tag  = f"tnΔ_{str(ano)[-2:]}vs{str(ant)[-2:]}"
                comb[pvp_tag] = np.where(comb[f"pvp_{ant}"] > 0,
                    ((comb[f"pvp_{ano}"].fillna(0) - comb[f"pvp_{ant}"]) / comb[f"pvp_{ant}"] * 100).round(1), np.nan)
                comb[tn_tag]  = np.where(comb[f"tn_{ant}"] > 0,
                    ((comb[f"tn_{ano}"].fillna(0) - comb[f"tn_{ant}"]) / comb[f"tn_{ant}"] * 100).round(1), np.nan)
                col_ord += [pvp_tag, tn_tag]
        comb = comb[[c for c in col_ord if c in comb.columns]]

        last_tn = f"tn_{_anos_se[-1]}" if _anos_se else "Sector"
        if last_tn in comb.columns:
            comb = comb.sort_values(last_tn, ascending=False, na_position="last")

        pvp_cols  = [c for c in comb.columns if c.startswith("pvp_")]
        tn_cols   = [c for c in comb.columns if c.startswith("tn_")]
        pvpd_cols = [c for c in comb.columns if c.startswith("pvpΔ_")]
        tnd_cols  = [c for c in comb.columns if c.startswith("tnΔ_")]
        disp_ren  = {}
        disp_ren.update({c: c.replace("pvp_","") + " €/TN"   for c in pvp_cols})
        disp_ren.update({c: c.replace("tn_","")  + " TN"     for c in tn_cols})
        disp_ren.update({c: c.replace("pvpΔ_","PvP Δ ") + "%" for c in pvpd_cols})
        disp_ren.update({c: c.replace("tnΔ_","TN Δ ")   + "%" for c in tnd_cols})
        comb2 = comb.rename(columns=disp_ren)

        val_pvp  = [disp_ren[c] for c in pvp_cols  if c in disp_ren]
        val_tn   = [disp_ren[c] for c in tn_cols   if c in disp_ren]
        val_pctd = [disp_ren[c] for c in pvpd_cols + tnd_cols if c in disp_ren]

        fmt_se = {}
        fmt_se.update({c: lambda v: (eutn(v) + " €/TN") if pd.notna(v) else "-" for c in val_pvp})
        fmt_se.update({c: lambda v: (eutn(v) + " TN")   if pd.notna(v) else "-" for c in val_tn})
        fmt_se.update({c: eupct for c in val_pctd})

        def _cpct_se(col):
            out = []
            for v in col:
                if pd.isna(v): out.append("")
                elif v > 0:    out.append(f"color:{ZUKAN_GREEN};font-weight:600")
                elif v < 0:    out.append(f"color:{ZUKAN_RED};font-weight:600")
                else:          out.append("")
            return out

        sect_styled = (comb2.style
            .apply(_cpct_se, subset=val_pctd)
            .format(fmt_se, na_rep="-")
            .set_properties(**{"font-size":"12px","text-align":"right"}, subset=val_pvp+val_tn+val_pctd)
            .set_properties(**{"text-align":"left","font-weight":"500"}, subset=["Sector"]))
        st.dataframe(sect_styled, use_container_width=True, hide_index=True)

        def _build_evol_pivot(raw_df, dim_col, dim_label):
            _anos = sorted(int(a) for a in raw_df["ejercicio"].dropna().unique())
            pvp_p = (raw_df.pivot_table(index=dim_col, columns="ejercicio",
                                        values="pvp_medio", aggfunc="mean")
                     .rename(columns=lambda a: f"pvp_{a}").reset_index())
            tn_p  = (raw_df.pivot_table(index=dim_col, columns="ejercicio",
                                        values="toneladas", aggfunc="sum")
                     .rename(columns=lambda a: f"tn_{a}").reset_index())
            cm = pvp_p.merge(tn_p, on=dim_col, how="outer").rename(columns={dim_col: dim_label})
            ord_ = [dim_label]
            for i, ano in enumerate(_anos):
                ord_.append(f"pvp_{ano}"); ord_.append(f"tn_{ano}")
                if i > 0:
                    ant = _anos[i-1]
                    cm[f"pvpΔ_{str(ano)[-2:]}vs{str(ant)[-2:]}"] = np.where(
                        cm[f"pvp_{ant}"] > 0,
                        ((cm[f"pvp_{ano}"].fillna(0) - cm[f"pvp_{ant}"]) / cm[f"pvp_{ant}"] * 100).round(1),
                        np.nan)
                    cm[f"tnΔ_{str(ano)[-2:]}vs{str(ant)[-2:]}"] = np.where(
                        cm[f"tn_{ant}"] > 0,
                        ((cm[f"tn_{ano}"].fillna(0) - cm[f"tn_{ant}"]) / cm[f"tn_{ant}"] * 100).round(1),
                        np.nan)
                    ord_ += [f"pvpΔ_{str(ano)[-2:]}vs{str(ant)[-2:]}", f"tnΔ_{str(ano)[-2:]}vs{str(ant)[-2:]}"]
            cm = cm[[c for c in ord_ if c in cm.columns]]
            last_tn = f"tn_{_anos[-1]}" if _anos else dim_label
            if last_tn in cm.columns:
                cm = cm.sort_values(last_tn, ascending=False, na_position="last")
            pvp_c  = [c for c in cm.columns if c.startswith("pvp_")]
            tn_c   = [c for c in cm.columns if c.startswith("tn_")]
            pvpd_c = [c for c in cm.columns if c.startswith("pvpΔ_")]
            tnd_c  = [c for c in cm.columns if c.startswith("tnΔ_")]
            ren = {}
            ren.update({c: c.replace("pvp_","") + " €/TN" for c in pvp_c})
            ren.update({c: c.replace("tn_","")  + " TN"   for c in tn_c})
            ren.update({c: "PvP Δ " + c.replace("pvpΔ_","") + "%" for c in pvpd_c})
            ren.update({c: "TN Δ "  + c.replace("tnΔ_","")  + "%" for c in tnd_c})
            cm2 = cm.rename(columns=ren)
            val_pv = [ren[c] for c in pvp_c if c in ren]
            val_tn = [ren[c] for c in tn_c  if c in ren]
            val_dt = [ren[c] for c in pvpd_c + tnd_c if c in ren]
            fmt_   = {}
            fmt_.update({c: lambda v: (eutn(v) + " €/TN") if pd.notna(v) else "-" for c in val_pv})
            fmt_.update({c: lambda v: (eutn(v) + " TN")   if pd.notna(v) else "-" for c in val_tn})
            fmt_.update({c: eupct for c in val_dt})
            styled_ = (cm2.style
                .apply(_cpct_se, subset=val_dt)
                .format(fmt_, na_rep="-")
                .set_properties(**{"font-size":"12px","text-align":"right"}, subset=val_pv+val_tn+val_dt)
                .set_properties(**{"text-align":"left","font-weight":"500"}, subset=[dim_label]))
            return styled_, len(cm2)

        st.markdown("---")
        st.markdown(f"<h3 style='color:{ZUKAN_BLACK}'>Evolutivo por familia de producto · PvP medio y Volumen (TN)</h3>",
                    unsafe_allow_html=True)
        fam_evol = load_familia_pvp_evol()
        _fam_styled, _fam_n = _build_evol_pivot(fam_evol, "familia", "Familia")
        st.caption(f"{_fam_n} familias")
        st.dataframe(_fam_styled, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"<h3 style='color:{ZUKAN_BLACK}'>Evolutivo por zona geográfica · PvP medio y Volumen (TN)</h3>",
                    unsafe_allow_html=True)
        _zona_tab1, _zona_tab2 = st.tabs(["🌍  Por país", "📍  Por provincia (España)"])
        with _zona_tab1:
            _pais_evol = load_pais_pvp_evol()
            _pais_styled, _pais_n = _build_evol_pivot(_pais_evol, "pais", "País")
            st.caption(f"{_pais_n} países")
            st.dataframe(_pais_styled, use_container_width=True, hide_index=True)
        with _zona_tab2:
            _prov_evol = load_provincia_pvp_evol()
            _prov_styled, _prov_n = _build_evol_pivot(_prov_evol, "provincia", "Provincia")
            st.caption(f"{_prov_n} provincias")
            st.dataframe(_prov_styled, use_container_width=True, hide_index=True)

    # ── TAB DIAGNÓSTICO ───────────────────────────────────────────────────────
    with _tab_diag:
        _dg = load_diagnostico_data()
        _anos_d = [y for y in ANOS if y in _dg["tot_yr"]]

        _diag_counter = [0]
        def _diag_block(num, titulo, texto):
            if num is not None:
                _diag_counter[0] = num
            else:
                _diag_counter[0] += 1
            _label = f"PUNTO {_diag_counter[0]}"
            st.markdown(
                f'<div style="border-left:4px solid {ZUKAN_BLUE};background:#F0F4FF;'
                f'border-radius:8px;padding:14px 18px;margin:22px 0 6px 0;">'
                f'<div style="font-size:9px;color:{ZUKAN_BLUE};font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.7px;margin-bottom:5px;">{_label}</div>'
                f'<div style="font-size:14px;font-weight:700;color:{ZUKAN_BLACK};margin-bottom:6px;">{titulo}</div>'
                f'<div style="font-size:12px;color:#444;line-height:1.65;">{texto}</div>'
                f'</div>', unsafe_allow_html=True)

        # ── P1: Trayectoria ───────────────────────────────────────────────────
        _diag_block(1, "Crecimiento real, base de clientes en contracción permanente",
            "La facturación creció de <b>95M€ a 178M€</b> entre 2021 y 2024 (+87%). "
            "Sin embargo, la base de clientes activos cayó de <b>1.218 a 680</b> en el mismo periodo. "
            "La facturación por cliente pasó de 78K€ a 262K€: el crecimiento fue concentración, no expansión.")

        _tot_v = [_dg["tot_yr"].get(y, 0) for y in _anos_d]
        _cli_v = [_dg["cli_yr"].get(y, 0) for y in _anos_d]
        _fig1 = go.Figure()
        _fig1.add_trace(go.Bar(
            x=_anos_d, y=_tot_v, name="Facturación (M€)",
            marker_color=ZUKAN_BLUE, opacity=0.85,
            text=[f"{v:.1f}" for v in _tot_v], textposition="outside",
        ))
        _fig1.add_trace(go.Scatter(
            x=_anos_d, y=_cli_v, name="Clientes activos", yaxis="y2",
            mode="lines+markers", line=dict(color=ZUKAN_RED, width=2.5), marker=dict(size=8),
        ))
        _fig1.update_layout(
            yaxis=dict(title="Facturación (M€)", showgrid=True),
            yaxis2=dict(title="Clientes activos", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.1), height=320,
            margin=dict(t=30, b=30), plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(_fig1, use_container_width=True)

        # ── P2: Concentración ─────────────────────────────────────────────────
        _diag_block(2, "Concentración creciente: el top-20 pasa del 58% al 73%",
            "En 2021, los 20 mayores clientes representaban el <b>57,6%</b> de la facturación. "
            "En 2024 y 2025 ya suponen el <b>72-73%</b>. Cuanto mayor la concentración, "
            "más catastrófico es el impacto cuando una cuenta grande cambia su comportamiento.")

        _t20_v = [_dg["top20_pct"].get(y, 0) for y in _anos_d]
        _fig2 = go.Figure()
        _fig2.add_trace(go.Scatter(
            x=_anos_d, y=_t20_v, mode="lines+markers+text",
            line=dict(color=ZUKAN_GOLD, width=2.5), marker=dict(size=8),
            text=[f"{v:.1f}%" for v in _t20_v], textposition="top center",
            fill="tozeroy", fillcolor="rgba(249,168,37,0.10)",
        ))
        _fig2.add_hline(y=60, line_dash="dash", line_color="#ccc",
                        annotation_text="Umbral 60%", annotation_position="right")
        _fig2.update_layout(
            yaxis=dict(title="% facturación top-20 clientes", range=[0, 100]),
            height=280, margin=dict(t=20, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(_fig2, use_container_width=True)

        # ── P3: Churn ─────────────────────────────────────────────────────────
        _diag_block(3, "Churn estructural: cada año se pierde más de lo que entra",
            "Sin excepción, cada año desde 2021 se han perdido más clientes de los captados. "
            "El neto acumulado 2021-2026 es de <b>-783 clientes</b>. "
            "Y los nuevos no retienen: solo el 16-28% de los captados en 2022-23 seguían activos 3 años después.")

        _ch_yrs  = [y for y in ANOS if y in _dg["churn_new"]]
        _ch_new  = [_dg["churn_new"].get(y, 0)  for y in _ch_yrs]
        _ch_lost = [-_dg["churn_lost"].get(y, 0) for y in _ch_yrs]
        _fig3 = go.Figure()
        _fig3.add_trace(go.Bar(
            x=_ch_yrs, y=_ch_new, name="Nuevos captados",
            marker_color=ZUKAN_GREEN, text=_ch_new, textposition="outside",
        ))
        _fig3.add_trace(go.Bar(
            x=_ch_yrs, y=_ch_lost, name="Clientes perdidos",
            marker_color=ZUKAN_RED, text=[abs(v) for v in _ch_lost], textposition="outside",
        ))
        _fig3.update_layout(
            barmode="group", yaxis=dict(title="Clientes"),
            height=300, margin=dict(t=20, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(_fig3, use_container_width=True)

        # ── P4: BASES ─────────────────────────────────────────────────────────
        _diag_block(4, "BASES/Mixco Sweet: el motor del crecimiento y el origen del riesgo",
            "La familia BASES pasó de <b>9M€ (2021) a 82M€ (2025)</b>: es prácticamente todo el crecimiento. "
            "El problema es que está sostenida por 4-5 cuentas grandes de alimentación y bebidas. "
            "En 2026 (ene-sep) ya ha perdido 55M€ respecto al mismo periodo de 2025.")

        _fam = _dg["fam_yr"]
        _f_bases = _fam[_fam["familia"] == "BASES"].set_index("ejercicio")["fac_M"]
        _f_azs   = _fam[_fam["familia"] == "AZUCAR SOLIDO"].set_index("ejercicio")["fac_M"]
        _f_tot   = _fam.groupby("ejercicio")["fac_M"].sum()
        _f_rest  = (_f_tot - _f_bases.reindex(_anos_d).fillna(0)
                           - _f_azs.reindex(_anos_d).fillna(0))
        _fig4 = go.Figure()
        _fig4.add_trace(go.Bar(x=_anos_d, y=_f_bases.reindex(_anos_d).fillna(0).tolist(),
                               name="BASES (Mixco Sweet)", marker_color=ZUKAN_BLUE))
        _fig4.add_trace(go.Bar(x=_anos_d, y=_f_azs.reindex(_anos_d).fillna(0).tolist(),
                               name="Azúcar Sólido", marker_color="#90A4AE"))
        _fig4.add_trace(go.Bar(x=_anos_d, y=_f_rest.reindex(_anos_d).fillna(0).tolist(),
                               name="Resto familias", marker_color=ZUKAN_GOLD))
        _fig4.update_layout(
            barmode="stack", yaxis=dict(title="Facturación (M€)"),
            height=320, margin=dict(t=20, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(_fig4, use_container_width=True)

        # ── P5 & P6: Waterfall bridge ─────────────────────────────────────────
        def _render_bridge(bridge_data, label_ant, label_act, titulo, texto, nota_pie=""):
            _diag_block(None, titulo, texto)
            comp, top_caidas, top_subidas = bridge_data

            # ── Waterfall superior: 4 componentes agregados ───────────────────
            _wf_x = [label_ant, "Bajan", "Nuevos", "Se van", "Suben", label_act]
            _wf_y = [
                comp["tot_ant"],
                comp["bajan"],
                comp["nuevos"],
                comp["se_van"],
                comp["suben"],
                comp["tot_act"],
            ]
            _wf_measure = ["absolute", "relative", "relative", "relative", "relative", "total"]
            _wf_colors  = [ZUKAN_BLUE, ZUKAN_RED, ZUKAN_GREEN, ZUKAN_RED, ZUKAN_GREEN, ZUKAN_BLUE]
            _wf_text    = [
                f"{comp['tot_ant']:.1f}",
                f"{comp['bajan']:.1f}",
                f"+{comp['nuevos']:.1f}",
                f"{comp['se_van']:.1f}",
                f"+{comp['suben']:.1f}",
                f"{comp['tot_act']:.1f}",
            ]
            _figw = go.Figure(go.Waterfall(
                x=_wf_x, y=_wf_y, measure=_wf_measure,
                text=_wf_text, textposition="outside",
                connector=dict(line=dict(color="#ddd", width=1)),
                increasing=dict(marker_color=ZUKAN_GREEN),
                decreasing=dict(marker_color=ZUKAN_RED),
                totals=dict(marker_color=ZUKAN_BLUE),
            ))
            _figw.update_layout(
                yaxis=dict(title="M€"), height=300,
                margin=dict(t=20, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(_figw, use_container_width=True)

            # ── Detalle: top caídas y subidas side-by-side ───────────────────
            _col_c, _col_s = st.columns(2)
            with _col_c:
                st.markdown(
                    f"<div style='font-size:10px;font-weight:700;color:{ZUKAN_RED};"
                    f"text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;'>"
                    f"Top caídas (clientes existentes)</div>", unsafe_allow_html=True)
                for nm, v in top_caidas.items():
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:12px;padding:3px 0;border-bottom:1px solid #f0f0f0;'>"
                        f"<span style='color:#333'>{nm[:28]}</span>"
                        f"<span style='color:{ZUKAN_RED};font-weight:600'>{v:.2f} M€</span>"
                        f"</div>", unsafe_allow_html=True)
            with _col_s:
                st.markdown(
                    f"<div style='font-size:10px;font-weight:700;color:{ZUKAN_GREEN};"
                    f"text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;'>"
                    f"Top subidas (clientes existentes)</div>", unsafe_allow_html=True)
                for nm, v in top_subidas.items():
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:12px;padding:3px 0;border-bottom:1px solid #f0f0f0;'>"
                        f"<span style='color:#333'>{nm[:28]}</span>"
                        f"<span style='color:{ZUKAN_GREEN};font-weight:600'>+{v:.2f} M€</span>"
                        f"</div>", unsafe_allow_html=True)
            if nota_pie:
                st.caption(nota_pie)

        _render_bridge(
            _dg["bridge_2425"], "2024", "2025",
            "Caída 2024 → 2025: Zumos y Frutas explica el 86%, pero el movimiento bruto es enorme",
            "La facturación bruta que <b>baja en clientes existentes</b> es de −72,3M€, "
            "parcialmente compensada por +26,8M€ en subidas y +17,2M€ en nuevos. "
            "<b>Zumos y Frutas</b> (−30,6M€) domina la caída, pero Alljuicemed, Sugar Global, "
            "Fruit Tech y otros suman otros −15M€. Sin los tres entrantes clave "
            "(Aletta, Campo Real, MAK) el año habría sido mucho peor.",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        _render_bridge(
            _dg["bridge_2526"], "2025", "2026",
            "Caída 2025 → 2026: los relevos del año anterior ahora son los que más caen",
            "Los tres clientes que habían compensado en 2025 — <b>MAK Food</b> (−14,0M€), "
            "<b>Quirante Fruits</b> (−15,3M€) y <b>Aletta Developments</b> (−12,8M€) — "
            "son ahora las principales caídas. Y lo más crítico: "
            "las subidas de clientes existentes solo suman +5,6M€ vs +26,8M€ en 2025. "
            "<b>No hay relevos.</b>",
            "Datos 2026 hasta septiembre (9 meses). La comparativa con 2025 completo sobreestima la caída."
        )

        # ── P7: LQ/FB mix ────────────────────────────────────────────────────
        _diag_block(7, "Mix LQ/FB estratégico: porcentaje estancado en el 10-12%",
            "Los 125 productos de la cartera estratégica LQ/FB representan entre el <b>10-12%</b> "
            "de la facturación durante 4 años. El 16,9% de 2026 es óptico: el azúcar sólido se ha hundido, "
            "no es que LQ/FB haya crecido. En valor absoluto, pasó de 11,8M€ (2021) a 14,7M€ (2025) — casi plano en 5 años.")

        _lq_v = [_dg["lqfb_val"].get(y, 0) for y in _anos_d]
        _lq_p = [_dg["lqfb_pct"].get(y, 0) for y in _anos_d]
        _fig7 = go.Figure()
        _fig7.add_trace(go.Bar(x=_anos_d, y=_lq_v, name="LQ/FB M€",
                               marker_color=ZUKAN_GREEN, opacity=0.8))
        _fig7.add_trace(go.Scatter(
            x=_anos_d, y=_lq_p, name="% sobre total", yaxis="y2",
            mode="lines+markers+text", line=dict(color=ZUKAN_GOLD, width=2),
            marker=dict(size=7),
            text=[f"{v:.1f}%" for v in _lq_p], textposition="top center",
        ))
        _fig7.update_layout(
            yaxis=dict(title="Facturación LQ/FB (M€)"),
            yaxis2=dict(title="% sobre total", overlaying="y", side="right",
                        showgrid=False, range=[0, 25]),
            height=300, margin=dict(t=30, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(_fig7, use_container_width=True)

        # ── P8: Nutraceutical ────────────────────────────────────────────────
        _diag_block(8, "Nutraceutical: la única señal de crecimiento real en 2026",
            "De prácticamente cero en 2021 (23K€) a <b>2,6M€ en 2026</b>, con aceleración visible. "
            "Clientes de mayor fidelidad (producto formulado, especificaciones técnicas), "
            "natural afinidad a la cartera LQ/FB. Es el vector de crecimiento más relevante para los próximos años.")

        _nt_v = [_dg["nutra_yr"].get(y, 0) for y in _anos_d]
        _fig8 = go.Figure(go.Bar(
            x=_anos_d, y=_nt_v, marker_color=ZUKAN_GREEN,
            text=[f"{v:.2f}" for v in _nt_v], textposition="outside",
        ))
        _fig8.update_layout(
            yaxis=dict(title="Facturación nutraceutical (M€)"),
            height=260, margin=dict(t=20, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(_fig8, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA VARIACION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["pagina"] == "variacion":

    if st.button("Volver al inicio"):
        st.session_state["pagina"] = "home"
        st.rerun()

    st.markdown(f"<h2 style='color:{ZUKAN_BLACK}'>Detalle variación</h2>", unsafe_allow_html=True)

    cpx = load_bridge_data()
    alb, ped, prods_yr = load_orders_data()

    # ── Filtros ───────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        transiciones = [f"{ANOS[i]} - {ANOS[i-1]}" for i in range(1, len(ANOS))]
        trans_sel    = st.selectbox("Año de estudio - Año anterior", transiciones,
                                    index=len(transiciones) - 2, key="var_trans")

    ano_act = int(trans_sel.split(" - ")[0])
    ano_ant = int(trans_sel.split(" - ")[1])

    # Resetear selección de clientes cuando cambia el año
    if st.session_state.get("_var_last_trans") != trans_sel:
        top5 = (fac[fac["ejercicio"] == ano_act]
                .sort_values("fac_M", ascending=False)["nombre_comercial"]
                .dropna().tolist()[:5])
        st.session_state["_var_last_trans"]  = trans_sel
        st.session_state["var_clientes_sel"] = top5

    with fc2:
        todos_clientes = sorted(fac["nombre_comercial"].dropna().unique().tolist())
        clientes_sel = st.multiselect(
            "Clientes", todos_clientes,
            key="var_clientes_sel",
            placeholder="Selecciona clientes...")

    if not clientes_sel:
        st.info("Selecciona al menos un cliente para ver el análisis.")
    else:
        cli_map = (cpx[["cod_cliente","nombre_comercial"]].drop_duplicates()
                   .query("nombre_comercial in @clientes_sel"))

        for _, cli_row in cli_map.iterrows():
            cod    = cli_row["cod_cliente"]
            nombre = cli_row["nombre_comercial"]
            b      = compute_bridge(cpx, cod, ano_ant, ano_act)

            if b["fac_ant"] == 0 and b["fac_act"] == 0:
                continue

            pct_str    = f"({eupct(b['pct_fac'])})" if not np.isnan(b["pct_fac"]) else ""
            delta_sign = "▲" if b["delta_fac"] >= 0 else "▼"
            header_lbl = (
                f"{nombre}  ·  {eu(b['fac_ant']/1e6, 2)} M€ → {eu(b['fac_act']/1e6, 2)} M€"
                f"  {delta_sign} {eu_s(b['delta_fac']/1e6)} M€ {pct_str}"
            )

            with st.expander(header_lbl, expanded=True):
                tab_res, tab_evo = st.tabs(["\U0001f4ca  Resumen año", "\U0001f4c8  Evolución histórica"])

                # ── TAB 1: Story card D ───────────────────────────────────────
                with tab_res:
                    # ── Resumen narrativo automático ──────────────────────────
                    _d    = b["delta_fac"]
                    _pct  = b["pct_fac"]
                    _tn_d = b["tn_act"] - b["tn_ant"]
                    _pvp_ant = b.get("pvp_ant") or 0
                    _pvp_act = b.get("pvp_act") or 0
                    _pv_d = (_pvp_act - _pvp_ant
                             if not (np.isnan(_pvp_ant) or np.isnan(_pvp_act)) else 0)
                    _n4, _n5 = len(b["e4"]), len(b["e5"])

                    _color_txt = ZUKAN_GREEN if _d >= 0 else ZUKAN_RED

                    c1, c2, c3, c4 = st.columns(4)

                    tn_delta  = b["tn_act"] - b["tn_ant"]
                    pvp_delta = (b["pvp_act"] - b["pvp_ant"]
                                 if not (np.isnan(b["pvp_ant"]) or np.isnan(b["pvp_act"]))
                                 else None)

                    tn_pct_str = f" ({eupct(tn_delta / b['tn_ant'] * 100)})" if b["tn_ant"] > 0 else ""
                    c1.metric(
                        "E1 · Volumen entregado",
                        f"{eutn(b['tn_act'])} TN",
                        delta=f"{eutn(tn_delta, signed=True)} TN{tn_pct_str}",
                    )
                    pvp_act_str = f"{eutn(b['pvp_act'])} €/TN" if b["pvp_act"] is not None and not np.isnan(b["pvp_act"]) else "-"
                    if pvp_delta is not None and b["pvp_ant"] and not np.isnan(b["pvp_ant"]) and b["pvp_ant"] > 0:
                        pvp_pct_str  = f" ({eupct(pvp_delta / b['pvp_ant'] * 100)})"
                        pvp_delta_str = f"{eutn(pvp_delta, signed=True)} €/TN{pvp_pct_str}"
                    else:
                        pvp_delta_str = f"{eutn(pvp_delta, signed=True)} €/TN" if pvp_delta is not None else None
                    c2.metric("E2 · PvP medio", pvp_act_str, delta=pvp_delta_str)
                    c3.metric("E4 · Prods ganados",  f"+{len(b['e4'])}")
                    c4.metric("E5 · Prods perdidos", f"−{len(b['e5'])}")

                    st.markdown("---")
                    st.markdown("**Detalle variación de productos (E3 · Cambio vol+pvp · E4 · Nuevo · E5 · Perdido)**")

                    # Tabla unificada E3+E4+E5
                    # pvp_medio: media aritmética de las líneas (no ponderada por volumen)
                    def _pvp_m(r, col_m, col_fb):
                        v = r.get(col_m)
                        return float(v) if pd.notna(v) else float(r.get(col_fb) or 0)

                    unified_rows = []
                    for _, r in b["e3"].sort_values("delta_fac", ascending=False).iterrows():
                        unified_rows.append({
                            "Tipo":            "E3 · Cambio",
                            "Producto":        r["descripcion"],
                            "TN ant.":         r["toneladas_ant"],
                            "TN act.":         r["toneladas_act"],
                            "ΔTN":            r["delta_tn"],
                            "PvP ant. (€/TN)": _pvp_m(r, "pvp_medio_ant", "pvp_ant"),
                            "PvP act. (€/TN)": _pvp_m(r, "pvp_medio_act", "pvp_act"),
                            "ΔPvP (€/TN)":    r["delta_pvp"],
                            "Fac. ant. (M€)":  r["facturacion_ant"] / 1e6,
                            "Fac. act. (M€)":  r["facturacion_act"] / 1e6,
                            "ΔFac (M€)":      r["delta_fac"] / 1e6,
                        })
                    for _, r in b["e4"].sort_values("facturacion_act", ascending=False).iterrows():
                        pvp_a = _pvp_m(r, "pvp_medio_act", "pvp_act")
                        unified_rows.append({
                            "Tipo":            "E4 · Nuevo",
                            "Producto":        r["descripcion"],
                            "TN ant.":         0.0,
                            "TN act.":         r["toneladas_act"],
                            "ΔTN":            r["toneladas_act"],
                            "PvP ant. (€/TN)": 0.0,
                            "PvP act. (€/TN)": pvp_a,
                            "ΔPvP (€/TN)":    pvp_a,
                            "Fac. ant. (M€)":  0.0,
                            "Fac. act. (M€)":  r["facturacion_act"] / 1e6,
                            "ΔFac (M€)":      r["facturacion_act"] / 1e6,
                        })
                    for _, r in b["e5"].sort_values("facturacion_ant", ascending=False).iterrows():
                        pvp_p = _pvp_m(r, "pvp_medio_ant", "pvp_ant")
                        unified_rows.append({
                            "Tipo":            "E5 · Perdido",
                            "Producto":        r["descripcion"],
                            "TN ant.":         r["toneladas_ant"],
                            "TN act.":         0.0,
                            "ΔTN":            -r["toneladas_ant"],
                            "PvP ant. (€/TN)": pvp_p,
                            "PvP act. (€/TN)": 0.0,
                            "ΔPvP (€/TN)":    -pvp_p,
                            "Fac. ant. (M€)":  r["facturacion_ant"] / 1e6,
                            "Fac. act. (M€)":  0.0,
                            "ΔFac (M€)":      -r["facturacion_ant"] / 1e6,
                        })

                    if unified_rows:
                        _udf = pd.DataFrame(unified_rows)
                        unified_show = pd.DataFrame({
                            "Tipo":            _udf["Tipo"],
                            "Producto":        _udf["Producto"],
                            "TN ant.":         _udf["TN ant."].map(lambda v: eutn(v)),
                            "TN act.":         _udf["TN act."].map(lambda v: eutn(v)),
                            "ΔTN":            _udf["ΔTN"].map(lambda v: eutn(v, signed=True)),
                            "PvP ant. (€/TN)": _udf["PvP ant. (€/TN)"].map(lambda v: eutn(v)),
                            "PvP act. (€/TN)": _udf["PvP act. (€/TN)"].map(lambda v: eutn(v)),
                            "ΔPvP (€/TN)":    _udf["ΔPvP (€/TN)"].map(lambda v: eutn(v, signed=True) + " €/TN"),
                            "Fac. ant. (M€)":  _udf["Fac. ant. (M€)"].map(lambda v: eu(v, 3)),
                            "Fac. act. (M€)":  _udf["Fac. act. (M€)"].map(lambda v: eu(v, 3)),
                            "ΔFac (M€)":      _udf["ΔFac (M€)"].map(lambda v: eu_s(v, 3)),
                        })
                        def _bg_unified(val):
                            if isinstance(val, str) and val.startswith("+"): return "background-color:#e8f5e9"
                            if isinstance(val, str) and val.startswith("-"): return "background-color:#fce4ec"
                            return ""
                        unified_styled = unified_show.style.map(_bg_unified, subset=["ΔTN","ΔPvP (€/TN)","ΔFac (M€)"])
                        st.dataframe(unified_styled, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Sin variaciones de producto en el período seleccionado.")

                    # ── E6 y E7 ──────────────────────────────────────────────────
                    e67 = compute_e6_e7(alb, ped, prods_yr, cod, ano_ant, ano_act)
                    e6  = e67["e6"]

                    st.markdown("---")
                    st.markdown("**E6 · Variación de pedidos**")
                    p0, p1, p2, p3 = st.columns(4)

                    n_ped_delta = e6["n_ped_act"] - e6["n_ped_ant"]
                    p0.metric(
                        "Nº pedidos",
                        f"{eutn(e6['n_ped_act'])}",
                        delta=f"{eutn(n_ped_delta, signed=True)}",
                    )

                    n_delta = e6["n_act"] - e6["n_ant"]
                    p1.metric(
                        "Nº albaranes",
                        f"{eutn(e6['n_act'])}",
                        delta=f"{eutn(n_delta, signed=True)}",
                    )

                    t_delta = (e6["ticket_act"] - e6["ticket_ant"]
                               if not (np.isnan(e6["ticket_ant"]) or np.isnan(e6["ticket_act"])) else None)
                    t_pct   = (eupct(t_delta / e6["ticket_ant"] * 100)
                               if t_delta is not None and e6["ticket_ant"] > 0 else "")
                    t_delta_str = (f"{eu_s(t_delta, 0)} € ({t_pct})" if t_delta is not None else None)
                    p2.metric(
                        "Ticket medio por albarán",
                        f"{eu(e6['ticket_act'], 0)} €" if e6["ticket_act"] and not np.isnan(e6["ticket_act"]) else "-",
                        delta=t_delta_str,
                    )

                    tn_delta_alb = (e6["tn_act"] - e6["tn_ant"]
                                    if not (np.isnan(e6["tn_ant"]) or np.isnan(e6["tn_act"])) else None)
                    tn_pct_alb   = (eupct(tn_delta_alb / e6["tn_ant"] * 100)
                                    if tn_delta_alb is not None and e6["tn_ant"] > 0 else "")
                    tn_alb_delta_str = (f"{eutn(tn_delta_alb, signed=True)} TN ({tn_pct_alb})"
                                        if tn_delta_alb is not None else None)
                    p3.metric(
                        "TN media por albarán",
                        f"{eu(e6['tn_act'], 2)} TN" if e6["tn_act"] and not np.isnan(e6["tn_act"]) else "-",
                        delta=tn_alb_delta_str,
                    )

                    st.markdown("---")
                    st.markdown("**E7 · Total productos distintos**")
                    q1, q2, q3 = st.columns([1, 1, 2])

                    _np_ant = e67["n_prods_ant"]
                    _np_act = e67["n_prods_act"]
                    _np_ok  = not (isinstance(_np_ant, float) and np.isnan(_np_ant)) and \
                              not (isinstance(_np_act, float) and np.isnan(_np_act))
                    np_delta     = (int(_np_act) - int(_np_ant)) if _np_ok else None
                    np_pct       = (eupct(np_delta / _np_ant * 100)
                                    if np_delta is not None and _np_ant > 0 else "")
                    q1.metric(
                        f"Prods distintos {ano_ant}",
                        eutn(_np_ant) if _np_ok else "-",
                    )
                    q2.metric(
                        f"Prods distintos {ano_act}",
                        eutn(_np_act) if _np_ok else "-",
                        delta=(f"{eutn(np_delta, signed=True)} ({np_pct})" if np_delta is not None else None),
                    )

                    with q3:
                        with st.expander("Ver distribución prods/pedido"):
                            d1, d2 = st.columns(2)
                            _pct_fmt = {"%": lambda v: f"{v:.1f}".replace(".", ",")}
                            with d1:
                                st.caption(f"{ano_ant} · media {eu(e67['e7_avg_ant'],1) if not (isinstance(e67['e7_avg_ant'],float) and np.isnan(e67['e7_avg_ant'])) else '-'} prods/ped")
                                if not e67["dist_ant"].empty:
                                    st.dataframe(e67["dist_ant"].style.format(_pct_fmt),
                                                 use_container_width=True, hide_index=True)
                            with d2:
                                st.caption(f"{ano_act} · media {eu(e67['e7_avg_act'],1) if not (isinstance(e67['e7_avg_act'],float) and np.isnan(e67['e7_avg_act'])) else '-'} prods/ped")
                                if not e67["dist_act"].empty:
                                    st.dataframe(e67["dist_act"].style.format(_pct_fmt),
                                                 use_container_width=True, hide_index=True)

                # ── TAB 2: Timeline A (todos los años) ──────────────────────────
                with tab_evo:
                    for i in range(len(ANOS) - 1, 0, -1):
                        a_ant, a_act = ANOS[i - 1], ANOS[i]
                        bt = compute_bridge(cpx, cod, a_ant, a_act)

                        if bt["fac_ant"] == 0 and bt["fac_act"] == 0:
                            continue

                        border_col = ZUKAN_GREEN if bt["delta_fac"] >= 0 else ZUKAN_RED
                        pct_s  = f"({eupct(bt['pct_fac'])})" if not np.isnan(bt["pct_fac"]) else ""
                        tn_d   = bt["tn_act"] - bt["tn_ant"]
                        tn_col = ZUKAN_GREEN if tn_d >= 0 else ZUKAN_RED
                        pvp_d_s = ""
                        pvp_col = ZUKAN_BLACK
                        if not (np.isnan(bt["pvp_ant"]) or np.isnan(bt["pvp_act"])):
                            pvp_d   = bt["pvp_act"] - bt["pvp_ant"]
                            pvp_d_s = f"{eutn(pvp_d, signed=True)} €/TN"
                            pvp_col = ZUKAN_GREEN if pvp_d >= 0 else ZUKAN_RED

                        st.markdown(f"""
<div style='border-left:4px solid {border_col};padding:10px 16px;
            background:white;border-radius:6px;margin-bottom:8px;'>
  <div style='font-size:11px;font-weight:700;color:#888;margin-bottom:8px;'>
    {a_ant} → {a_act}
  </div>
  <div style='display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;'>
    <div>
      <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;'>
        ΔFac</div>
      <div style='font-size:15px;font-weight:800;color:{border_col};'>
        {eu_s(bt['delta_fac']/1e6)} M€ {pct_s}</div>
    </div>
    <div>
      <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;'>
        E1 · Volumen</div>
      <div style='font-size:13px;font-weight:700;'>
        {eutn(bt['tn_ant'])} → {eutn(bt['tn_act'])} TN
        <span style='color:{tn_col}'>({eutn(tn_d, signed=True)})</span>
      </div>
    </div>
    <div>
      <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;'>
        E2 · PvP medio</div>
      <div style='font-size:13px;font-weight:700;'>
        {eutn(bt['pvp_ant']) if bt['pvp_ant'] and not np.isnan(bt['pvp_ant']) else '-'}
        →
        {eutn(bt['pvp_act']) if bt['pvp_act'] and not np.isnan(bt['pvp_act']) else '-'}
        €/TN
        <span style='color:{pvp_col}'>{pvp_d_s}</span>
      </div>
    </div>
    <div>
      <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;'>
        E3 · Amb. cambios</div>
      <div style='font-size:13px;font-weight:700;'>{len(bt['e3'])} prods</div>
    </div>
    <div>
      <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;'>
        Cartera</div>
      <div style='font-size:11px;font-weight:600;'>
        <span style='color:{ZUKAN_GREEN}'>+{len(bt['e4'])} ganados</span>
        &nbsp;
        <span style='color:{ZUKAN_RED}'>−{len(bt['e5'])} perdidos</span>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA ACTIVACION COMERCIAL
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["pagina"] == "activacion":

    if st.button("Volver al inicio"):
        st.session_state["pagina"] = "home"
        st.rerun()

    st.markdown(f"<h2 style='color:{ZUKAN_BLACK}'>Activación Comercial</h2>", unsafe_allow_html=True)

    sub = st.radio("Segmento", ["Resumen ejecutivo", "Clientes actuales", "Clientes pasados", "No clientes"],
                   horizontal=True, label_visibility="collapsed")

    st.markdown("---")

    def _three_tables(cod, ano_pico, ano_act_yr, label_act, pvp_ref, cpx_all):
        """Devuelve (html_pico, html_act, html_gap) para 3 tablas paralelas."""

        def _tbl(rows_html, header, bg, border_color, cols=("Producto","Fac.","TN","PvP")):
            ths = "".join(
                f'<th style="text-align:{"left" if i==0 else "right"};font-weight:500;'
                f'color:#aaa;padding-bottom:3px;font-size:0.68em">{c}</th>'
                for i, c in enumerate(cols)
            )
            return (
                f'<div style="background:{bg};border-radius:8px;padding:10px 12px;'
                f'border-top:3px solid {border_color};height:100%">'
                f'<div style="font-size:0.65em;color:#888;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.05em;margin-bottom:5px">{header}</div>'
                f'<table style="width:100%;font-size:0.71em;border-collapse:collapse">'
                f'<tr>{ths}</tr>{rows_html}</table></div>'
            )

        def _rows_for_year(yr):
            df_yr = (cpx_all[(cpx_all["cod_cliente"] == cod) & (cpx_all["ejercicio"] == yr)]
                     .sort_values("facturacion", ascending=False).head(8))
            if df_yr.empty:
                return "", set()
            html = ""
            prods = set()
            for __, p in df_yr.iterrows():
                d  = str(p["descripcion_producto"])
                d  = d[:32] + "…" if len(d) > 32 else d
                f  = eu(p["facturacion"] / 1e6, 2)
                t  = eutn(p["toneladas"])
                pv = eutn(p["pvp"]) if not (isinstance(p["pvp"], float) and np.isnan(p["pvp"])) else "—"
                html += (
                    f'<tr style="border-top:1px solid #ebebeb">'
                    f'<td style="padding:2px 6px 2px 0;color:#444">{d}</td>'
                    f'<td style="text-align:right;padding:2px 4px;white-space:nowrap">{f} M€</td>'
                    f'<td style="text-align:right;padding:2px 4px;white-space:nowrap">{t} TN</td>'
                    f'<td style="text-align:right;padding:2px 0;white-space:nowrap">{pv}</td>'
                    f'</tr>'
                )
                prods.add(str(p["descripcion_producto"]))
            return html, prods

        rows_pico, prods_pico = _rows_for_year(ano_pico)
        rows_act,  prods_act  = _rows_for_year(ano_act_yr)

        html_pico = _tbl(rows_pico or '<tr><td colspan="4" style="color:#ccc;font-size:0.9em">Sin datos</td></tr>',
                         f"Productos · pico {ano_pico}", "#f8f9fa", ZUKAN_BLACK)
        html_act  = _tbl(rows_act  or '<tr><td colspan="4" style="color:#ccc;font-size:0.9em">Sin datos</td></tr>',
                         f"Productos · {label_act}", "#f0f7ff", ZUKAN_BLUE)

        # Opción B: todos los productos del pico con delta TN > 0
        df_pico_all = (cpx_all[(cpx_all["cod_cliente"] == cod) & (cpx_all["ejercicio"] == ano_pico)]
                       .sort_values("toneladas", ascending=False))
        tn_act_dict = (cpx_all[(cpx_all["cod_cliente"] == cod) & (cpx_all["ejercicio"] == ano_act_yr)]
                       .set_index("descripcion_producto")["toneladas"].to_dict())

        gap_rows = ""
        for __, p in df_pico_all.iterrows():
            prod_name = str(p["descripcion_producto"])
            tn_pico   = p["toneladas"] or 0.0
            tn_now    = tn_act_dict.get(prod_name, 0.0)
            delta_tn  = tn_pico - tn_now
            if delta_tn <= 0:
                continue
            pvp_pico = p["pvp"] if not (isinstance(p["pvp"], float) and np.isnan(p["pvp"])) else None
            pvp_now  = pvp_ref.get(prod_name, pvp_pico)
            gap_eur  = (delta_tn * pvp_now / 1e6) if pvp_now else np.nan
            d_short  = prod_name[:28] + "…" if len(prod_name) > 28 else prod_name
            tn_p_str = eutn(tn_pico)
            tn_n_str = eutn(tn_now) if tn_now > 0 else "0"
            pvp_str2 = eutn(pvp_now) if pvp_now else "—"
            gap_str  = eu(gap_eur, 2) + " M€" if pvp_now and not (isinstance(gap_eur, float) and np.isnan(gap_eur)) else "—"
            row_color = ZUKAN_RED if tn_now == 0 else "#E76F51"
            gap_rows += (
                f'<tr style="border-top:1px solid #ebebeb">'
                f'<td style="padding:2px 6px 2px 0;color:#444">{d_short}</td>'
                f'<td style="text-align:right;padding:2px 4px;white-space:nowrap;color:#999">{tn_p_str}</td>'
                f'<td style="text-align:right;padding:2px 4px;white-space:nowrap;color:#999">{tn_n_str}</td>'
                f'<td style="text-align:right;padding:2px 0;white-space:nowrap;'
                f'color:{row_color};font-weight:600">{gap_str}</td>'
                f'</tr>'
            )
        if not gap_rows:
            gap_rows = '<tr><td colspan="4" style="color:#aaa;font-size:0.9em">Sin gap de productos</td></tr>'

        html_gap = _tbl(gap_rows, "Productos gap · PvP actual", "#fff5f7", ZUKAN_RED,
                        cols=("Producto", "TN pico", "TN act.", "Gap €"))
        return html_pico, html_act, html_gap

    if sub == "Resumen ejecutivo":

        res = load_resumen_ejecutivo()
        fac_a  = res["fac_actual"]
        max_yr = res["max_historico"]
        gap_a  = res["gap_act_total"]
        rec_p  = res["rec_total"]
        pot_l  = res["pot_leads_total"]
        pot_t  = res["potencial_total"]
        tabla  = res["tabla"]

        _ins_re = load_lqfb_insights()
        _tot_lqfb = _ins_re["gap_lqfb_act_total"] + _ins_re["rec_lqfb_total"]
        st.markdown(_yellow_block(
            "Gap estratégico en la cartera de alto valor (LQ/FB)",
            f"<p style='font-size:12px;color:#555;margin:0;'>"
            f"Del potencial total, <b>{eu(_tot_lqfb, 2)} M€</b> corresponden a la "
            f"<b>cartera de alto valor</b> (125 productos LQ/FB estratégicos): "
            f"<b>{eu(_ins_re['gap_lqfb_act_total'], 2)} M€</b> en "
            f"{_ins_re['n_gap_lqfb_act']} clientes activos que compraban más antes, "
            f"más <b>{eu(_ins_re['rec_lqfb_total'], 2)} M€</b> en "
            f"{_ins_re['n_rec_lqfb']} clientes anteriores. "
            f"Priorizar estos clientes alinea la recuperación con la estrategia de margen: "
            f"mayor diferenciación y mayor valor por tonelada.</p>"
        ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI cards ─────────────────────────────────────────────────────────
        def _kpi_card(label, value, sub_label, color, bg):
            return (
                f'<div style="background:{bg};border-radius:10px;padding:16px 20px;'
                f'border-top:4px solid {color};height:100%">'
                f'<div style="font-size:10px;color:#888;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.7px;margin-bottom:4px">{label}</div>'
                f'<div style="font-size:26px;font-weight:700;color:{color}">{value}</div>'
                f'<div style="font-size:11px;color:#aaa;margin-top:2px">{sub_label}</div>'
                f'</div>'
            )

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(_kpi_card(
                "Máximos históricos",
                f"{eu(max_yr, 2)} M€",
                "Suma del mejor año de cada cliente",
                ZUKAN_BLACK, "#f8f9fa"
            ), unsafe_allow_html=True)
        with k2:
            st.markdown(_kpi_card(
                "Zukán hoy (2026)",
                f"{eu(fac_a, 2)} M€",
                f"Gap vs máximos: {eu(max_yr - fac_a, 2)} M€",
                ZUKAN_BLUE, "#f0f7ff"
            ), unsafe_allow_html=True)
        with k3:
            st.markdown(_kpi_card(
                "Potencial incremental",
                f"{eu(pot_t, 2)} M€",
                f"Gap activos + recuperación anteriores + leads cluster 1",
                ZUKAN_GREEN, "#f0fff8"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Fila desglose potencial
        b1, b2, b3, b4 = st.columns(4)
        _sub_style = (
            "font-size:11px;background:#f8f9fa;border-radius:8px;padding:10px 14px;"
            "border-left:3px solid {c}"
        )
        for col, label, val, color in [
            (b1, "Facturación base 2026",       fac_a, ZUKAN_BLUE),
            (b2, "Gap clientes actuales",        gap_a, "#F4A261"),
            (b3, "Recuperación anteriores",      rec_p, "#9B59B6"),
            (b4, "Potencial leads",              pot_l, ZUKAN_GREEN),
        ]:
            with col:
                st.markdown(
                    f'<div style="{_sub_style.format(c=color)}">'
                    f'<div style="font-size:9px;color:#888;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.5px">{label}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{color};margin-top:2px">'
                    f'{eu(val, 2)} M€</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Mapa de calor ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:11px;color:#6c757d;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.7px;margin-bottom:8px'>Mapa de calor: gap por tipo de cliente × sector</div>",
            unsafe_allow_html=True,
        )

        _top_n = 15
        top_tabla = tabla.head(_top_n)
        _sectors = top_tabla["sector_label"].tolist()
        _rows = ["Clientes activos", "Clientes anteriores", "Leads"]
        _matrix = [
            top_tabla["gap_act"].tolist(),
            top_tabla["gap_ant"].tolist(),
            top_tabla["pot_leads"].tolist(),
        ]
        _text_matrix = [
            [f"{eu(v, 2)} M€" if v > 0 else "—" for v in row]
            for row in _matrix
        ]
        _hm_fig = go.Figure(data=go.Heatmap(
            z=_matrix,
            x=_sectors,
            y=_rows,
            text=_text_matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#f0f7ff"], [0.5, "#4dbae8"], [1, ZUKAN_BLUE]],
            showscale=True,
            hovertemplate="Sector: %{x}<br>Tipo: %{y}<br>Gap: %{text}<extra></extra>",
        ))
        _hm_fig.update_layout(
            height=280,
            margin=dict(l=120, r=20, t=10, b=120),
            xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=12)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_hm_fig, use_container_width=True, key="heatmap_resumen")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabla resumen por sector ───────────────────────────────────────────
        st.markdown(
            "<div style='font-size:11px;color:#6c757d;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.7px;margin-bottom:8px'>Potencial por sector</div>",
            unsafe_allow_html=True,
        )
        tbl_display = pd.DataFrame({
            "Sector":               tabla["sector_label"],
            "Gap clientes activos (M€)": tabla["gap_act"].map(lambda v: eu(v, 2) if v > 0 else "—"),
            "N activos":            tabla["n_act"].map(lambda v: str(int(v)) if v > 0 else "—"),
            "Recuper. anteriores (M€)":  tabla["gap_ant"].map(lambda v: eu(v, 2) if v > 0 else "—"),
            "N anteriores":         tabla["n_ant"].map(lambda v: str(int(v)) if v > 0 else "—"),
            "Potencial leads (M€)": tabla["pot_leads"].map(lambda v: eu(v, 2) if v > 0 else "—"),
            "N leads":              tabla["n_leads"].map(lambda v: str(int(v)) if v > 0 else "—"),
            "TOTAL GAP (M€)":       tabla["total"].map(lambda v: eu(v, 2) if v > 0 else "—"),
        })
        tot_row = pd.DataFrame([{
            "Sector": "TOTAL",
            "Gap clientes activos (M€)": eu(gap_a, 2),
            "N activos": str(int(tabla["n_act"].sum())),
            "Recuper. anteriores (M€)": eu(rec_p, 2),
            "N anteriores": str(int(tabla["n_ant"].sum())),
            "Potencial leads (M€)": eu(pot_l, 2),
            "N leads": str(int(tabla["n_leads"].sum())),
            "TOTAL GAP (M€)": eu(gap_a + rec_p + pot_l, 2),
        }])
        tbl_display = pd.concat([tbl_display, tot_row], ignore_index=True)

        def _highlight_total(row):
            if row["Sector"] == "TOTAL":
                return ["font-weight:700;background:#f0f7ff"] * len(row)
            return [""] * len(row)

        st.dataframe(
            tbl_display.style.apply(_highlight_total, axis=1),
            use_container_width=True, hide_index=True,
        )

    elif sub == "Clientes actuales":

        # ── Tabla base ────────────────────────────────────────────────────────
        fac_2026 = (fac[fac["ejercicio"] == 2026]
                    .groupby(["cod_cliente","nombre_comercial"], as_index=False)
                    .agg(fac_act=("fac_M","sum")))
        fac_hist_yr = (fac[fac["ejercicio"].between(2021, 2025)]
                       .groupby(["cod_cliente","nombre_comercial","ejercicio"], as_index=False)
                       .agg(fac_anual=("fac_M","sum")))
        idx_max  = fac_hist_yr.groupby(["cod_cliente","nombre_comercial"])["fac_anual"].idxmax()
        fac_hist = (fac_hist_yr.loc[idx_max]
                    [["cod_cliente","nombre_comercial","fac_anual","ejercicio"]]
                    .rename(columns={"fac_anual":"fac_max","ejercicio":"ano_max"})
                    .reset_index(drop=True))
        df_act = (fac_2026.merge(fac_hist, on=["cod_cliente","nombre_comercial"], how="left")
                  .sort_values("fac_act", ascending=False)
                  .reset_index(drop=True))
        df_act["gap"] = (df_act["fac_max"] - df_act["fac_act"]).clip(lower=0)
        df_act["pct"] = (df_act["fac_act"] / df_act["fac_max"] * 100).clip(0, 100)
        df_act = df_act[df_act["fac_act"] > 0].reset_index(drop=True)

        _ins3 = load_lqfb_insights()
        st.markdown(_yellow_block(
            "Clientes activos sin ningún producto de la cartera estratégica LQ/FB",
            f"<p style='font-size:12px;color:#555;margin:0 0 5px 0;'>"
            f"<b>{_ins3['n_cero_lqfb']} clientes activos</b> ({eu(_ins3['fac_cero_lqfb'], 2)} M€ en 2026) "
            f"no tienen comprado ninguno de los <b>125 productos de alto valor añadido</b> "
            f"(gama LQ/FB estratégica). Pueden estar comprando líquidos commodity, pero no los "
            f"productos de mayor margen y diferenciación. Son el target prioritario de <i>upsell</i>.</p>"
            f"<p style='font-size:12px;color:#555;margin:0;'>"
            f"Adicionalmente, <b>{_ins3['n_bajo_lqfb']} clientes</b> tienen productos LQ/FB de alto valor "
            f"por debajo del 20% de su mix — con relación ya establecida, hay recorrido de ampliación. "
            f"Gap recuperable en la cartera estratégica: "
            f"<b>{eu(_ins3['gap_lqfb_act_total'], 2)} M€</b>.</p>"
        ), unsafe_allow_html=True)

        # ── Desplegable: clientes sin cartera estratégica LQ/FB ──────────────
        with st.expander(
            f"Ver {_ins3['n_cero_lqfb']} clientes activos sin ningún producto de la cartera estratégica LQ/FB",
            expanded=False
        ):
            _xls_c = pd.read_excel("data/Productos_ValorAnadido_Zukan_Actualizado.xlsx")
            _xls_c = _xls_c[_xls_c["Cód. Producto"].notna()].copy()
            _xls_c["_cod"] = pd.to_numeric(_xls_c["Cód. Producto"], errors="coerce")
            _xls_c = _xls_c[
                _xls_c["_cod"].notna() &
                ~_xls_c["Descripción Comercial"].astype(str).str.startswith("TOTAL")
            ]
            _lqfb_set = set(_xls_c["_cod"].astype(int))

            _mac_2026 = pd.read_csv(
                "data/Maestro_Análisis_Comercial.csv", encoding="utf-8", low_memory=False
            )
            _mac_2026 = _mac_2026[_mac_2026["ejercicio"] == 2026].copy()
            _mac_2026["_fac_M"] = _mac_2026["total_base_neto"] / 1e6

            _cli_tot = (_mac_2026.groupby(["cod_cliente", "nombre_comercial"])["_fac_M"]
                        .sum().reset_index().rename(columns={"_fac_M": "fac_2026_M"}))
            _cli_tot = _cli_tot[_cli_tot["fac_2026_M"] > 0]

            _cli_lqfb = (_mac_2026[_mac_2026["cod_producto"].isin(_lqfb_set)]
                         .groupby("cod_cliente")["_fac_M"].sum().reset_index()
                         .rename(columns={"_fac_M": "fac_lqfb_M"}))

            _cli_sin = (_cli_tot.merge(_cli_lqfb, on="cod_cliente", how="left")
                        .fillna({"fac_lqfb_M": 0}))
            _cli_sin = _cli_sin[_cli_sin["fac_lqfb_M"] == 0].copy()

            _mac_sec = (_mac_2026.groupby("cod_cliente")["Sector_3"]
                        .agg(lambda x: x.mode()[0] if x.notna().any() else None)
                        .reset_index().rename(columns={"Sector_3": "sector"}))
            _cli_sin = (_cli_sin.merge(_mac_sec, on="cod_cliente", how="left")
                        .sort_values("fac_2026_M", ascending=False)
                        .reset_index(drop=True))
            _cli_sin["sector"] = _cli_sin["sector"].fillna("Sin sector")

            _df_exp = pd.DataFrame({
                "Cliente":         _cli_sin["nombre_comercial"],
                "Sector":          _cli_sin["sector"],
                "Facturación 2026 (M€)": _cli_sin["fac_2026_M"].map(lambda v: eu(v, 3)),
            })
            st.dataframe(_df_exp, use_container_width=True, hide_index=True)

        # ── Precomputar TN/PvP 2026 para todos los clientes ──────────────────
        cpx_all  = load_bridge_data()
        tn_pvp_2026 = (cpx_all[cpx_all["ejercicio"] == 2026]
                       .groupby("cod_cliente")
                       .agg(tn_act=("toneladas","sum"), fac_act_e=("facturacion","sum"))
                       .reset_index())
        tn_pvp_2026["pvp_act"] = (tn_pvp_2026["fac_act_e"]
                                  / tn_pvp_2026["tn_act"].replace(0, np.nan)).round(0)

        # ── Sector media ──────────────────────────────────────────────────────
        cli_sector, sector_avg = load_sector_data()
        sector_avg_2026 = sector_avg[sector_avg["ejercicio"] == 2026].copy()
        sector_avg_2026 = sector_avg_2026.assign(fac_media_M=sector_avg_2026["fac_media"] / 1e6)

        PASTEL_BLUE  = "#BBDEFB"
        PASTEL_EMPTY = "#F5F5F5"

        # PvP de referencia actual por producto (años más recientes disponibles)
        pvp_ref_act = (
            cpx_all[cpx_all["ejercicio"].isin([2025, 2026])]
            .groupby("descripcion_producto")
            .apply(lambda g: g["facturacion"].sum() / g["toneladas"].sum()
                   if g["toneladas"].sum() > 0 else np.nan)
            .to_dict()
        )

        for _, row in df_act.iterrows():
            cod   = row["cod_cliente"]
            cli_n = row["nombre_comercial"]
            fac_a = row["fac_act"]
            fac_m = row["fac_max"]
            ano_m = int(row["ano_max"]) if not np.isnan(row["ano_max"]) else None
            gap   = row["gap"]
            pct   = row["pct"]

            tn_row  = tn_pvp_2026[tn_pvp_2026["cod_cliente"] == cod]
            tn_2026 = tn_row["tn_act"].values[0]  if len(tn_row) > 0 else np.nan
            pvp_act = tn_row["pvp_act"].values[0] if len(tn_row) > 0 else np.nan

            sec_row       = cli_sector[cli_sector["cod_cliente"] == cod]
            sector_nombre = sec_row["Sector_3"].values[0] if len(sec_row) > 0 else None
            fac_sector_media = np.nan
            if sector_nombre and not (isinstance(sector_nombre, float) and np.isnan(sector_nombre)):
                s_row = sector_avg_2026[sector_avg_2026["Sector_3"] == sector_nombre]
                if len(s_row) > 0:
                    fac_sector_media = s_row["fac_media_M"].values[0]
            sec_label = (sector_nombre
                         if (sector_nombre and not (isinstance(sector_nombre, float)
                                                    and np.isnan(sector_nombre))) else "—")

            _chart_h = 300

            col_vela, col_info = st.columns([1, 4])

            with col_vela:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[""], y=[fac_a],
                    marker_color=PASTEL_BLUE,
                    marker_line=dict(color="#90CAF9", width=1),
                    width=0.45,
                    hovertemplate=f"2026: {eu(fac_a,2)} M€ ({eu(pct,1)}%)<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=[""], y=[max(gap, 0)],
                    marker=dict(color=PASTEL_EMPTY, line=dict(color="#ddd", width=1)),
                    width=0.45,
                    hovertemplate=f"Gap: {eu(gap,2)} M€<extra></extra>",
                ))
                # y-axis top = fac_m (o sector_media si es mayor) sin padding
                _y_max = fac_m if (fac_m and not np.isnan(fac_m)) else 1
                if not np.isnan(fac_sector_media) and fac_sector_media > _y_max:
                    _y_max = fac_sector_media * 1.05
                if fac_m and not np.isnan(fac_m):
                    fig.add_shape(type="line", x0=-0.35, x1=0.35, y0=fac_m, y1=fac_m,
                                  line=dict(color=ZUKAN_BLACK, width=1.2, dash="dot"))
                    fig.add_annotation(x=0, y=fac_m, yshift=-10,
                        text=f"<b>Máx. {ano_m}</b>: {eu(fac_m,2)} M€",
                        showarrow=False, xanchor="center",
                        font=dict(size=8, color=ZUKAN_BLACK), bgcolor="white", borderpad=1)
                if not np.isnan(fac_sector_media):
                    fig.add_shape(type="line", x0=-0.35, x1=0.35,
                                  y0=fac_sector_media, y1=fac_sector_media,
                                  line=dict(color=ZUKAN_GREEN, width=1.2, dash="dash"))
                    _sm_yshift = 7 if fac_sector_media < fac_m else -10
                    fig.add_annotation(x=0, y=fac_sector_media, yshift=_sm_yshift,
                        text=f"<b>Media sector</b>: {eu(fac_sector_media,2)} M€",
                        showarrow=False, xanchor="center",
                        font=dict(size=8, color=ZUKAN_GREEN), bgcolor="white", borderpad=1)
                fig.add_shape(type="line", x0=-0.35, x1=0.35, y0=fac_a, y1=fac_a,
                              line=dict(color=ZUKAN_BLUE, width=1.5))
                fig.add_annotation(x=0, y=fac_a, yshift=-10,
                    text=f"<b>2026</b>: {eu(fac_a,2)} M€ ({eu(pct,1)}%)",
                    showarrow=False, xanchor="center",
                    font=dict(size=8, color=ZUKAN_BLUE), bgcolor="white", borderpad=1)
                fig.update_layout(
                    barmode="stack", height=_chart_h,
                    margin=dict(l=25, r=15, t=8, b=8),
                    showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(visible=False),
                    yaxis=dict(title="M€", gridcolor="#f5f5f5", zeroline=False,
                               tickfont=dict(size=8), range=[0, _y_max]),
                    font=dict(family="sans-serif", size=9, color=ZUKAN_BLACK),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"vela_{cod}")

            with col_info:
                delta_vs_sector = fac_a - fac_sector_media if not np.isnan(fac_sector_media) else np.nan
                gap_tn = gap * 1e6 / pvp_act if (not np.isnan(pvp_act) and pvp_act > 0) else np.nan
                vs_sector_str = (
                    f"{eu_s(delta_vs_sector,2)} M€ "
                    f"({eupct(delta_vs_sector / fac_sector_media * 100)})"
                    if (not np.isnan(delta_vs_sector) and fac_sector_media > 0) else "—"
                )
                vs_sector_color = (
                    ZUKAN_GREEN if (not np.isnan(delta_vs_sector) and delta_vs_sector >= 0)
                    else ZUKAN_RED
                )
                gap_color = ZUKAN_BLUE if gap > 0 else ZUKAN_GREEN

                st.markdown(f"""
<div style="padding:4px 0 8px 0">
  <div style="font-size:0.9em;font-weight:600;color:{ZUKAN_BLACK}">
    {cli_n}
    <span style="font-size:0.78em;font-weight:400;color:{ZUKAN_BLUE};margin-left:8px">{sec_label}</span>
  </div>
  <div style="display:flex;gap:20px;margin-top:6px;flex-wrap:wrap">
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Fac. 2026</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{eu(fac_a,2)} M€</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Toneladas</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{"—" if np.isnan(tn_2026) else eutn(tn_2026)} TN</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">PvP medio</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{"—" if np.isnan(pvp_act) else eutn(pvp_act)} €/TN</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Máx. hist. ({ano_m})</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{"—" if (not fac_m or np.isnan(fac_m)) else eu(fac_m,2)} M€</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">vs. sector</span><br>
         <span style="font-size:0.85em;color:{vs_sector_color}">{vs_sector_str}</span></div>
  </div>
  <div style="margin-top:8px;background:#f0f7ff;border-left:3px solid {gap_color};
              padding:8px 12px;border-radius:0 4px 4px 0;display:inline-block;min-width:260px">
    <div style="font-size:0.65em;color:#888;text-transform:uppercase;letter-spacing:0.06em">
      Oportunidad comercial · Gap hasta máximo histórico
    </div>
    <div style="font-size:1.5em;font-weight:700;color:{gap_color};line-height:1.2">
      {"—" if np.isnan(gap_tn) else f"+{eutn(gap_tn)} TN"}
    </div>
    <div style="font-size:0.72em;color:#555;margin-top:2px">
      {eu(gap,2)} M€ al PvP actual de {"—" if np.isnan(pvp_act) else eutn(pvp_act)} €/TN
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                if ano_m:
                    t1, t2, t3 = st.columns(3)
                    h_pico, h_act, h_gap = _three_tables(
                        cod, ano_m, 2026, "2026", pvp_ref_act, cpx_all
                    )
                    t1.markdown(h_pico, unsafe_allow_html=True)
                    t2.markdown(h_act,  unsafe_allow_html=True)
                    t3.markdown(h_gap,  unsafe_allow_html=True)

            st.markdown("<hr style='margin:4px 0;border:none;border-top:1px solid #eee'>",
                        unsafe_allow_html=True)

    elif sub == "Clientes pasados":

        # Clientes con actividad 2021-2025 pero sin actividad en 2026
        fac_2026_all = (fac[fac["ejercicio"] == 2026]
                        .groupby("cod_cliente", as_index=False)
                        .agg(fac_2026=("fac_M","sum")))
        cli_activos_2026 = set(fac_2026_all[fac_2026_all["fac_2026"] > 0]["cod_cliente"])

        fac_hist_all = (fac[fac["ejercicio"].between(2021, 2025)]
                        .groupby(["cod_cliente","nombre_comercial","ejercicio"], as_index=False)
                        .agg(fac_anual=("fac_M","sum")))
        fac_hist_past = fac_hist_all[~fac_hist_all["cod_cliente"].isin(cli_activos_2026)]

        # año y fac del pico correcto (idxmax) y del último año activo
        idx_max_p   = fac_hist_past.groupby(["cod_cliente","nombre_comercial"])["fac_anual"].idxmax()
        df_past_max = (fac_hist_past.loc[idx_max_p]
                       [["cod_cliente","nombre_comercial","fac_anual","ejercicio"]]
                       .rename(columns={"fac_anual":"fac_max","ejercicio":"ano_max"})
                       .reset_index(drop=True))
        df_past_ult_yr = (fac_hist_past.groupby(["cod_cliente","nombre_comercial"])["ejercicio"]
                          .max().reset_index().rename(columns={"ejercicio":"ano_ultimo"}))
        df_past = df_past_max.merge(df_past_ult_yr, on=["cod_cliente","nombre_comercial"])

        # fac en el último año activo
        df_past_ult = (fac_hist_past
                       .merge(df_past[["cod_cliente","ano_ultimo"]], on="cod_cliente")
                       .query("ejercicio == ano_ultimo")
                       [["cod_cliente","fac_anual"]]
                       .rename(columns={"fac_anual":"fac_ultimo"})
                       .drop_duplicates("cod_cliente"))
        df_past = (df_past.merge(df_past_ult, on="cod_cliente", how="left")
                   .sort_values("fac_max", ascending=False)
                   .reset_index(drop=True))
        df_past["fac_ultimo"] = df_past["fac_ultimo"].fillna(0)

        # TN/PvP por cliente y año
        cpx_all   = load_bridge_data()
        tn_pvp_yr = (cpx_all.groupby(["cod_cliente","ejercicio"])
                     .agg(tn=("toneladas","sum"), fac_e=("facturacion","sum"))
                     .reset_index())
        tn_pvp_yr["pvp"] = (tn_pvp_yr["fac_e"] / tn_pvp_yr["tn"].replace(0, np.nan)).round(0)

        cli_sector, _ = load_sector_data()

        PASTEL_EMPTY = "#F5F5F5"

        pvp_ref_past = (
            cpx_all[cpx_all["ejercicio"].isin([2024, 2025])]
            .groupby("descripcion_producto")
            .apply(lambda g: g["facturacion"].sum() / g["toneladas"].sum()
                   if g["toneladas"].sum() > 0 else np.nan)
            .to_dict()
        )

        for _, row in df_past.iterrows():
            cod     = row["cod_cliente"]
            cli_n   = row["nombre_comercial"]
            fac_m   = row["fac_max"]
            ano_m   = int(row["ano_max"])    if not np.isnan(row["ano_max"])    else None
            ano_ult = int(row["ano_ultimo"]) if not np.isnan(row["ano_ultimo"]) else None
            fac_ult = row["fac_ultimo"]  # último año activo (referencia)

            tn_row  = tn_pvp_yr[(tn_pvp_yr["cod_cliente"] == cod) & (tn_pvp_yr["ejercicio"] == ano_ult)]
            tn_ult  = tn_row["tn"].values[0]  if len(tn_row) > 0 else np.nan
            pvp_ult = tn_row["pvp"].values[0] if len(tn_row) > 0 else np.nan
            # oportunidad: recuperar el nivel del último año activo
            rec_tn  = fac_ult * 1e6 / pvp_ult if (not np.isnan(pvp_ult) and pvp_ult > 0) else np.nan

            sec_row       = cli_sector[cli_sector["cod_cliente"] == cod]
            sector_nombre = sec_row["Sector_3"].values[0] if len(sec_row) > 0 else None
            sec_label     = (sector_nombre
                             if (sector_nombre and not (isinstance(sector_nombre, float)
                                                        and np.isnan(sector_nombre))) else "—")

            _chart_h = 300

            col_vela, col_info = st.columns([1, 4])

            with col_vela:
                fig = go.Figure()
                # Barra completamente vacía: potencial sin realizar en 2026
                fig.add_trace(go.Bar(
                    x=[""], y=[fac_m],
                    marker=dict(color=PASTEL_EMPTY, line=dict(color="#ddd", width=1)),
                    width=0.45,
                    hovertemplate=f"Potencial: {eu(fac_m,2)} M€<extra></extra>",
                ))
                # Línea máximo histórico
                fig.add_shape(type="line", x0=-0.35, x1=0.35, y0=fac_m, y1=fac_m,
                              line=dict(color=ZUKAN_BLACK, width=1.2, dash="dot"))
                fig.add_annotation(x=0, y=fac_m, yshift=7,
                    text=f"<b>Máx. {ano_m}</b>: {eu(fac_m,2)} M€",
                    showarrow=False, xanchor="center",
                    font=dict(size=8, color=ZUKAN_BLACK), bgcolor="white", borderpad=1)
                # Línea de referencia: último año activo
                if fac_ult > 0:
                    fig.add_shape(type="line", x0=-0.35, x1=0.35, y0=fac_ult, y1=fac_ult,
                                  line=dict(color=ZUKAN_BLUE, width=1.5))
                    fig.add_annotation(x=0, y=fac_ult, yshift=-10,
                        text=f"<b>{ano_ult}</b>: {eu(fac_ult,2)} M€",
                        showarrow=False, xanchor="center",
                        font=dict(size=8, color=ZUKAN_BLUE), bgcolor="white", borderpad=1)
                fig.update_layout(
                    barmode="stack", height=_chart_h,
                    margin=dict(l=25, r=15, t=8, b=8),
                    showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(visible=False),
                    yaxis=dict(title="M€", gridcolor="#f5f5f5", zeroline=False,
                               tickfont=dict(size=8)),
                    font=dict(family="sans-serif", size=9, color=ZUKAN_BLACK),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"vela_past_{cod}")

            with col_info:
                st.markdown(f"""
<div style="padding:4px 0 8px 0">
  <div style="font-size:0.9em;font-weight:600;color:{ZUKAN_BLACK}">
    {cli_n}
    <span style="font-size:0.78em;font-weight:400;color:{ZUKAN_BLUE};margin-left:8px">{sec_label}</span>
    <span style="font-size:0.72em;font-weight:400;color:{ZUKAN_RED};margin-left:8px">Sin actividad en 2026</span>
  </div>
  <div style="display:flex;gap:20px;margin-top:6px;flex-wrap:wrap">
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Último año activo</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{ano_ult}</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Fac. {ano_ult}</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{eu(fac_ult,2)} M€</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Toneladas</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{"—" if np.isnan(tn_ult) else eutn(tn_ult)} TN</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">PvP</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{"—" if np.isnan(pvp_ult) else eutn(pvp_ult)} €/TN</span></div>
    <div><span style="font-size:0.68em;color:#999;text-transform:uppercase">Máx. hist. ({ano_m})</span><br>
         <span style="font-size:0.85em;color:{ZUKAN_BLACK}">{"—" if (not fac_m or np.isnan(fac_m)) else eu(fac_m,2)} M€</span></div>
  </div>
  <div style="margin-top:8px;background:#fff5f7;border-left:3px solid {ZUKAN_RED};
              padding:8px 12px;border-radius:0 4px 4px 0;display:inline-block;min-width:260px">
    <div style="font-size:0.65em;color:#888;text-transform:uppercase;letter-spacing:0.06em">
      Oportunidad de reactivación
    </div>
    <div style="font-size:1.5em;font-weight:700;color:{ZUKAN_RED};line-height:1.2">
      {"—" if np.isnan(rec_tn) else f"+{eutn(rec_tn)} TN"}
    </div>
    <div style="font-size:0.72em;color:#555;margin-top:2px">
      Recuperar nivel {ano_ult}: {eu(fac_ult,2)} M€ · PvP ref. {"—" if np.isnan(pvp_ult) else eutn(pvp_ult)} €/TN
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                if ano_m and ano_ult:
                    t1, t2, t3 = st.columns(3)
                    h_pico, h_act, h_gap = _three_tables(
                        cod, ano_m, ano_ult, str(ano_ult), pvp_ref_past, cpx_all
                    )
                    t1.markdown(h_pico, unsafe_allow_html=True)
                    t2.markdown(h_act,  unsafe_allow_html=True)
                    t3.markdown(h_gap,  unsafe_allow_html=True)

            st.markdown("<hr style='margin:4px 0;border:none;border-top:1px solid #eee'>",
                        unsafe_allow_html=True)

    else:
        tab_leads, tab_mercado = st.tabs(["Leads", "Cuota de mercado"])

        with tab_leads:
            # ── LEADS · Similitud de ingredientes con clientes Zukán ──────────
            leads_df          = load_leads_cluster_data()
            pvp_lookup, sector_ings = load_ingrediente_sector_pvp()

            total_leads = len(leads_df)
            n1 = (leads_df["cluster"] == "1 - Alta").sum()
            n2 = (leads_df["cluster"] == "2 - Media").sum()
            n3 = (leads_df["cluster"] == "3 - Baja").sum()
            pot1 = leads_df[leads_df["cluster"]=="1 - Alta"]["potencial_est"].sum()
            pot2 = leads_df[leads_df["cluster"]=="2 - Media"]["potencial_est"].sum()
            pot3 = leads_df[leads_df["cluster"]=="3 - Baja"]["potencial_est"].sum()

            st.markdown(f"""
            <div style='display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px;'>
              <div style='flex:1;min-width:160px;background:#f8f9fa;border-radius:10px;
                          padding:14px 18px;border-left:4px solid {ZUKAN_BLACK};'>
                <div style='font-size:10px;color:#888;font-weight:700;text-transform:uppercase;'>Total leads externos</div>
                <div style='font-size:2rem;font-weight:800;color:{ZUKAN_BLACK};'>{total_leads}</div>
                <div style='font-size:10px;color:#aaa;'>Innova / Carrefour / El Corte Inglés / Mercadona</div>
              </div>
              <div style='flex:1;min-width:160px;background:#e8f5e9;border-radius:10px;
                          padding:14px 18px;border-left:4px solid {ZUKAN_GREEN};'>
                <div style='font-size:10px;color:#388e3c;font-weight:700;text-transform:uppercase;'>Cluster 1 · Alta similitud (&ge;75%)</div>
                <div style='font-size:2rem;font-weight:800;color:{ZUKAN_GREEN};'>{n1}</div>
                <div style='font-size:10px;color:#aaa;'>Pot. est.: {eu(pot1/1e6,2)} M€</div>
              </div>
              <div style='flex:1;min-width:160px;background:#fff8e1;border-radius:10px;
                          padding:14px 18px;border-left:4px solid #F4A261;'>
                <div style='font-size:10px;color:#e65100;font-weight:700;text-transform:uppercase;'>Cluster 2 · Media similitud (50-75%)</div>
                <div style='font-size:2rem;font-weight:800;color:#E76F51;'>{n2}</div>
                <div style='font-size:10px;color:#aaa;'>Pot. est.: {eu(pot2/1e6,2)} M€</div>
              </div>
              <div style='flex:1;min-width:160px;background:#fce4ec;border-radius:10px;
                          padding:14px 18px;border-left:4px solid {ZUKAN_RED};'>
                <div style='font-size:10px;color:#c62828;font-weight:700;text-transform:uppercase;'>Cluster 3 · Baja similitud (&lt;50%)</div>
                <div style='font-size:2rem;font-weight:800;color:{ZUKAN_RED};'>{n3}</div>
                <div style='font-size:10px;color:#aaa;'>Pot. est.: {eu(pot3/1e6,2)} M€</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Desglose por nivel de potencial (Cluster 1)
            _sub1_niv = leads_df[leads_df["cluster"] == "1 - Alta"]["potencial_est"].apply(_nivel_potencial).value_counts()
            _niv_badges = "".join(
                f"<span style='display:inline-block;margin:2px 4px 2px 0;"
                f"padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700;"
                f"color:white;background:{nc};'>{nl} · {_sub1_niv.get(nl, 0)} leads</span>"
                for nl, _, _, _, nc in _POT_NIVELES
            )
            st.markdown(
                f"<div style='margin-bottom:12px;'>"
                f"<span style='font-size:10px;color:#888;font-weight:600;text-transform:uppercase;"
                f"letter-spacing:.6px;margin-right:8px;'>Cluster 1 por nivel de potencial:</span>"
                f"{_niv_badges}</div>",
                unsafe_allow_html=True,
            )

            _ins4 = load_lqfb_insights()
            st.markdown(_yellow_block(
                "Leads con mayor afinidad a la cartera de alto valor LQ/FB",
                f"<p style='font-size:12px;color:#555;margin:0 0 5px 0;'>"
                f"Sectores como lácteos, bebidas, confitería industrial y nutrición deportiva "
                f"concentran la mayor demanda de la <b>cartera estratégica</b>: edulcorantes líquidos "
                f"de alto valor (soluciones a medida frente a commodity) y fibras solubles "
                f"(enriquecimiento nutricional). Los leads cluster 1 de estos sectores son el encaje "
                f"más directo con los 125 productos de alto valor añadido de Zukán.</p>"
                f"<p style='font-size:12px;color:#555;margin:0;'>"
                f"Referencia interna: hay <b>{_ins4['n_rec_lqfb']} clientes anteriores</b> con historial "
                f"en la cartera estratégica que ya no compran — su propuesta de valor probada puede "
                f"transferirse directamente a leads en los mismos sectores.</p>"
            ), unsafe_allow_html=True)

            tidx1, tidx2, tidx3 = st.tabs([
                f"Cluster 1 — Alta similitud ({n1})",
                f"Cluster 2 — Media similitud ({n2})",
                f"Cluster 3 — Baja similitud ({n3})",
            ])

            def _render_lead_detail(lead, vela_key):
                all_sectors  = lead.get("sector_leads_list") or []
                if isinstance(all_sectors, str):
                    all_sectors = [s.strip() for s in all_sectors.split(",") if s.strip()]
                sk_best      = lead.get("sector_lead") or (all_sectors[0] if all_sectors else "")
                ings_raw     = lead.get("ingredientes_lead") or ""
                ings_lead    = [i.strip() for i in str(ings_raw).split(",") if i.strip()] if ings_raw else []
                ings_lead_keys = {_norm(i) for i in ings_lead}

                jaccard   = lead.get("jaccard_max", 0) or 0
                ref_cli   = lead.get("nombre_cliente") or "—"
                n_comunes = lead.get("n_comunes", 0) or 0
                ali_eur   = lead.get("alimarket_ventas_eur")

                # Badges de sectores
                _sec_colors = ["#1565C0","#2E7D32","#6A1B9A","#BF360C","#00695C","#4527A0","#558B2F"]
                sec_badges_html = " ".join(
                    f"<span style='display:inline-block;margin:2px 3px 2px 0;"
                    f"padding:2px 9px;border-radius:10px;font-size:10px;font-weight:600;"
                    f"color:white;background:{_sec_colors[i % len(_sec_colors)]};'>"
                    f"{sk.title()}</span>"
                    for i, sk in enumerate(all_sectors)
                ) if all_sectors else "<span style='color:#aaa;font-size:10px;'>—</span>"

                # ── Tarjeta de similitud ──────────────────────────────────────
                ali_html = ""
                if ali_eur and not (isinstance(ali_eur, float) and np.isnan(ali_eur)):
                    ali_html = (f"<div><div style='font-size:9px;color:#aaa;text-transform:uppercase;"
                                f"font-weight:600;'>Facturación Alimarket</div>"
                                f"<div style='font-size:1rem;font-weight:700;color:{ZUKAN_BLACK};'>"
                                f"{eu(float(ali_eur)/1e6,1)} M€</div></div>")

                st.markdown(
                    f"<div style='background:#f0f7ff;border-radius:10px;padding:16px 20px;"
                    f"border-left:4px solid {ZUKAN_BLUE};margin-bottom:10px;'>"
                    f"  <div style='font-size:10px;color:#6c757d;font-weight:700;text-transform:uppercase;"
                    f"              letter-spacing:.7px;margin-bottom:10px;'>Similitud con clientes Zukán</div>"
                    f"  <div style='display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start;'>"
                    f"    <div><div style='font-size:9px;color:#aaa;text-transform:uppercase;font-weight:600;'>Similitud Jaccard</div>"
                    f"         <div style='font-size:2rem;font-weight:800;color:{ZUKAN_BLUE};'>"
                    f"           {jaccard*100:.0f}<span style='font-size:1rem;font-weight:400;color:#888;'>%</span></div></div>"
                    f"    <div><div style='font-size:9px;color:#aaa;text-transform:uppercase;font-weight:600;'>Mejor referencia</div>"
                    f"         <div style='font-size:1rem;font-weight:700;color:{ZUKAN_BLACK};margin-top:4px;'>{ref_cli[:40]}</div>"
                    f"         <div style='font-size:10px;color:#aaa;'>{n_comunes} ingrediente(s) comun(es)</div></div>"
                    f"    <div style='flex:1;min-width:160px;'>"
                    f"         <div style='font-size:9px;color:#aaa;text-transform:uppercase;font-weight:600;"
                    f"         margin-bottom:4px;'>Sectores Zukán ({len(all_sectors)})</div>"
                    f"         <div style='line-height:1.8;'>{sec_badges_html}</div></div>"
                    f"    {ali_html}"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Tarjeta azul: ingredientes (1 por ingrediente, mejor sector) ─
                if ings_lead and all_sectors:
                    pot_grand = 0.0
                    tn_grand  = 0.0
                    rows_html = ""
                    _sec_color_map = {sk: _sec_colors[i % len(_sec_colors)]
                                      for i, sk in enumerate(all_sectors)}
                    for ing in ings_lead:
                        ik = _norm(ing)
                        best_pot, best_pvp, best_tn, best_sk = 0.0, 0.0, 0.0, None
                        for sk in all_sectors:
                            data = pvp_lookup.get((ik, sk))
                            if data and data["tn_media"] > 0 and data["pvp_medio"] > 0:
                                pot = data["tn_media"] * data["pvp_medio"]
                                if pot > best_pot:
                                    best_pot = pot
                                    best_pvp = data["pvp_medio"]
                                    best_tn  = data["tn_media"]
                                    best_sk  = sk
                        if best_sk is None:
                            continue
                        pot_grand += best_pot
                        tn_grand  += best_tn
                        sc = _sec_color_map.get(best_sk, "#555")
                        rows_html += (
                            f"<div style='display:flex;align-items:center;gap:10px;padding:5px 0;"
                            f"border-bottom:1px solid #e0eaf8;'>"
                            f"  <span style='background:{ZUKAN_BLUE};color:white;padding:2px 8px;"
                            f"  border-radius:4px;font-size:10px;font-weight:600;min-width:160px;"
                            f"  text-align:center;flex-shrink:0;'>{ing}</span>"
                            f"  <span style='background:{sc};color:white;padding:1px 7px;"
                            f"  border-radius:8px;font-size:9px;font-weight:600;min-width:90px;"
                            f"  text-align:center;flex-shrink:0;'>{best_sk.title()}</span>"
                            f"  <span style='font-size:10px;color:#555;min-width:90px;'>"
                            f"    <b>{eu(best_pvp,0)}</b> €/TN</span>"
                            f"  <span style='font-size:10px;color:#555;min-width:80px;'>"
                            f"    <b>{eutn(best_tn)}</b> TN/año</span>"
                            f"  <span style='font-size:10px;color:{ZUKAN_GREEN};font-weight:700;'>"
                            f"    {eu(best_pot,0)} €</span>"
                            f"</div>"
                        )
                    pvp_pond = (pot_grand / tn_grand) if tn_grand > 0 else 0
                    st.markdown(
                        f"<div style='background:#f0f7ff;border-radius:10px;padding:16px 20px;"
                        f"border-left:4px solid {ZUKAN_BLUE};margin-bottom:10px;'>"
                        f"  <div style='font-size:10px;color:#6c757d;font-weight:700;text-transform:uppercase;"
                        f"              letter-spacing:.7px;margin-bottom:4px;'>Ingredientes Detectados</div>"
                        f"  <div style='font-size:11px;color:#999;margin-bottom:10px;'>"
                        f"    Cada ingrediente usa el sector de mejor referencia PvP · Histórico 2022-2025</div>"
                        f"  <div style='display:flex;gap:8px;font-size:9px;color:#aaa;font-weight:700;"
                        f"       text-transform:uppercase;padding:0 0 4px;letter-spacing:.4px;'>"
                        f"    <span style='min-width:160px;'>Ingrediente</span>"
                        f"    <span style='min-width:90px;'>Sector ref.</span>"
                        f"    <span style='min-width:90px;'>PvP medio</span>"
                        f"    <span style='min-width:80px;'>TN media/año</span>"
                        f"    <span>Potencial €/año</span>"
                        f"  </div>"
                        f"  {rows_html}"
                        f"  <div style='margin-top:10px;padding-top:8px;border-top:2px solid #d0e4f8;"
                        f"       display:flex;gap:32px;'>"
                        f"    <div><div style='font-size:9px;color:#aaa;text-transform:uppercase;"
                        f"         font-weight:600;'>Potencial estimado</div>"
                        f"         <div style='font-size:1.6rem;font-weight:800;color:{ZUKAN_GREEN};'>"
                        f"           {eu(pot_grand/1e6,2)} M€</div></div>"
                        f"    <div><div style='font-size:9px;color:#aaa;text-transform:uppercase;"
                        f"         font-weight:600;'>Volumen estimado</div>"
                        f"         <div style='font-size:1.6rem;font-weight:800;color:{ZUKAN_BLACK};'>"
                        f"           {eutn(tn_grand)} TN</div></div>"
                        f"    <div><div style='font-size:9px;color:#aaa;text-transform:uppercase;"
                        f"         font-weight:600;'>PvP medio ponderado</div>"
                        f"         <div style='font-size:1.6rem;font-weight:800;color:{ZUKAN_BLACK};'>"
                        f"           {eu(pvp_pond,0)} €/TN</div></div>"
                        f"  </div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Tarjeta verde: upsell — unión de todos los sectores ───────
                all_upsell: dict[str, dict] = {}
                for sk in all_sectors:
                    for ik in sector_ings.get(sk, []):
                        if ik not in ings_lead_keys and ik not in all_upsell:
                            data = pvp_lookup.get((ik, sk), {})
                            all_upsell[ik] = {
                                "display": data.get("ingrediente_display", ik.title()),
                                "pvp": data.get("pvp_medio", 0) or 0,
                                "tn":  data.get("tn_media",  0) or 0,
                                "sector": sk.title(),
                            }
                if all_upsell:
                    verde_rows = ""
                    for ik, d in all_upsell.items():
                        pvp_str_i = eu(d["pvp"], 0) if d["pvp"] > 0 else "—"
                        tn_str_i  = eutn(d["tn"])   if d["tn"]  > 0 else "—"
                        verde_rows += (
                            f"<div style='display:flex;align-items:center;gap:10px;padding:5px 0;"
                            f"border-bottom:1px solid #d4f0e8;'>"
                            f"  <span style='background:#2A9D8F;color:white;padding:2px 8px;"
                            f"  border-radius:4px;font-size:10px;font-weight:600;min-width:160px;"
                            f"  text-align:center;flex-shrink:0;'>{d['display']}</span>"
                            f"  <span style='font-size:9px;color:#aaa;min-width:110px;'>{d['sector']}</span>"
                            f"  <span style='font-size:10px;color:#555;min-width:95px;'>"
                            f"    <b>{pvp_str_i}</b> €/TN</span>"
                            f"  <span style='font-size:10px;color:#555;'>"
                            f"    <b>{tn_str_i}</b> TN/año</span>"
                            f"</div>"
                        )
                    st.markdown(
                        f"<div style='background:#f0faf8;border-radius:10px;padding:16px 20px;"
                        f"border-left:4px solid #2A9D8F;margin-bottom:10px;'>"
                        f"  <div style='font-size:10px;color:#6c757d;font-weight:700;text-transform:uppercase;"
                        f"              letter-spacing:.7px;margin-bottom:4px;'>Oportunidades de Upsell</div>"
                        f"  <div style='font-size:11px;color:#999;margin-bottom:10px;'>"
                        f"    Ingredientes Zukán no detectados en este lead, por sector · Palanca de cross-sell</div>"
                        f"  <div style='display:flex;gap:8px;font-size:9px;color:#aaa;font-weight:700;"
                        f"       text-transform:uppercase;padding:0 0 4px;letter-spacing:.4px;'>"
                        f"    <span style='min-width:160px;'>Ingrediente</span>"
                        f"    <span style='min-width:110px;'>Sector</span>"
                        f"    <span style='min-width:95px;'>PvP sector</span>"
                        f"    <span>TN media sector/año</span>"
                        f"  </div>"
                        f"  {verde_rows}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            def _render_leads_table(subset, color, plan_texto, idx_key):
                st.markdown(
                    f"<div style='background:#f8f9fa;border-left:4px solid {color};"
                    f"border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:12px;'>"
                    f"<b>Plan de activación:</b> {plan_texto}</div>",
                    unsafe_allow_html=True,
                )
                subset_s = subset.sort_values("score", ascending=False).reset_index(drop=True)

                tbl = subset_s[["empresa_lead","sector_lead","jaccard_max",
                                 "n_productos_potenciales","alimarket_ventas_eur",
                                 "score","potencial_est","origen"]].copy()
                tbl.columns = ["Empresa","Sector","Similitud","Productos pot.",
                               "Facturación Ali. (€)","Score","Pot. est. (€)","Origen"]
                tbl.insert(2, "Nivel", subset_s["potencial_est"].apply(_nivel_potencial))
                tbl["Similitud"]         = tbl["Similitud"].map(lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—")
                tbl["Score"]             = tbl["Score"].map(lambda v: f"{v:.2f}".replace(".",",") if pd.notna(v) else "—")
                tbl["Pot. est. (€)"]     = tbl["Pot. est. (€)"].map(lambda v: eu(v, 0) if pd.notna(v) and v > 0 else "—")
                tbl["Facturación Ali. (€)"] = tbl["Facturación Ali. (€)"].map(
                    lambda v: eu(float(v)/1e6,1)+" M€" if pd.notna(v) and float(v) > 0 else "—")
                tbl["Sector"] = tbl["Sector"].str.title().fillna("—")

                ev = st.dataframe(
                    tbl, use_container_width=True, hide_index=True, height=350,
                    on_select="rerun", selection_mode="multi-row",
                    key=f"tbl_{idx_key}",
                )
                sel = ev.selection.rows
                for row_idx in sel:
                    lead_row = subset_s.iloc[row_idx]
                    sector_disp = str(lead_row.get("sector_lead") or "").title()
                    st.divider()
                    st.markdown(
                        f"<div style='font-size:13px;font-weight:700;color:{ZUKAN_BLACK};"
                        f"margin-bottom:6px;'>{lead_row['empresa_lead']}"
                        f"<span style='font-size:11px;font-weight:400;color:#888;margin-left:10px;'>"
                        f"{sector_disp}"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )
                    _render_lead_detail(lead_row, vela_key=f"vela_{idx_key}_{row_idx}")

            with tidx1:
                sub1 = leads_df[leads_df["cluster"] == "1 - Alta"]
                _render_leads_table(sub1, ZUKAN_GREEN,
                    "Contacto directo inmediato — similitud Jaccard >= 75% con clientes actuales. "
                    "Preparar oferta con los ingredientes detectados y proponer los ingredientes verdes "
                    "como oportunidad de ampliación de cartera.",
                    "idx1")

            with tidx2:
                sub2 = leads_df[leads_df["cluster"] == "2 - Media"]
                _render_leads_table(sub2, "#E76F51",
                    "Seguimiento activo — similitud 50-75%. Campaña de nurturing sectorial con casos "
                    "de éxito en ingredientes comunes. Revisar cada 30-45 días.",
                    "idx2")

            with tidx3:
                sub3 = leads_df[leads_df["cluster"] == "3 - Baja"]
                _render_leads_table(sub3, ZUKAN_RED,
                    "Activación a largo plazo — similitud < 50%. Incluir en newsletters y eventos. "
                    "Revisar trimestralmente si aparecen nuevos ingredientes comunes.",
                    "idx3")

        with tab_mercado:
            # ── NO CLIENTES · Cuota de mercado por sector ─────────────────────
            tam_df = load_circana_data()

            tot_tam_eur = tam_df["tam_eur_2025"].sum()
            tot_fac_25  = tam_df["fac_2025"].sum()
            tot_fac_26  = tam_df["fac_2026"].sum()
            tot_tam_tn  = tam_df["tam_tn_2025"].sum()
            tot_tn_25   = tam_df["tn_2025"].sum()
            tot_tn_26   = tam_df["tn_2026"].sum()
            pct_eur_tot = tot_fac_25 / tot_tam_eur * 100 if tot_tam_eur > 0 else 0
            pct_tn_tot  = tot_tn_25  / tot_tam_tn  * 100 if tot_tam_tn  > 0 else 0

            st.markdown(f"""
            <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;'>

              <div style='flex:1;min-width:220px;background:#f8f9fa;border-radius:12px;
                          padding:18px 22px;border-left:4px solid {ZUKAN_BLUE};'>
                <div style='font-size:10px;color:#888;font-weight:700;text-transform:uppercase;
                            letter-spacing:.6px;margin-bottom:12px;'>Mercado TAM — jun 2025</div>
                <div style='display:flex;gap:0;'>
                  <div style='flex:1;border-right:1px solid #e0e0e0;padding-right:14px;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>Facturación</div>
                    <div style='font-size:1.55rem;font-weight:700;color:{ZUKAN_BLACK};line-height:1.1;'>{eu(tot_tam_eur/1e6,1)}<span style='font-size:.85rem;font-weight:400;color:#666;'> M€</span></div>
                  </div>
                  <div style='flex:1;padding-left:14px;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>Volumen</div>
                    <div style='font-size:1.55rem;font-weight:700;color:{ZUKAN_BLACK};line-height:1.1;'>{eutn(tot_tam_tn)}<span style='font-size:.85rem;font-weight:400;color:#666;'> TN</span></div>
                  </div>
                </div>
              </div>

              <div style='flex:1.6;min-width:300px;background:#f8f9fa;border-radius:12px;
                          padding:18px 22px;border-left:4px solid {ZUKAN_GREEN};'>
                <div style='font-size:10px;color:#888;font-weight:700;text-transform:uppercase;
                            letter-spacing:.6px;margin-bottom:12px;'>Cuota Zukán 2025</div>
                <div style='display:flex;gap:0;align-items:center;'>
                  <div style='flex:1;border-right:1px solid #e0e0e0;padding-right:14px;text-align:center;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>% sobre TAM</div>
                    <div style='font-size:2rem;font-weight:800;color:{ZUKAN_GREEN};line-height:1.1;'>{eu(pct_eur_tot,2)}<span style='font-size:1rem;'>%</span></div>
                  </div>
                  <div style='flex:1;border-right:1px solid #e0e0e0;padding:0 14px;text-align:center;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>Facturación</div>
                    <div style='font-size:1.55rem;font-weight:700;color:{ZUKAN_BLACK};line-height:1.1;'>{eu(tot_fac_25/1e6,2)}<span style='font-size:.85rem;font-weight:400;color:#666;'> M€</span></div>
                  </div>
                  <div style='flex:1;padding-left:14px;text-align:center;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>Volumen</div>
                    <div style='font-size:1.55rem;font-weight:700;color:{ZUKAN_BLACK};line-height:1.1;'>{eutn(tot_tn_25)}<span style='font-size:.85rem;font-weight:400;color:#666;'> TN</span></div>
                  </div>
                </div>
              </div>

              <div style='flex:1;min-width:220px;background:#f8f9fa;border-radius:12px;
                          padding:18px 22px;border-left:4px solid {ZUKAN_BLUE};'>
                <div style='font-size:10px;color:#888;font-weight:700;text-transform:uppercase;
                            letter-spacing:.6px;margin-bottom:12px;'>Cartera 2026 (acumulado)</div>
                <div style='display:flex;gap:0;'>
                  <div style='flex:1;border-right:1px solid #e0e0e0;padding-right:14px;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>Facturación</div>
                    <div style='font-size:1.55rem;font-weight:700;color:{ZUKAN_BLACK};line-height:1.1;'>{eu(tot_fac_26/1e6,2)}<span style='font-size:.85rem;font-weight:400;color:#666;'> M€</span></div>
                  </div>
                  <div style='flex:1;padding-left:14px;'>
                    <div style='font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;margin-bottom:3px;'>Volumen</div>
                    <div style='font-size:1.55rem;font-weight:700;color:{ZUKAN_BLACK};line-height:1.1;'>{eutn(tot_tn_26)}<span style='font-size:.85rem;font-weight:400;color:#666;'> TN</span></div>
                  </div>
                </div>
              </div>

            </div>
            """, unsafe_allow_html=True)

            # ── Donuts cuota de mercado ───────────────────────────────────────
            _TOP_N = 12
            _PALETTE = [
                ZUKAN_GREEN, ZUKAN_BLUE, "#F4A261", "#E76F51", "#2A9D8F",
                "#264653", "#A8DADC", "#457B9D", "#E9C46A", "#F4E285",
                "#B5838D", "#6D6875", "#C0C0C0",
            ]

            def _prep_donut(series, labels_col, n=_TOP_N):
                df_s = pd.DataFrame({"label": labels_col, "val": series})
                df_s = df_s.sort_values("val", ascending=False).reset_index(drop=True)
                otros_df = pd.DataFrame()
                if len(df_s) > n:
                    top = df_s.iloc[:n]
                    otros_df = df_s.iloc[n:].copy()
                    otros = pd.DataFrame([{"label": "Otros", "val": otros_df["val"].sum()}])
                    df_s = pd.concat([top, otros], ignore_index=True)
                colors = _PALETTE[:len(df_s)]
                return df_s["label"].tolist(), df_s["val"].tolist(), colors, otros_df

            _tab_fac, _tab_vol = st.tabs(["Facturación", "Volumen"])

            with _tab_fac:
                lbl, val, col, _otros_tam_eur = _prep_donut(tam_df["tam_eur_2025"], tam_df["ZUKAN"])
                lbl2, val2, col2, _otros_zukan_eur = _prep_donut(tam_df["fac_2025"], tam_df["ZUKAN"])
                dc1, dc2 = st.columns(2)
                with dc1:
                    fig_d1 = _donut_fig(lbl, val, col, "Mercado total por sector (TAM €)", "€", height=480)
                    st.plotly_chart(fig_d1, use_container_width=True, key="donut_tam_eur")
                with dc2:
                    fig_d2 = _donut_fig(lbl2, val2, col2, "Facturación Zukán 2025 por sector", "€", height=480)
                    st.plotly_chart(fig_d2, use_container_width=True, key="donut_zukan_eur")

                _has_otros_fac = not _otros_zukan_eur.empty
                if _has_otros_fac:
                    with st.expander(f"Desglose 'Otros' — Facturación ({len(_otros_zukan_eur)} sectores)"):
                        _df_fac = _otros_zukan_eur.sort_values("val", ascending=False).reset_index(drop=True)
                        _fig_fac = go.Figure(go.Bar(
                            x=_df_fac["val"] / 1e6,
                            y=_df_fac["label"],
                            orientation="h",
                            marker_color=ZUKAN_BLUE,
                            text=[f"{eu(v/1e6,2)} M€" for v in _df_fac["val"]],
                            textposition="outside",
                        ))
                        _fig_fac.update_layout(
                            height=max(320, len(_df_fac) * 30 + 60),
                            margin=dict(l=10, r=90, t=20, b=20),
                            plot_bgcolor="white", paper_bgcolor="white",
                            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            xaxis=dict(title="M€", tickfont=dict(size=9), gridcolor="#f0f0f0"),
                            font=dict(size=10),
                        )
                        st.plotly_chart(_fig_fac, use_container_width=True, key="otros_fac_bar")

            with _tab_vol:
                lbl3, val3, col3, _otros_tam_tn = _prep_donut(tam_df["tam_tn_2025"], tam_df["ZUKAN"])
                lbl4, val4, col4, _otros_zukan_tn = _prep_donut(tam_df["tn_2025"], tam_df["ZUKAN"])
                dc3, dc4 = st.columns(2)
                with dc3:
                    fig_d3 = _donut_fig(lbl3, val3, col3, "Mercado total por sector (TAM TN)", "TN", height=480)
                    st.plotly_chart(fig_d3, use_container_width=True, key="donut_tam_tn")
                with dc4:
                    fig_d4 = _donut_fig(lbl4, val4, col4, "Volumen Zukán 2025 por sector", "TN", height=480)
                    st.plotly_chart(fig_d4, use_container_width=True, key="donut_zukan_tn")

                _has_otros_tn = not _otros_zukan_tn.empty
                if _has_otros_tn:
                    with st.expander(f"Desglose 'Otros' — Volumen ({len(_otros_zukan_tn)} sectores)"):
                        _df_tn = _otros_zukan_tn.sort_values("val", ascending=False).reset_index(drop=True)
                        _fig_tn = go.Figure(go.Bar(
                            x=_df_tn["val"],
                            y=_df_tn["label"],
                            orientation="h",
                            marker_color=ZUKAN_GREEN,
                            text=[f"{eutn(v)} TN" for v in _df_tn["val"]],
                            textposition="outside",
                        ))
                        _fig_tn.update_layout(
                            height=max(320, len(_df_tn) * 30 + 60),
                            margin=dict(l=10, r=90, t=20, b=20),
                            plot_bgcolor="white", paper_bgcolor="white",
                            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            xaxis=dict(title="TN", tickfont=dict(size=9), gridcolor="#f0f0f0"),
                            font=dict(size=10),
                        )
                        st.plotly_chart(_fig_tn, use_container_width=True, key="otros_tn_bar")

            with st.expander("Ver tabla detalle por sector"):
                tbl = pd.DataFrame({
                    "Sector":              tam_df["ZUKAN"],
                    "TAM € YA-2025 (M€)": tam_df["tam_eur_2025"].map(lambda v: eu(v/1e6, 1)),
                    "Zukán € 2025 (M€)":  tam_df["fac_2025"].map(lambda v: eu(v/1e6, 2)),
                    "Cuota €":             tam_df["pct_eur"].map(lambda v: eu(v, 2) + "%"),
                    "Zukán € 2026 (M€)":  tam_df["fac_2026"].map(lambda v: eu(v/1e6, 2)),
                    "TAM TN YA-2025":      tam_df["tam_tn_2025"].map(lambda v: eutn(v)),
                    "Zukán TN 2025":       tam_df["tn_2025"].map(lambda v: eutn(v)),
                    "Cuota TN":            tam_df["pct_tn"].map(lambda v: eu(v, 2) + "%"),
                    "Zukán TN 2026":       tam_df["tn_2026"].map(lambda v: eutn(v)),
                    "Clientes 2025":       tam_df["n_cli_2025"].map(lambda v: str(int(v)) if v > 0 else "—"),
                })
                st.dataframe(tbl, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA IMPACTO ZUKÁN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["pagina"] == "impacto":

    if st.button("Volver al inicio"):
        st.session_state["pagina"] = "home"
        st.rerun()

    imp  = load_impacto_data()
    fac_a = imp["fac_actual"]
    gap_a = imp["gap_actuales"]
    rec_p = imp["recuperacion_pasados"]
    pot_l = imp["potencial_leads"]
    n_ag  = imp["n_act_gap"]
    n_p   = imp["n_pasados"]
    n_l1  = imp["n_leads_1"]
    _leads_niv = imp["leads_nivel_df"]
    _NIV_COLORS = {n[0]: n[4] for n in _POT_NIVELES}
    _NIV_LABELS = {n[0]: n[1] for n in _POT_NIVELES}

    # ── Cabecera ──────────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='color:{ZUKAN_BLACK};margin-bottom:4px;'>Plan Comercial 2027 — Simulación de Impacto</h2>"
        f"<p style='color:#888;font-size:13px;margin-top:0;'>"
        f"Estimación de la facturación alcanzable en 2027 ejecutando las tres palancas comerciales: "
        f"recuperar el máximo histórico de clientes activos, reactivar clientes pasados y convertir "
        f"leads prioritarios. Ajusta los parámetros para explorar distintos escenarios.</p>",
        unsafe_allow_html=True,
    )

    # ── Session state ─────────────────────────────────────────────────────────
    for _k, _v in [("imp_gap", 100), ("imp_past", 100)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v
    for _, _nrow in _leads_niv.iterrows():
        _nk = f"imp_niv_{_nrow['_nivel']}"
        if _nk not in st.session_state:
            st.session_state[_nk] = 100

    def _sl_changed(k):
        v = st.session_state[f"_sl_{k}"]
        st.session_state[k] = v
        st.session_state[f"_ni_{k}"] = v   # sync number input

    def _ni_changed(k):
        v = int(st.session_state[f"_ni_{k}"])
        st.session_state[k] = v
        st.session_state[f"_sl_{k}"] = v   # sync slider

    # ── Parámetros del escenario (área principal, arriba) ────────────────────
    with st.container(border=True):
        st.markdown(
            "<div style='font-size:10px;color:#6c757d;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.7px;margin-bottom:10px;'>Parámetros del escenario</div>",
            unsafe_allow_html=True,
        )

        # Fila 1: gap + pasados con slider+número
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            st.markdown(
                f"<div style='font-size:10px;color:#888;font-weight:600;margin-bottom:2px;'>"
                f"Recuperación gap clientes activos &nbsp;"
                f"<span style='font-weight:400;color:#bbb;'>Máx. {eu(gap_a,2)} M€ · {n_ag} clientes</span></div>",
                unsafe_allow_html=True,
            )
            _sca, _nca, _pca = st.columns([5, 1, 0.25])
            with _sca:
                st.slider("gap_lbl", 0, 100, step=5, format="%d%%", label_visibility="collapsed",
                          key="_sl_imp_gap", value=st.session_state["imp_gap"],
                          on_change=_sl_changed, args=("imp_gap",))
            with _nca:
                st.number_input("gap_n", 0, 100, step=5,
                                label_visibility="collapsed",
                                key="_ni_imp_gap", value=st.session_state["imp_gap"],
                                on_change=_ni_changed, args=("imp_gap",))
            with _pca:
                st.markdown("<div style='padding-top:30px;color:#888;font-weight:600'>%</div>",
                            unsafe_allow_html=True)
            pct_gap = st.session_state["imp_gap"]

        with _fc2:
            st.markdown(
                f"<div style='font-size:10px;color:#888;font-weight:600;margin-bottom:2px;'>"
                f"Reactivación clientes pasados &nbsp;"
                f"<span style='font-weight:400;color:#bbb;'>Máx. {eu(rec_p,2)} M€ · {n_p} clientes</span></div>",
                unsafe_allow_html=True,
            )
            _scb, _ncb, _pcb = st.columns([5, 1, 0.25])
            with _scb:
                st.slider("past_lbl", 0, 100, step=5, format="%d%%", label_visibility="collapsed",
                          key="_sl_imp_past", value=st.session_state["imp_past"],
                          on_change=_sl_changed, args=("imp_past",))
            with _ncb:
                st.number_input("past_n", 0, 100, step=5,
                                label_visibility="collapsed",
                                key="_ni_imp_past", value=st.session_state["imp_past"],
                                on_change=_ni_changed, args=("imp_past",))
            with _pcb:
                st.markdown("<div style='padding-top:30px;color:#888;font-weight:600'>%</div>",
                            unsafe_allow_html=True)
            pct_past = st.session_state["imp_past"]

        # Fila 2: leads Cluster 1 por nivel de potencial estimado
        st.markdown(
            "<div style='font-size:10px;color:#6c757d;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.6px;margin-top:10px;margin-bottom:8px;'>"
            "Leads Cluster 1 — % conversión por nivel de potencial estimado</div>",
            unsafe_allow_html=True,
        )
        _niv_cols = st.columns(4)
        for _ci, (_, _nr) in enumerate(_leads_niv.iterrows()):
            _nk    = _nr["_nivel"]
            _ss_k  = f"imp_niv_{_nk}"
            _ni_k  = f"_ni_{_ss_k}"
            _col   = _NIV_COLORS.get(_nk, "#888")
            _rng   = _NIV_LABELS.get(_nk, "")
            with _niv_cols[_ci]:
                st.markdown(
                    f"<div style='font-size:10px;font-weight:700;color:{_col};"
                    f"line-height:1.3;'>{_nk}</div>"
                    f"<div style='font-size:8px;color:#bbb;margin-bottom:2px;'>"
                    f"<span style='font-size:9px;color:#999;'>{_rng}</span></div>"
                    f"<div style='font-size:8px;color:#bbb;'>"
                    f"<span style='font-size:11px;font-weight:700;color:{_col};'>"
                    f"{int(_nr['n_leads'])}</span> leads · {eu(_nr['potencial_M'],2)} M€ máx.</div>",
                    unsafe_allow_html=True,
                )
                _inp_col, _pct_col = st.columns([4, 1])
                with _inp_col:
                    st.number_input(
                        _ss_k, 0, 100, step=5,
                        label_visibility="collapsed",
                        key=_ni_k, value=st.session_state[_ss_k],
                        on_change=lambda k=_ss_k: st.session_state.__setitem__(
                            k, int(st.session_state[f"_ni_{k}"])),
                    )
                with _pct_col:
                    st.markdown("<div style='padding-top:30px;color:#888;font-weight:600'>%</div>",
                                unsafe_allow_html=True)

    # ── Cálculo del escenario ajustado ────────────────────────────────────────
    gap_adj  = gap_a  * pct_gap  / 100
    rec_adj  = rec_p  * pct_past / 100
    pot_adj  = sum(
        _nr["potencial_M"] * st.session_state.get(f"imp_niv_{_nr['_nivel']}", 100) / 100
        for _, _nr in _leads_niv.iterrows()
    )
    pot_t    = fac_a + gap_adj + rec_adj + pot_adj
    uplift   = (pot_t / fac_a - 1) * 100 if fac_a > 0 else 0
    pct_leads_eff = pot_adj / pot_l * 100 if pot_l > 0 else 0
    n_leads_adj = int(sum(
        round(_nr["n_leads"] * st.session_state.get(f"imp_niv_{_nr['_nivel']}", 100) / 100)
        for _, _nr in _leads_niv.iterrows()
    ))

    # ── Banner de escenario ───────────────────────────────────────────────────
    _all_100 = (pct_gap == 100 and pct_past == 100 and abs(pct_leads_eff - 100) < 0.1)
    escenario_label = ("Escenario 100%" if _all_100
                       else f"Escenario ajustado ({pct_gap}% / {pct_past}% / leads {eu(pct_leads_eff,0)}%)")
    st.markdown(
        f"<div style='background:transparent;border-radius:12px;padding:22px 28px;"
        f"margin-bottom:18px;display:flex;align-items:center;justify-content:center;"
        f"gap:36px;flex-wrap:wrap;border:2px solid #C0392B;'>"
        f"  <div style='text-align:center;'>"
        f"    <div style='font-size:10px;color:#888;text-transform:uppercase;"
        f"                letter-spacing:.7px;margin-bottom:3px;'>Facturación actual 2026</div>"
        f"    <div style='font-size:2.4rem;font-weight:800;color:{ZUKAN_BLACK};line-height:1;'>"
        f"      {eu(fac_a,2)}"
        f"      <span style='font-size:1.1rem;font-weight:400;color:#999;'> M€</span></div>"
        f"  </div>"
        f"  <div style='font-size:2.4rem;color:{ZUKAN_GREEN};font-weight:300;line-height:1;'>→</div>"
        f"  <div style='text-align:center;'>"
        f"    <div style='font-size:10px;color:#888;text-transform:uppercase;"
        f"                letter-spacing:.7px;margin-bottom:3px;'>{escenario_label}</div>"
        f"    <div style='font-size:2.4rem;font-weight:800;color:{ZUKAN_GREEN};line-height:1;'>"
        f"      {eu(pot_t,2)}"
        f"      <span style='font-size:1.1rem;font-weight:400;color:#999;'> M€</span></div>"
        f"  </div>"
        f"  <div style='text-align:center;'>"
        f"    <div style='font-size:10px;color:#888;text-transform:uppercase;"
        f"                letter-spacing:.7px;margin-bottom:3px;'>Incremento potencial</div>"
        f"    <div style='font-size:2.4rem;font-weight:800;color:{ZUKAN_GREEN};line-height:1;'>"
        f"      +{eu(uplift,1)}%</div>"
        f"    <div style='font-size:11px;color:#666;margin-top:2px;'>"
        f"      +{eu(pot_t - fac_a,2)} M€ adicionales</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 4 KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    def _kpi_card(col, title, value_str, unit, subtitle, pct_label, color, border_color):
        col.markdown(
            f"<div style='background:white;border-radius:10px;padding:16px 12px;"
            f"border-top:4px solid {border_color};box-shadow:0 1px 4px rgba(0,0,0,.07);"
            f"text-align:center;'>"
            f"  <div style='font-size:9px;color:#999;text-transform:uppercase;"
            f"              font-weight:700;letter-spacing:.6px;margin-bottom:6px;'>{title}</div>"
            f"  <div style='font-size:1.9rem;font-weight:800;color:{color};line-height:1.1;'>"
            f"    {value_str}"
            f"    <span style='font-size:.9rem;font-weight:400;color:#aaa;'> {unit}</span></div>"
            f"  <div style='font-size:10px;color:#aaa;margin-top:5px;'>{subtitle}</div>"
            f"  <div style='font-size:10px;color:{border_color};font-weight:600;margin-top:3px;'>"
            f"    {pct_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _kpi_card(k1, "Facturación actual 2026",
              eu(fac_a, 2), "M€",
              "Base de referencia", "Datos reales",
              ZUKAN_BLACK, ZUKAN_BLACK)
    _kpi_card(k2, "Gap clientes actuales",
              eu(gap_adj, 2), "M€",
              f"{n_ag} clientes con histórico superior",
              f"{pct_gap}% del gap máximo ({eu(gap_a,2)} M€)",
              ZUKAN_BLUE, ZUKAN_BLUE)
    _kpi_card(k3, "Recuperación clientes pasados",
              eu(rec_adj, 2), "M€",
              f"{n_p} clientes sin actividad en 2026",
              f"{pct_past}% del potencial máx. ({eu(rec_p,2)} M€)",
              "#E76F51", "#E76F51")
    _kpi_card(k4, "Nuevos clientes — Leads Índice 1",
              eu(pot_adj, 2), "M€",
              f"{n_leads_adj} de {n_l1} leads convertidos",
              f"{eu(pct_leads_eff,0)}% efectivo ({eu(pot_l,2)} M€ máx.)",
              ZUKAN_GREEN, ZUKAN_GREEN)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Waterfall chart con colores por barra (simulado con go.Bar apilado) ───
    x_labs  = ["Actual 2026", "+ Gap actuales", "+ Pasados", "+ Leads Índice 1", "Potencial total"]
    bar_vals = [fac_a, gap_adj, rec_adj, pot_adj, pot_t]
    bar_base = [0,     fac_a,   fac_a+gap_adj, fac_a+gap_adj+rec_adj, 0]
    bar_cols = ["#1B4D3E", ZUKAN_BLUE, "#E76F51", "#C9B1D9", "#95D5B2"]
    bar_text = [f"<b>{eu(v,2)} M€</b>" for v in bar_vals]

    # Barra invisible de base
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(
        x=x_labs,
        y=bar_base,
        marker_color="rgba(0,0,0,0)",
        showlegend=False,
        hoverinfo="skip",
    ))
    # Barras visibles
    fig_wf.add_trace(go.Bar(
        x=x_labs,
        y=bar_vals,
        marker_color=bar_cols,
        marker_line=dict(color="white", width=1),
        text=bar_text,
        textposition="outside",
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>",
        customdata=[f"{eu(v,2)} M€" for v in bar_vals],
    ))
    # Líneas conectoras
    cum = fac_a
    shapes = []
    for v in [gap_adj, rec_adj, pot_adj]:
        shapes.append(dict(
            type="line", xref="x", yref="y",
            x0=x_labs.index(["+ Gap actuales","+ Pasados","+ Leads Índice 1"][[gap_adj,rec_adj,pot_adj].index(v)]) - 0.5,
            x1=x_labs.index(["+ Gap actuales","+ Pasados","+ Leads Índice 1"][[gap_adj,rec_adj,pot_adj].index(v)]) + 0.5,
            y0=cum + v, y1=cum + v,
            line=dict(color="#ccc", width=1, dash="dot"),
        ))
        cum += v

    fig_wf.update_layout(
        barmode="stack",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            title="M€",
            tickformat=".1f",
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
        ),
        xaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
        shapes=shapes,
    )
    st.plotly_chart(fig_wf, use_container_width=True, key="waterfall_impacto")

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:10px;color:#6c757d;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:.7px;margin-bottom:8px;margin-top:4px;'>Desglose por palanca</div>",
        unsafe_allow_html=True,
    )
    resumen_df = pd.DataFrame({
        "Palanca": [
            "Facturación actual 2026",
            "Gap clientes actuales",
            "Recuperación clientes pasados",
            "Leads prioritarios (Índice 1)",
            "Potencial total escenario",
        ],
        "Potencial máx. (M€)": [
            eu(fac_a, 2),
            eu(gap_a, 2),
            eu(rec_p, 2),
            eu(pot_l, 2),
            eu(fac_a + gap_a + rec_p + pot_l, 2),
        ],
        "Escenario ajustado (M€)": [
            eu(fac_a, 2),
            eu(gap_adj, 2),
            eu(rec_adj, 2),
            eu(pot_adj, 2),
            eu(pot_t, 2),
        ],
        "% captura": [
            "100%",
            f"{pct_gap}%",
            f"{pct_past}%",
            f"{eu(pct_leads_eff,0)}%",
            "—",
        ],
        "Sobre facturación actual": [
            "—",
            eu(gap_adj / fac_a * 100, 1) + "%" if fac_a > 0 else "—",
            eu(rec_adj / fac_a * 100, 1) + "%" if fac_a > 0 else "—",
            eu(pot_adj / fac_a * 100, 1) + "%" if fac_a > 0 else "—",
            "+" + eu(uplift, 1) + "%",
        ],
        "Clientes / Leads": [
            "—",
            str(n_ag),
            str(n_p),
            f"{n_leads_adj} de {n_l1}",
            str(n_ag + n_p + n_leads_adj),
        ],
    })
    st.dataframe(resumen_df, use_container_width=True, hide_index=True)
