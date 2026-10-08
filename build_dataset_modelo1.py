# ============================================================
# BUILD DATASET · MODELO 1 · CLUSTERING + COMPRA 2027
# Zukán · Inteligencia Comercial
# ============================================================
# Objetivo: construir paso a paso el dataset de features por
# cliente-año que alimentará el clustering y el modelo de
# probabilidad de compra.
#
# Output final: data/dataset_modelo1.csv
#   - Una fila por (cod_cliente, year)
#   - Todos los features RFM + cartera + maestro cliente
#   - Variable target: compra_next (0/1)
# ============================================================

# %% [markdown]
# ## 0. Dependencias

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ============================================================
# %% [markdown]
# ## 1. Carga del fichero fuente
# ============================================================

# %%
DATA_PATH = "data/Maestro_Análisis_Comercial.csv"

raw = pd.read_csv(DATA_PATH, encoding="utf-8", low_memory=False)

print(f"Filas:    {len(raw):>10,}")
print(f"Columnas: {len(raw.columns):>10,}")
print(f"\nColumnas disponibles:")
for c in raw.columns:
    dtype = str(raw[c].dtype)
    nulls = raw[c].isna().sum()
    print(f"  {c:<35} {dtype:<10} nulls: {nulls:,}")

# %%
# Filtrar ejercicios de interés y hacer copia de trabajo
ANOS   = [2021, 2022, 2023, 2024, 2025, 2026]
ANO_ACTUAL = 2026

df = raw[raw["ejercicio"].isin(ANOS)].copy()
df["fAlbaran"] = pd.to_datetime(df["fAlbaran"], errors="coerce")

# Asegurar tipos numéricos
for col in ["total_base_neto", "Cantidad_TN", "Cantidad_kg", "eur_por_tn"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"\nFilas tras filtrar 2021-2026: {len(df):,}")
print(f"Distribución por año:")
print(df["ejercicio"].value_counts().sort_index())

# ============================================================
# %% [markdown]
# ## 2. Calidad y anomalías
# ============================================================

# %%
# Líneas con facturación o TN negativas / nulas
neg_fac = df["total_base_neto"] < 0
neg_tn  = df["Cantidad_TN"] < 0
null_fac = df["total_base_neto"].isna()
null_tn  = df["Cantidad_TN"].isna()

print("Líneas con valores problemáticos:")
print(f"  total_base_neto < 0   : {neg_fac.sum():>7,}")
print(f"  total_base_neto nulo  : {null_fac.sum():>7,}")
print(f"  Cantidad_TN < 0       : {neg_tn.sum():>7,}")
print(f"  Cantidad_TN nulo      : {null_tn.sum():>7,}")

# %%
# ¿Cuánto representan en facturación las líneas negativas?
print(f"\nFac total positiva : {df.loc[~neg_fac,'total_base_neto'].sum()/1e6:,.1f} M€")
print(f"Fac total negativa : {df.loc[ neg_fac,'total_base_neto'].sum()/1e6:,.1f} M€")
print(f"Neto total         : {df['total_base_neto'].sum()/1e6:,.1f} M€")

# Decisión: mantener líneas negativas (abonos / devoluciones)
# ya están incluidas en la facturación neta del cliente

# %%
# Clientes que aparecen en cada año
print("\nPresencia de clientes por año:")
print(f"{'Año':<6} {'Clientes':<12}")
print("-" * 18)
for y in ANOS:
    n = df[df["ejercicio"] == y]["cod_cliente"].nunique()
    print(f"{y:<6} {n:<12,}")

# ============================================================
# %% [markdown]
# ## 3. Base transaccional por (cliente, año)
# ============================================================
# Primera agregación: resumen financiero y de volumen
# para cada par (cod_cliente, ejercicio)

# %%
base = (
    df.groupby(["cod_cliente", "ejercicio"])
    .agg(
        fac          = ("total_base_neto", "sum"),
        tn           = ("Cantidad_TN",     "sum"),
        n_pedidos    = ("id_pedido_vta",    "nunique"),
        n_albaranes  = ("id_albaran_vta",   "nunique"),
        n_lineas     = ("id_lin_albaran_vta","nunique"),
        n_skus       = ("cod_producto",     "nunique"),
        n_familias   = ("familia",          "nunique"),
        n_subfamilias= ("subfamilia",       "nunique"),
        primera_venta= ("fAlbaran",         "min"),
        ultima_venta = ("fAlbaran",         "max"),
    )
    .reset_index()
)

# PvP: facturación / TN (siempre sobre el total del año)
base["pvp"] = np.where(base["tn"] > 0, base["fac"] / base["tn"], np.nan)

print(f"Registros en base (cliente x año): {len(base):,}")
print(f"\nDistribución por año:")
print(base["ejercicio"].value_counts().sort_index())
print(f"\nVistazo:")
print(base.head(10).to_string(index=False))

# ============================================================
# %% [markdown]
# ## 4. Concentración de cartera (top producto)
# ============================================================
# Calculamos el % de facturación que representa el producto
# principal de cada cliente en cada año (índice HHI simplificado)

# %%
fac_prod = (
    df.groupby(["cod_cliente", "ejercicio", "cod_producto"])["total_base_neto"]
    .sum()
    .reset_index()
)

# Top producto por cliente-año
top_prod = (
    fac_prod.sort_values("total_base_neto", ascending=False)
    .groupby(["cod_cliente", "ejercicio"])
    .first()
    .reset_index()
    .rename(columns={"total_base_neto": "fac_top_prod", "cod_producto": "top_producto"})
)

# % del top producto sobre total
top_prod = top_prod.merge(
    base[["cod_cliente","ejercicio","fac"]],
    on=["cod_cliente","ejercicio"], how="left"
)
top_prod["conc_top1"] = np.where(
    top_prod["fac"] > 0,
    top_prod["fac_top_prod"] / top_prod["fac"],
    np.nan
)

base = base.merge(
    top_prod[["cod_cliente","ejercicio","conc_top1","top_producto"]],
    on=["cod_cliente","ejercicio"], how="left"
)

print("Concentración top producto:")
print(base["conc_top1"].describe().round(3))
print(f"\n  < 50% (cartera diversa)       : {(base['conc_top1']<0.5).sum():,}")
print(f"  50%-80% (1 producto dominante): {((base['conc_top1']>=0.5)&(base['conc_top1']<0.8)).sum():,}")
print(f"  > 80% (cliente mono-producto) : {(base['conc_top1']>=0.8).sum():,}")

# ============================================================
# %% [markdown]
# ## 5. Features de tendencia (variación año anterior)
# ============================================================

# %%
# Para cada cliente-año calculamos los valores del año anterior
base_sorted = base.sort_values(["cod_cliente", "ejercicio"])

base["fac_ant"] = (
    base_sorted.groupby("cod_cliente")["fac"]
    .shift(1)
    .values
)
base["tn_ant"] = (
    base_sorted.groupby("cod_cliente")["tn"]
    .shift(1)
    .values
)
base["pvp_ant"] = (
    base_sorted.groupby("cod_cliente")["pvp"]
    .shift(1)
    .values
)

# Comprobar que el año anterior es realmente el año anterior
# (puede haber saltos — en ese caso el shift no es válido)
base["ano_ant_esperado"] = base["ejercicio"] - 1
base["ano_ant_real"]     = base_sorted.groupby("cod_cliente")["ejercicio"].shift(1).values
base["hay_ant_valido"]   = base["ano_ant_real"] == base["ano_ant_esperado"]

# Si no hay año anterior válido, poner NaN
base.loc[~base["hay_ant_valido"], ["fac_ant","tn_ant","pvp_ant"]] = np.nan

# Tendencia en porcentaje
base["trend_fac_pct"] = np.where(
    base["fac_ant"] > 0,
    (base["fac"] - base["fac_ant"]) / base["fac_ant"] * 100,
    np.nan
)
base["trend_tn_pct"] = np.where(
    base["tn_ant"] > 0,
    (base["tn"] - base["tn_ant"]) / base["tn_ant"] * 100,
    np.nan
)
base["delta_pvp"] = base["pvp"] - base["pvp_ant"]

print("Tendencia facturación (% YoY):")
print(base.groupby("ejercicio")["trend_fac_pct"].describe().round(1).to_string())

# ============================================================
# %% [markdown]
# ## 6. Features de frecuencia y continuidad (RFM)
# ============================================================

# %%
# Para cada cliente, calculamos:
#   - n_anos: nº de años con actividad (de 2021 a year)
#   - cont2:  compró en year Y year-1
#   - cont3:  compró en year, year-1, year-2
#   - recencia: años desde la última compra (siempre 0 si está activo)

# Tabla de presencia binaria por cliente-año
presencia = pd.crosstab(
    df["cod_cliente"], df["ejercicio"]
).clip(0, 1)  # 1 = activo ese año, 0 = no

print("Presencia binaria (primeras filas):")
print(presencia.head(10))

# %%
def frecuencia_continuidad(presencia_df, cod, year):
    """Calcula n_anos, cont2, cont3 para un cliente hasta `year`."""
    row = presencia_df.loc[cod] if cod in presencia_df.index else pd.Series(0, index=ANOS)
    anos_hasta = [y for y in ANOS if y <= year]
    n_anos = int(row.reindex(anos_hasta, fill_value=0).sum())
    cont2  = int(all(row.get(y, 0) for y in [year, year-1] if y >= 2021))
    cont3  = int(all(row.get(y, 0) for y in [year, year-1, year-2] if y >= 2021))
    return n_anos, cont2, cont3

# Añadir a base
n_anos_list = []
cont2_list  = []
cont3_list  = []

for _, row in base.iterrows():
    na, c2, c3 = frecuencia_continuidad(presencia, row["cod_cliente"], row["ejercicio"])
    n_anos_list.append(na)
    cont2_list.append(c2)
    cont3_list.append(c3)

base["n_anos"]  = n_anos_list
base["cont2"]   = cont2_list
base["cont3"]   = cont3_list

print("\nDistribución de años de relación:")
print(base.groupby("ejercicio")["n_anos"].describe().round(1).to_string())

print("\n% clientes con continuidad 2 años:")
print((base.groupby("ejercicio")["cont2"].mean() * 100).round(1))

# ============================================================
# %% [markdown]
# ## 7. Features del maestro cliente
# ============================================================
# El Maestro ya tiene las columnas del maestro de clientes
# fusionadas. Las extraemos tomando el último registro
# conocido de cada cliente.

# %%
COLS_MAESTRO = [
    "cod_cliente", "nombre_comercial", "razon_social",
    "fecha_alta", "fecha_baja",
    "provincia", "municipio", "codigo_postal", "cod_pais",
    "id_sector_principal", "id_sector_secundario",
    "id_actividad", "id_industria",
    "Sector_1", "Sector_2", "Sector_3", "Sector_4",
    "es_cliente_activo", "es_exportacion",
]

# Un registro por cliente (el más reciente)
maestro = (
    df.sort_values("ejercicio", ascending=False)
    .drop_duplicates(subset="cod_cliente")
    [COLS_MAESTRO]
    .copy()
)

print(f"Clientes en maestro: {len(maestro):,}")
print("\nNulos en columnas clave:")
for c in ["Sector_3","provincia","cod_pais","fecha_alta","es_exportacion"]:
    n = maestro[c].isna().sum()
    pct = n / len(maestro) * 100
    print(f"  {c:<25}: {n:>5,}  ({pct:.1f}%)")

# %%
# Antigüedad en años desde fecha_alta hasta año de referencia
# (la calculamos después de hacer el merge)

maestro["fecha_alta_dt"] = pd.to_datetime(maestro["fecha_alta"], errors="coerce")
maestro["fecha_baja_dt"] = pd.to_datetime(maestro["fecha_baja"],  errors="coerce")

# ¿Ha causado baja?
maestro["ha_causado_baja"] = maestro["fecha_baja_dt"].notna().astype(int)

# Exportación
maestro["es_exportacion"] = pd.to_numeric(maestro["es_exportacion"], errors="coerce").fillna(0).astype(int)
maestro["es_nacional"]    = (maestro["cod_pais"].fillna("ES").str.strip().str.upper() == "ES").astype(int)

# Sector limpio (usamos Sector_3 como más granular)
maestro["sector"] = maestro["Sector_3"].fillna(maestro["Sector_2"]).fillna(maestro["Sector_1"]).fillna("Desconocido")

print("\nTop 10 sectores:")
print(maestro["sector"].value_counts().head(10))

print("\nDistribución exportación vs nacional:")
print(maestro["es_exportacion"].value_counts().rename({0: "Nacional", 1: "Exportación"}))

# ============================================================
# %% [markdown]
# ## 8. Merge: base transaccional + maestro cliente
# ============================================================

# %%
dataset = base.merge(
    maestro[[
        "cod_cliente", "nombre_comercial",
        "fecha_alta_dt", "ha_causado_baja",
        "provincia", "cod_pais",
        "sector",
        "es_exportacion", "es_nacional",
        "id_sector_principal", "id_actividad",
    ]],
    on="cod_cliente", how="left"
)

# Antigüedad dinámica: años desde fecha_alta hasta el año del registro
def antiguedad_anos(row):
    if pd.isna(row["fecha_alta_dt"]):
        return np.nan
    try:
        ref = datetime(int(row["ejercicio"]), 12, 31)
        return (ref - row["fecha_alta_dt"].to_pydatetime()).days / 365.25
    except Exception:
        return np.nan

dataset["antiguedad"] = dataset.apply(antiguedad_anos, axis=1)

print(f"Dataset combinado: {len(dataset):,} filas x {len(dataset.columns)} columnas")
print(f"\nVistazo:")
print(dataset[["cod_cliente","nombre_comercial","ejercicio","fac","tn","pvp",
               "n_anos","cont3","n_skus","sector","antiguedad"]].head(10).to_string(index=False))

# ============================================================
# %% [markdown]
# ## 9. Variable target: ¿compró el año siguiente?
# ============================================================

# %%
# Para cada (cliente, año T), ¿aparece en el año T+1?
clientes_por_ano = {
    y: set(df[df["ejercicio"] == y]["cod_cliente"])
    for y in ANOS
}

def compra_next(row):
    next_year = row["ejercicio"] + 1
    if next_year not in clientes_por_ano:
        return np.nan   # 2026 → 2027, aún no sabemos
    return int(row["cod_cliente"] in clientes_por_ano[next_year])

dataset["compra_next"] = dataset.apply(compra_next, axis=1)

print("Distribución de compra_next:")
print(dataset["compra_next"].value_counts(dropna=False))
print(f"\nTasa por año:")
print(dataset.groupby("ejercicio")["compra_next"]
      .mean().dropna().map(lambda v: f"{v:.1%}"))

# ============================================================
# %% [markdown]
# ## 10. Estadísticas descriptivas del dataset final
# ============================================================

# %%
FEATURE_COLS = [
    "fac","fac_ant","tn","tn_ant","pvp","pvp_ant",
    "trend_fac_pct","trend_tn_pct","delta_pvp",
    "n_pedidos","n_albaranes","n_lineas",
    "n_skus","n_familias","n_subfamilias",
    "conc_top1",
    "n_anos","cont2","cont3",
    "antiguedad","es_exportacion","es_nacional",
]

print("Estadísticas de features (todos los años 2021-2026):")
print(dataset[FEATURE_COLS].describe().round(2).to_string())

# %%
# Distribución por año
print("\nFacturación media por año (M€):")
print((dataset.groupby("ejercicio")["fac"].mean() / 1e6).round(3))

print("\nNº SKUs medio por año:")
print(dataset.groupby("ejercicio")["n_skus"].mean().round(1))

print("\nAntigüedad media por año (años):")
print(dataset.groupby("ejercicio")["antiguedad"].mean().round(1))

# %%
# Chequeo de nulls en el dataset final
print("\nNulos por columna:")
nulls = dataset[FEATURE_COLS].isna().sum()
pcts  = (nulls / len(dataset) * 100).round(1)
for col, n, p in zip(nulls.index, nulls.values, pcts.values):
    if n > 0:
        print(f"  {col:<25}: {n:>6,}  ({p:.1f}%)")
print("  (el resto: 0 nulls)")

# ============================================================
# %% [markdown]
# ## 11. Subset: clientes activos 2026 (universo del modelo)
# ============================================================

# %%
clientes_2026 = clientes_por_ano[ANO_ACTUAL]

dataset_2026 = dataset[
    (dataset["ejercicio"] == ANO_ACTUAL) &
    (dataset["cod_cliente"].isin(clientes_2026))
].copy()

print(f"Clientes activos 2026: {len(dataset_2026)}")
print(f"\nRecurrencia (¿cuántos también compraron en años anteriores?):")
for y in [2025, 2024, 2023, 2022, 2021]:
    n = len(dataset_2026[dataset_2026["cod_cliente"].isin(clientes_por_ano[y])])
    pct = n / len(dataset_2026) * 100
    print(f"  {ANO_ACTUAL} ∩ {y}: {n:>4} clientes  ({pct:.1f}%)")

# %%
print("\nPerfil de los 432 clientes activos 2026:")
print(dataset_2026[["fac","tn","pvp","n_anos","n_skus","conc_top1","antiguedad"]].describe().round(2).to_string())

print("\nDistribución por sector:")
print(dataset_2026["sector"].value_counts().head(15))

print("\nDistribución por provincia (top 10):")
print(dataset_2026["provincia"].value_counts().head(10))

print("\nNacionales vs exportación:")
print(dataset_2026["es_exportacion"].value_counts().rename({0:"Nacional",1:"Exportación"}))

# ============================================================
# %% [markdown]
# ## 12. Exportar dataset
# ============================================================

# %%
COLS_EXPORT = [
    # Identificadores
    "cod_cliente", "nombre_comercial", "ejercicio",
    # Transaccional
    "fac", "fac_ant", "tn", "tn_ant", "pvp", "pvp_ant",
    "trend_fac_pct", "trend_tn_pct", "delta_pvp",
    # Cartera
    "n_pedidos", "n_albaranes", "n_skus", "n_familias",
    "n_subfamilias", "conc_top1",
    # RFM
    "n_anos", "cont2", "cont3",
    # Maestro cliente
    "antiguedad", "es_exportacion", "es_nacional",
    "provincia", "cod_pais", "sector",
    "id_sector_principal", "id_actividad",
    # Target
    "compra_next",
]

dataset_out = dataset[COLS_EXPORT].reset_index(drop=True)

OUTPUT_PATH = "data/dataset_modelo1.csv"
dataset_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"Dataset guardado: {OUTPUT_PATH}")
print(f"  Filas:    {len(dataset_out):,}")
print(f"  Columnas: {len(dataset_out.columns)}")
print(f"\nDistribución final:")
print(dataset_out.groupby("ejercicio").agg(
    clientes=("cod_cliente","nunique"),
    con_target=("compra_next","count"),
    tasa_compra=("compra_next","mean"),
).round(3).to_string())
