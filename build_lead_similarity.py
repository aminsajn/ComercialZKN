"""
build_lead_similarity.py
Calcula similitud Jaccard entre leads (fuentes externas MKI) y clientes Zukán
basándose en qué ingredientes Zukán compran/usan.

Fuente leads:   data/leads_fuentes_externas.csv   (campo ingredientes_zukan_detectados)
Fuente clientes: data/Maestro_Análisis_Comercial.csv (mapeo producto -> ingrediente)

Outputs:
  data/leads_ingredientes.csv          -- lead -> set de ingredientes Zukán detectados
  data/clientes_ingredientes.csv       -- cliente -> set de ingredientes Zukán comprados
  data/productos_sin_mapear.csv        -- productos Zukán que no se pudieron mapear
  data/lead_similitud.csv              -- todos los pares lead x cliente con Jaccard > 0
  data/lead_similitud_resumen.csv      -- resumen por lead: mejor cliente, sector, score
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

# --- 1. Ingredientes maestro --------------------------------------------------

print("Cargando maestro de ingredientes...")
ing_df = pd.read_csv(DATA / "zukan_ingredientes.csv", encoding="utf-8")
ing_df = ing_df[ing_df["id_ingrediente"].notna() & ing_df["ingrediente"].notna()].copy()

ID = {row["ingrediente"]: row["id_ingrediente"] for _, row in ing_df.iterrows()}
NOMBRE = {v: k for k, v in ID.items()}  # id -> nombre

# IDs por nombre corto (para uso en el código)
AZ       = ID["Azúcar"]
AZ_LIQ   = ID["Azúcar Líquido"]
AZ_INV   = ID["Azúcar Líquido Invertido"]
AZ_INT   = ID["Azúcar Integral"]
DEX      = ID["Dextrosa"]
FRU      = ID["Fructosa"]
FRU_SYR  = ID["Jarabe de fructosa"]
GLU_SYR  = ID["Jarabe de glucosa"]
GLU_FRU  = ID["Jarabe de glucosa y fructosa"]
MAL_LIQ  = ID["Jarabe de maltitol"]
SOR_LIQ  = ID["Jarabe de sorbitol"]
MAL      = ID["Maltitol"]
SOR      = ID["Sorbitol"]
MEL      = ID["Melaza de caña"]
FOS      = ID["Fructooligosacaridos"]
STE      = ID["Glucósidos de esteviol (E-960a)"]
MIEL     = ID["Miel pura de abejas"]
CAR      = ID["Caramelo natural"]

print(f"  {len(ID)} ingredientes cargados.")

# --- 2. Mapeo producto Zukán -> ingredientes -----------------------------------
# Reglas deterministas basadas en la nomenclatura de Zukán.
# Un producto puede contener varios ingredientes (ej: Mixco Sweet = azúcar + jarabe glucosa).

def mapear_producto_zukan(descripcion: str) -> frozenset:
    """Devuelve frozenset de id_ingrediente para un producto Zukán."""
    if not descripcion or pd.isna(descripcion):
        return frozenset()

    d = str(descripcion).upper()
    ids = set()

    # -- Ingredientes puros no azúcar -----------------------------------------

    # FOS / Fructooligosacáridos
    if "FOS" in d or "FRUCTOOLIGOSAC" in d:
        ids.add(FOS)

    # Dextrosa
    if "DEXTROSA" in d:
        ids.add(DEX)

    # Fructosa cristal (≠ Fructor que es jarabe)
    if "FRUCTOSA" in d and "FRUCTOR" not in d:
        ids.add(FRU)

    # Sorbitol líquido -> Jarabe de sorbitol
    if "SORBITOL" in d:
        if any(x in d for x in ["LIQUIDO", "LÍQUIDO", "70%", "LIQUID"]):
            ids.add(SOR_LIQ)
        elif any(x in d for x in ["POLVO", "EN POLVO", "POWDER"]):
            ids.add(SOR)
        else:
            ids.add(SOR)  # default sólido si no hay calificador

    # Maltitol líquido -> Jarabe de maltitol
    if "MALTITOL" in d:
        if any(x in d for x in ["LIQUIDO", "LÍQUIDO", "LIQUID", "55/72", "55/75"]):
            ids.add(MAL_LIQ)
        elif any(x in d for x in ["POLVO", "EN POLVO", "POWDER"]):
            ids.add(MAL)
        else:
            ids.add(MAL)  # default sólido

    # Maltor = Jarabe de maltitol (producto comercial)
    if "MALTOR" in d:
        ids.add(MAL_LIQ)
        ids.discard(MAL)  # Maltor es siempre líquido

    # Melaza
    if "MELAZA" in d:
        ids.add(MEL)

    # Caramelo
    if "CARAMELO" in d or "CARAMEL" in d:
        ids.add(CAR)

    # -- Jarabes de glucosa/fructosa -------------------------------------------

    # Fructor = Jarabe de glucosa y fructosa (isoglucosa)
    if "FRUCTOR" in d:
        ids.add(GLU_FRU)

    # Glucor = Jarabe de glucosa puro
    if "GLUCOR" in d:
        ids.add(GLU_SYR)

    # Fosvitae / Fosfruit = productos a base de FOS (algunos llevan fructosa)
    if "FOSVITAE" in d or "FOSFRUIT" in d:
        ids.add(FOS)
        # Fosvitae es básicamente FOS + fructosa líquida
        ids.add(FRU_SYR)

    # -- Azúcares --------------------------------------------------------------

    # Azúcar Invertido -> Azúcar Líquido Invertido
    if any(x in d for x in ["INVERTIDO", "INVERTIDA", "INVERT",
                              "TIEN ", "COMPOTIENZ", "LEMONOX"]):
        ids.add(AZ_INV)

    # Azúcar Líquida Neutra (≠ invertido)
    if any(x in d for x in ["AZÚCAR LÍQUIDA", "AZÚCAR LIQUIDA",
                              "AZ.LÍQ", "AZ LÍQ", "ATI 6", "ATI 7",
                              "A. LÍQ.", "A. LIQ."]):
        if AZ_INV not in ids:
            ids.add(AZ_LIQ)

    # Azúcar Integral / Moreno / Raw / Panela
    if any(x in d for x in ["INTEGRAL", "MORENO", "MORENA", "PANELA",
                              ">150 ICUMSA", ">2500", ">3500", ">4000",
                              "2000-3000", "2000-3500", "3000-3500",
                              "BROWN", "RAW SUGAR", "DC RAW", "DEMERARA"]):
        ids.add(AZ_INT)

    # Azúcar blanco / granulado / extrafino / ICUMSA (claros)
    # Solo añadir Azúcar si no ya clasificado como Líquida/Invertida/Integral
    if any(x in d for x in ["AZÚCAR", "AZUCAR", "CORREFINO",
                              "ICUMSA", "EEC N", "AZ BL", "AZ. BL"]):
        if not {AZ_LIQ, AZ_INV, AZ_INT} & ids:
            ids.add(AZ)

    # -- Mezclas / blends ------------------------------------------------------

    # Mixco Sweet = mezcla de azúcar + jarabe de glucosa (composición estándar)
    # "Mixco Sweet 65-00.5050" -> 50% jarabe glucosa + 50% azúcar blanco
    # "Mixco Sweet 70-80.1010 Caramelo" -> incluye caramelo
    if "MIXCO SWEET" in d or "MIXCO" in d:
        ids.add(AZ)
        ids.add(GLU_SYR)
        if "CARAMELO" in d or "CARAMEL" in d:
            ids.add(CAR)
        if "FRUCTOSA" in d or "FRUCTOR" in d:
            ids.add(GLU_FRU)

    # Mix Maltitol Sorbitol -> ambos
    if "MIX MALTITOL" in d and "SORBITOL" in d:
        ids.add(MAL_LIQ)
        ids.add(SOR_LIQ)

    # Mix Maltitol Stevia
    if "MIX MALTITOL" in d and "STEVIA" in d:
        ids.add(MAL_LIQ)
        ids.add(STE)

    # -- Apicultura ------------------------------------------------------------
    # Apimix, Apipasta, Fondant, PEP58 = productos compuestos para abejas
    # Contienen principalmente azúcar + jarabe de glucosa/fructosa
    if any(x in d for x in ["APIMIX", "APIPASTA", "FONDANT",
                              "PEP58", "BEESUCRE", "PIAPI", "MIX ABEJORROS",
                              "MIX FRUCTOR APIC", "MIX GOLDEN JARABE",
                              "CANE GOLDEN SYRUP"]):
        ids.add(AZ)
        ids.add(GLU_FRU)  # Fructor es base de la mayoría

    # Jarabe de Azucar Invertido BIO
    if "JARABE DE AZUCAR INVERTIDO" in d or "JARABE AZUCAR INVERTIDO" in d:
        ids.add(AZ_INV)

    # Jarabe de Melaza
    if "JARABE DE MELAZA" in d or "JARABE MELAZA" in d:
        ids.add(MEL)

    # Mix Fru-Dex = Fructosa + Dextrosa
    if "MIX FRU-DEX" in d or "FRUDEX" in d:
        ids.add(FRU)
        ids.add(DEX)

    # Almidón de maíz -> NO es un ingrediente Zukán de los 19
    # Harina de levadura -> NO

    return frozenset(ids)


# --- 3. Procesar leads fuentes externas --------------------------------------

print("\nProcesando leads de fuentes externas...")
leads_df = pd.read_csv(DATA / "leads_fuentes_externas.csv", encoding="utf-8")
print(f"  {len(leads_df)} empresas cargadas.")
print(f"  Estado: {leads_df['es_cliente'].value_counts().to_dict()}")

# Solo nos interesan los que NO son clientes todavía
leads_no_cli = leads_df[leads_df["es_cliente"].isin(["no_cliente", "desconocido"])].copy()
print(f"  No-clientes o desconocidos: {len(leads_no_cli)}")

# Parsear ingredientes_zukan_detectados (nombre1, nombre2, ...)
def parsear_ingredientes(texto: str) -> frozenset:
    if not texto or pd.isna(texto):
        return frozenset()
    nombres = [x.strip() for x in str(texto).split(",") if x.strip()]
    ids = frozenset(ID[n] for n in nombres if n in ID)
    # Nombres no mapeados
    sin_mapear = [n for n in nombres if n not in ID]
    if sin_mapear:
        pass  # se reportan al final
    return ids

leads_no_cli["ing_ids"] = leads_no_cli["ingredientes_zukan_detectados"].apply(parsear_ingredientes)
leads_no_cli["n_ingredientes"] = leads_no_cli["ing_ids"].apply(len)
leads_no_cli["ingredientes"] = leads_no_cli["ing_ids"].apply(
    lambda s: ", ".join(sorted(NOMBRE.get(i, i) for i in s))
)

# Leads con al menos 1 ingrediente mapeado
leads_activos = leads_no_cli[leads_no_cli["n_ingredientes"] > 0].copy()
print(f"  Con >=1 ingrediente detectado: {len(leads_activos)}")

# Guardar
leads_out = leads_activos[["empresa", "origen", "es_cliente", "sectores_zukan",
                            "n_productos_potenciales", "ingredientes",
                            "alimarket_ventas_eur", "alimarket_empleados"]].copy()
leads_out.to_csv(DATA / "leads_ingredientes.csv", index=False, encoding="utf-8")
print(f"  -> data/leads_ingredientes.csv")

# --- 4. Procesar clientes Zukán -----------------------------------------------

print("\nProcesando clientes Zukán (últimos 3 años)...")
maestro = pd.read_csv(
    DATA / "Maestro_Análisis_Comercial.csv",
    encoding="utf-8", low_memory=False,
    usecols=["cod_cliente", "nombre_comercial", "Sector_3",
             "cod_producto", "descripcion_producto", "ejercicio",
             "total_base_neto", "Cantidad_TN"]
)
anio_max = int(maestro["ejercicio"].max())
maestro = maestro[maestro["ejercicio"] >= anio_max - 2].copy()
print(f"  Años considerados: {anio_max-2} - {anio_max}")

# Mapear productos -> ingredientes (sin duplicar por cliente)
productos_unicos = maestro.drop_duplicates("cod_producto")[
    ["cod_producto", "descripcion_producto"]
].copy()
productos_unicos["ing_ids"] = productos_unicos["descripcion_producto"].apply(
    mapear_producto_zukan
)
productos_unicos["n_ing"] = productos_unicos["ing_ids"].apply(len)

sin_mapear = productos_unicos[productos_unicos["n_ing"] == 0].copy()
print(f"  Productos únicos: {len(productos_unicos)}")
print(f"  Sin mapear a ingrediente: {len(sin_mapear)} ({len(sin_mapear)/len(productos_unicos)*100:.1f}%)")
sin_mapear[["cod_producto", "descripcion_producto"]].to_csv(
    DATA / "productos_sin_mapear.csv", index=False, encoding="utf-8"
)
print(f"  -> data/productos_sin_mapear.csv")

# Unir mapeo al maestro
maestro = maestro.merge(
    productos_unicos[["cod_producto", "ing_ids"]],
    on="cod_producto", how="left"
)

# Agregar por cliente: unión de ingredientes + facturación
def union_sets(series):
    result = frozenset()
    for s in series:
        if isinstance(s, frozenset):
            result = result | s
    return result

cliente_agg = (
    maestro.groupby(["cod_cliente", "nombre_comercial", "Sector_3"])
    .agg(
        ing_ids=("ing_ids", union_sets),
        facturacion_total=("total_base_neto", "sum"),
        toneladas_total=("Cantidad_TN", "sum"),
    )
    .reset_index()
)
cliente_agg["n_ingredientes"] = cliente_agg["ing_ids"].apply(len)
cliente_agg["ingredientes"] = cliente_agg["ing_ids"].apply(
    lambda s: ", ".join(sorted(NOMBRE.get(i, i) for i in s))
)

clientes_activos = cliente_agg[cliente_agg["n_ingredientes"] > 0].copy()
print(f"\n  Clientes con >=1 ingrediente mapeado: {len(clientes_activos)} / {len(cliente_agg)}")

cliente_agg[["cod_cliente", "nombre_comercial", "Sector_3",
             "n_ingredientes", "ingredientes",
             "facturacion_total", "toneladas_total"]].to_csv(
    DATA / "clientes_ingredientes.csv", index=False, encoding="utf-8"
)
print(f"  -> data/clientes_ingredientes.csv")

# --- 5. Similitud Jaccard -----------------------------------------------------

print(f"\nCalculando Jaccard ({len(leads_activos)} leads x {len(clientes_activos)} clientes)...")
print(f"  Total pares posibles: {len(leads_activos) * len(clientes_activos):,}")

sim_rows = []
for i, (_, lead) in enumerate(leads_activos.iterrows()):
    if i % 100 == 0:
        print(f"  [{i+1}/{len(leads_activos)}]...", end="\r")
    A = lead["ing_ids"]
    for _, cli in clientes_activos.iterrows():
        B = cli["ing_ids"]
        inter = A & B
        if not inter:
            continue
        union = A | B
        jaccard = round(len(inter) / len(union), 4)
        sim_rows.append({
            "empresa_lead": lead["empresa"],
            "origen_lead": lead["origen"],
            "sector_lead": lead["sectores_zukan"],
            "n_productos_lead": lead["n_productos_potenciales"],
            "cod_cliente": cli["cod_cliente"],
            "nombre_cliente": cli["nombre_comercial"],
            "sector_cliente": cli["Sector_3"],
            "jaccard": jaccard,
            "n_comunes": len(inter),
            "ingredientes_comunes": ", ".join(
                sorted(NOMBRE.get(x, x) for x in inter)
            ),
            "ingredientes_lead": lead["ingredientes"],
            "ingredientes_cliente": cli["ingredientes"],
        })

print(f"\n  Pares con Jaccard > 0: {len(sim_rows):,}")

if sim_rows:
    sim_df = pd.DataFrame(sim_rows).sort_values("jaccard", ascending=False)
    sim_df.to_csv(DATA / "lead_similitud.csv", index=False, encoding="utf-8")
    print(f"  -> data/lead_similitud.csv")

    # Resumen por lead
    resumen_rows = []
    for empresa, grp in sim_df.groupby("empresa_lead"):
        best = grp.loc[grp["jaccard"].idxmax()]
        resumen_rows.append({
            "empresa_lead": empresa,
            "origen_lead": best["origen_lead"],
            "sector_lead": best["sector_lead"],
            "n_productos_lead": best["n_productos_lead"],
            "jaccard_max": best["jaccard"],
            "jaccard_medio": round(grp["jaccard"].mean(), 4),
            "n_clientes_similares": grp["cod_cliente"].nunique(),
            "mejor_cliente": best["nombre_cliente"],
            "sector_mejor_cliente": best["sector_cliente"],
            "ingredientes_comunes": best["ingredientes_comunes"],
            "ingredientes_lead": best["ingredientes_lead"],
        })

    resumen_df = pd.DataFrame(resumen_rows).sort_values("jaccard_max", ascending=False)
    resumen_df.to_csv(DATA / "lead_similitud_resumen.csv", index=False, encoding="utf-8")
    print(f"  -> data/lead_similitud_resumen.csv")

    print("\n" + "=" * 80)
    print("RESUMEN ESTADÍSTICO")
    print("=" * 80)
    print(f"\nLeads procesados:            {len(leads_activos):>5}")
    print(f"Leads con algún match:       {resumen_df.shape[0]:>5} ({resumen_df.shape[0]/len(leads_activos)*100:.1f}%)")
    print(f"Jaccard máximo (max):        {sim_df['jaccard'].max():.4f}")
    print(f"Jaccard máximo (mediana):    {resumen_df['jaccard_max'].median():.4f}")
    print(f"Jaccard máximo (media):      {resumen_df['jaccard_max'].mean():.4f}")

    print(f"\n--- Distribución de Jaccard máximo por lead ---")
    bins = [0, 0.1, 0.2, 0.33, 0.5, 0.67, 1.01]
    labels = ["0-10%", "10-20%", "20-33%", "33-50%", "50-67%", "67-100%"]
    resumen_df["jaccard_bin"] = pd.cut(resumen_df["jaccard_max"], bins=bins, labels=labels, right=False)
    print(resumen_df["jaccard_bin"].value_counts().sort_index().to_string())

    print(f"\n--- Top 20 leads más similares ---")
    cols_show = ["empresa_lead", "jaccard_max", "n_clientes_similares",
                 "mejor_cliente", "ingredientes_comunes"]
    print(resumen_df[cols_show].head(20).to_string(index=False))
else:
    print("\n  AVISO: No se encontraron pares con similitud > 0.")
    print("  Revisa data/productos_sin_mapear.csv para ver qué productos no se mapearon.")

print("\nProceso completado.")
