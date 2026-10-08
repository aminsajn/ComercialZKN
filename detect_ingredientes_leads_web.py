"""
detect_ingredientes_leads_web.py
Detecta ingredientes Zukán en los productos scrapeados de webs corporativas.

Proceso:
  1. Lee leads_productos.csv (productos scrapeados por scraper_leads_productos.py)
  2. Construye un matcher con los sinónimos de zukan_sinonimos_ing.csv
  3. Para cada lead, busca qué sinónimos aparecen en su texto de productos
  4. Genera leads_web_ingredientes.csv con el mismo formato que leads_fuentes_externas.csv
     (compatible con build_lead_similarity.py)

Salida: data/leads_web_ingredientes.csv
"""

import re
import unicodedata
import pandas as pd
import numpy as np

DATA = "data/"

# ── 1. Cargar maestro de ingredientes ─────────────────────────────────────────

print("Cargando maestro de ingredientes...")
ing_df = pd.read_csv(DATA + "zukan_ingredientes.csv", encoding="utf-8")
ing_df = ing_df[ing_df["id_ingrediente"].notna() & ing_df["ingrediente"].notna()].copy()
ID_TO_NAME = dict(zip(ing_df["id_ingrediente"], ing_df["ingrediente"]))
NAME_TO_ID = dict(zip(ing_df["ingrediente"], ing_df["id_ingrediente"]))
print(f"  {len(ID_TO_NAME)} ingredientes.")

# ── 2. Construir matcher de sinónimos ─────────────────────────────────────────

print("Cargando sinónimos...")
sin_df = pd.read_csv(DATA + "zukan_sinonimos_ing.csv", encoding="utf-8", on_bad_lines="skip")
# Añadir los nombres canónicos también como "sinónimos"
canon_rows = pd.DataFrame({
    "id_ingrediente": list(NAME_TO_ID.values()),
    "sinonimo":       list(NAME_TO_ID.keys()),
})
sin_df = pd.concat([sin_df, canon_rows], ignore_index=True)
sin_df = sin_df.dropna(subset=["id_ingrediente", "sinonimo"]).copy()
sin_df["sinonimo"] = sin_df["sinonimo"].str.strip()
print(f"  {len(sin_df)} sinónimos + nombres canónicos.")


def _norm(text: str) -> str:
    """Normaliza texto: minúsculas, sin acentos, sin puntuación extra."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Construir lista ordenada por longitud descendente (más específicos primero)
SYNONYMS: list[tuple[str, str]] = []  # (patron_normalizado, id_ingrediente)
for _, row in sin_df.iterrows():
    pat = _norm(str(row["sinonimo"]))
    if len(pat) >= 3:
        SYNONYMS.append((pat, row["id_ingrediente"]))

SYNONYMS.sort(key=lambda x: -len(x[0]))  # más largo primero


def detect_ingredientes(productos_text: str) -> list[str]:
    """Devuelve lista de nombres canónicos de ingredientes detectados."""
    if not productos_text or (isinstance(productos_text, float) and np.isnan(productos_text)):
        return []

    text_norm = _norm(str(productos_text))
    found_ids = set()

    for pat, ing_id in SYNONYMS:
        # Búsqueda de palabra completa (word boundary)
        pattern = r"(?<!\w)" + re.escape(pat) + r"(?!\w)"
        if re.search(pattern, text_norm):
            found_ids.add(ing_id)

    return sorted(ID_TO_NAME[i] for i in found_ids if i in ID_TO_NAME)


# ── 3. Procesar leads_productos.csv ──────────────────────────────────────────

print("\nProcesando leads_productos.csv...")
leads = pd.read_csv(DATA + "leads_productos.csv", encoding="utf-8-sig", low_memory=False)
leads_ok = leads[leads["n_productos"] > 0].copy()
print(f"  Leads con productos: {len(leads_ok)} / {len(leads)}")

leads_ok["ingredientes_detectados"] = leads_ok["productos"].apply(detect_ingredientes)
leads_ok["n_ingredientes"] = leads_ok["ingredientes_detectados"].apply(len)
leads_ok["ingredientes_zukan_detectados"] = leads_ok["ingredientes_detectados"].apply(
    lambda x: ", ".join(x)
)

con_ing = leads_ok[leads_ok["n_ingredientes"] > 0]
print(f"  Con 1+ ingrediente Zukan detectado: {len(con_ing)}")
print(f"  Sin ingredientes detectados:        {len(leads_ok) - len(con_ing)}")

if len(con_ing) > 0:
    print(f"\n  Top ingredientes detectados:")
    from collections import Counter
    cnt = Counter()
    for ings in con_ing["ingredientes_detectados"]:
        cnt.update(ings)
    for ing, n in cnt.most_common(10):
        print(f"    {ing}: {n} leads")

# ── 4. Guardar resultado compatible con build_lead_similarity ─────────────────

out = leads_ok[[
    "leadid", "companyname", "sector", "websiteurl",
    "n_productos", "ingredientes_zukan_detectados", "n_ingredientes",
    "status"
]].copy()
out = out.rename(columns={
    "companyname": "empresa",
    "sector":      "sectores_zukan",
    "websiteurl":  "url_web",
})
out["origen"]                = "web_scraping"
out["es_cliente"]            = "desconocido"
out["n_productos_potenciales"] = out["n_productos"]
out["alimarket_ventas_eur"]  = np.nan
out["alimarket_empleados"]   = np.nan

output_path = DATA + "leads_web_ingredientes.csv"
out.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n  -> {output_path}")
print(f"  Leads exportados: {len(out)}")
print(f"  Con ingredientes: {(out['n_ingredientes']>0).sum()}")

# ── 5. Muestra de resultados ──────────────────────────────────────────────────

print("\n--- Muestra con ingredientes detectados ---")
sample = con_ing[["companyname", "sector", "n_ingredientes",
                   "ingredientes_zukan_detectados"]].head(15)
print(sample.to_string(index=False))

print("\nProceso completado.")
