"""
Scraper de productos por lead.
Lee Maestro Leads.csv, visita la web de cada lead y extrae su catálogo de productos.
Salida: data/leads_productos.csv
"""

import re
import time
import csv
import unicodedata
import warnings
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Configuración ─────────────────────────────────────────────────────────────
TIMEOUT        = 12          # segundos por petición
DELAY          = 1.2         # segundos entre peticiones (cortesía)
MAX_PROD_PAGES = 3           # máx. subpáginas de producto a explorar por lead
MAX_PRODUCTS   = 60          # máx. productos a guardar por lead
OUTPUT_FILE    = "data/leads_productos.csv"
PROGRESS_EVERY = 25          # imprimir progreso cada N leads

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Palabras en la URL que sugieren una página de productos
PRODUCT_URL_KEYWORDS = [
    "product", "producto", "productos", "gama", "catalogo", "catalogue",
    "catalog", "portfolio", "range", "shop", "tienda", "ingredien",
    "soluciones", "solutions", "linea", "lineas", "line", "lines",
    "oferta", "offer", "items", "goods", "collection", "coleccion",
]

# Palabras que indican páginas a ignorar
SKIP_URL_KEYWORDS = [
    "login", "signin", "signup", "register", "cart", "checkout",
    "privacy", "politica", "legal", "terms", "cookie", "contact",
    "about", "nosotros", "historia", "news", "blog", "press",
    "prensa", "job", "empleo", "career",
]


# ── Utilidades ────────────────────────────────────────────────────────────────

def normalize_url(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    try:
        p = urlparse(raw)
        return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/") or None
    except Exception:
        return None


def is_product_url(href: str) -> bool:
    low = href.lower()
    if any(k in low for k in SKIP_URL_KEYWORDS):
        return False
    return any(k in low for k in PRODUCT_URL_KEYWORDS)


def fetch(url: str, session: requests.Session) -> BeautifulSoup | None:
    try:
        resp = session.get(url, timeout=TIMEOUT, headers=HEADERS,
                           allow_redirects=True, verify=False)
        if resp.status_code == 200:
            ctype = resp.headers.get("Content-Type", "")
            if "html" in ctype or not ctype:
                return BeautifulSoup(resp.text, "lxml")
        return None
    except Exception:
        return None


def extract_products_from_soup(soup: BeautifulSoup) -> list[str]:
    """
    Estrategia por capas:
    1. Atributos semánticos típicos de e-commerce / catálogos
    2. Headings dentro de secciones de productos
    3. Elementos <li> / <td> en listas de producto
    """
    products = []
    seen = set()

    def add(text: str):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3 or len(text) > 120:
            return
        key = unicodedata.normalize("NFD", text.lower())
        key = "".join(c for c in key if unicodedata.category(c) != "Mn")
        key = re.sub(r"[^\w\s]", "", key).strip()
        if key and key not in seen:
            seen.add(key)
            products.append(text)

    # Capa 1: clases/atributos semánticos comunes
    selectors = [
        "[class*='product-title']", "[class*='product-name']",
        "[class*='product_title']", "[class*='product_name']",
        "[class*='item-title']", "[class*='item-name']",
        "[class*='card-title']", "[class*='card-name']",
        "[itemprop='name']", "[data-product-name]",
        "h2.product", "h3.product", ".product h2", ".product h3",
        ".producto h2", ".producto h3",
    ]
    for sel in selectors:
        for tag in soup.select(sel)[:MAX_PRODUCTS]:
            add(tag.get_text())

    if len(products) >= 5:
        return products[:MAX_PRODUCTS]

    # Capa 2: headings en zonas de catálogo
    for section in soup.find_all(
        ["section", "div", "article", "ul"],
        class_=re.compile(r"product|catalog|gama|range|portfolio|shop|tienda", re.I),
    ):
        for tag in section.find_all(["h2", "h3", "h4", "strong", "a"])[:30]:
            text = tag.get_text()
            if 3 < len(text.strip()) < 100:
                add(text)

    if len(products) >= 5:
        return products[:MAX_PRODUCTS]

    # Capa 3: headings generales de la página (h2/h3) — señal más débil
    for tag in soup.find_all(["h2", "h3"])[:60]:
        add(tag.get_text())

    return products[:MAX_PRODUCTS]


def find_product_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Devuelve hasta MAX_PROD_PAGES URLs que parecen secciones de producto."""
    links = []
    seen = set()
    base_domain = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        parsed = urlparse(full)

        if parsed.netloc != base_domain:
            continue
        if full in seen or full == base_url:
            continue
        if is_product_url(parsed.path + parsed.query):
            seen.add(full)
            links.append(full)
        if len(links) >= MAX_PROD_PAGES * 3:
            break

    return links[:MAX_PROD_PAGES]


# ── Pipeline principal ────────────────────────────────────────────────────────

def scrape_lead(row, session: requests.Session) -> dict:
    base_url = normalize_url(str(row.get("websiteurl", "") or ""))
    result = {
        "leadid":      row.get("leadid", ""),
        "companyname": row.get("companyname", ""),
        "sector":      row.get("bit_sectorprincipalname", ""),
        "websiteurl":  base_url or row.get("websiteurl", ""),
        "productos":   "",
        "n_productos": 0,
        "paginas_visitadas": "",
        "status":      "no_url",
    }

    if not base_url:
        return result

    # 1. Homepage
    soup_home = fetch(base_url, session)
    if soup_home is None:
        result["status"] = "error_home"
        return result

    all_products = []
    pages_visited = [base_url]

    # Productos en homepage
    all_products += extract_products_from_soup(soup_home)

    # 2. Subpáginas de producto
    product_links = find_product_links(soup_home, base_url)
    for link in product_links:
        time.sleep(DELAY * 0.5)
        soup_sub = fetch(link, session)
        if soup_sub:
            pages_visited.append(link)
            all_products += extract_products_from_soup(soup_sub)

    # Deduplicar manteniendo orden
    seen_keys = set()
    unique = []
    for p in all_products:
        k = re.sub(r"\s+", "", p.lower())
        if k not in seen_keys:
            seen_keys.add(k)
            unique.append(p)
        if len(unique) >= MAX_PRODUCTS:
            break

    result["productos"]          = " | ".join(unique)
    result["n_productos"]        = len(unique)
    result["paginas_visitadas"]  = " | ".join(pages_visited)
    result["status"]             = "ok" if unique else "ok_sin_productos"
    return result


def main():
    leads = pd.read_csv(
        "data/Maestro Leads.csv", encoding="utf-8", low_memory=False
    )
    excl = {"No Longer Interested", "Canceled", "Cannot Contact"}
    leads = leads[~leads["statuscodename"].isin(excl)].copy()

    # Solo leads con URL
    leads_url = leads[
        leads["websiteurl"].notna() & (leads["websiteurl"].str.strip() != "")
    ].copy()
    total = len(leads_url)
    print(f"Leads con URL: {total}. Iniciando scraping...\n")

    fieldnames = [
        "leadid", "companyname", "sector", "websiteurl",
        "productos", "n_productos", "paginas_visitadas", "status",
    ]

    session = requests.Session()
    session.headers.update(HEADERS)

    ok = errors = sin_prod = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (_, row) in enumerate(leads_url.iterrows(), 1):
            result = scrape_lead(row.to_dict(), session)

            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "ok_sin_productos":
                sin_prod += 1
            else:
                errors += 1

            writer.writerow(result)
            f.flush()

            if i % PROGRESS_EVERY == 0 or i == total:
                pct = i / total * 100
                name_safe = result['companyname'][:30].encode('ascii', 'replace').decode('ascii')
                print(
                    f"[{i:4d}/{total}] {pct:5.1f}%  "
                    f"ok={ok}  sin_prod={sin_prod}  errores={errors}  "
                    f"ultima: {name_safe} -> {result['status']}"
                )
            time.sleep(DELAY)

    print(f"\nFinalizado. Resultados en {OUTPUT_FILE}")
    print(f"  Con productos:     {ok}")
    print(f"  Sin productos:     {sin_prod}")
    print(f"  Errores/bloqueos:  {errors}")


if __name__ == "__main__":
    main()
