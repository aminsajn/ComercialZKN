"""
retry_scraper.py
Reintenta los leads con status error_home u ok_sin_productos del scraper anterior.
Mejoras respecto al original:
  - Timeout mayor (20s)
  - Reintento con http:// si https:// falla
  - Reintento con delay si el primer intento falla
  - Prueba rutas comunes de catálogo si no hay links de producto detectados
  - Más subpáginas exploradas (5 en vez de 3)
Actualiza data/leads_productos.csv en los registros afectados.
"""

import re
import time
import unicodedata
import warnings
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TIMEOUT        = 20
DELAY          = 1.5
MAX_PROD_PAGES = 5
MAX_PRODUCTS   = 60
OUTPUT_FILE    = "data/leads_productos.csv"
PROGRESS_EVERY = 25

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PRODUCT_URL_KEYWORDS = [
    "product", "producto", "productos", "gama", "catalogo", "catalogue",
    "catalog", "portfolio", "range", "shop", "tienda", "ingredien",
    "soluciones", "solutions", "linea", "lineas", "line", "lines",
    "oferta", "offer", "items", "goods", "collection", "coleccion",
]

SKIP_URL_KEYWORDS = [
    "login", "signin", "signup", "register", "cart", "checkout",
    "privacy", "politica", "legal", "terms", "cookie", "contact",
    "about", "nosotros", "historia", "news", "blog", "press",
    "prensa", "job", "empleo", "career",
]

# Rutas a probar si no se detectan links de producto en la homepage
FALLBACK_PATHS = [
    "/products", "/productos", "/product", "/producto",
    "/gama", "/catalog", "/catalogo", "/portfolio",
    "/range", "/solutions", "/soluciones", "/our-products",
    "/nuestros-productos", "/shop", "/tienda",
]


def normalize_url(raw: str):
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


def fetch(url: str, session: requests.Session):
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=TIMEOUT, headers=HEADERS,
                               allow_redirects=True, verify=False)
            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "")
                if "html" in ctype or not ctype:
                    return BeautifulSoup(resp.text, "lxml")
            return None
        except Exception:
            if attempt == 0:
                time.sleep(3)
    # Reintento con esquema alternativo
    alt = url.replace("https://", "http://", 1) if url.startswith("https://") \
          else url.replace("http://", "https://", 1)
    try:
        resp = session.get(alt, timeout=TIMEOUT, headers=HEADERS,
                           allow_redirects=True, verify=False)
        if resp.status_code == 200:
            ctype = resp.headers.get("Content-Type", "")
            if "html" in ctype or not ctype:
                return BeautifulSoup(resp.text, "lxml")
    except Exception:
        pass
    return None


def extract_products_from_soup(soup: BeautifulSoup) -> list:
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

    for tag in soup.find_all(["h2", "h3"])[:60]:
        add(tag.get_text())

    return products[:MAX_PRODUCTS]


def find_product_links(soup: BeautifulSoup, base_url: str) -> list:
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


def scrape_lead(row: dict, session: requests.Session) -> dict:
    base_url = normalize_url(str(row.get("websiteurl", "") or ""))
    result = {
        "leadid":      row.get("leadid", ""),
        "companyname": row.get("companyname", ""),
        "sector":      row.get("sector", ""),
        "websiteurl":  base_url or row.get("websiteurl", ""),
        "productos":   "",
        "n_productos": 0,
        "paginas_visitadas": "",
        "status":      "no_url",
    }

    if not base_url:
        return result

    soup_home = fetch(base_url, session)
    if soup_home is None:
        result["status"] = "error_home"
        return result

    all_products = []
    pages_visited = [base_url]

    all_products += extract_products_from_soup(soup_home)

    product_links = find_product_links(soup_home, base_url)

    # Si no hay links detectados, probar rutas de catálogo comunes
    if not product_links:
        base_domain = urlparse(base_url).netloc
        base_root = f"{urlparse(base_url).scheme}://{base_domain}"
        for path in FALLBACK_PATHS:
            candidate = base_root + path
            if candidate not in pages_visited:
                time.sleep(DELAY * 0.4)
                soup_fp = fetch(candidate, session)
                if soup_fp:
                    pages_visited.append(candidate)
                    all_products += extract_products_from_soup(soup_fp)
                    if len(all_products) >= 5:
                        break

    for link in product_links:
        time.sleep(DELAY * 0.5)
        soup_sub = fetch(link, session)
        if soup_sub:
            pages_visited.append(link)
            all_products += extract_products_from_soup(soup_sub)

    seen_keys = set()
    unique = []
    for p in all_products:
        k = re.sub(r"\s+", "", p.lower())
        if k not in seen_keys:
            seen_keys.add(k)
            unique.append(p)
        if len(unique) >= MAX_PRODUCTS:
            break

    result["productos"]         = " | ".join(unique)
    result["n_productos"]       = len(unique)
    result["paginas_visitadas"] = " | ".join(pages_visited)
    result["status"]            = "ok" if unique else "ok_sin_productos"
    return result


def main():
    existing = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig", low_memory=False)
    retry_df = existing[existing["status"].isin(["error_home", "ok_sin_productos"])].copy()
    total = len(retry_df)
    print(f"Leads a reintentar: {total} (error_home + ok_sin_productos)")
    print(f"  error_home:       {(retry_df['status']=='error_home').sum()}")
    print(f"  ok_sin_productos: {(retry_df['status']=='ok_sin_productos').sum()}")
    print()

    session = requests.Session()
    session.headers.update(HEADERS)

    updated = 0
    ok = errors = sin_prod = 0

    for i, (idx, row) in enumerate(retry_df.iterrows(), 1):
        result = scrape_lead(row.to_dict(), session)

        # Actualizar fila en el DataFrame original y guardar inmediatamente
        for col, val in result.items():
            existing.at[idx, col] = val
        existing.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        if result["status"] == "ok":
            ok += 1
            updated += 1
        elif result["status"] == "ok_sin_productos":
            sin_prod += 1
        else:
            errors += 1

        if i % PROGRESS_EVERY == 0 or i == total:
            pct = i / total * 100
            name_safe = result["companyname"][:30].encode("ascii", "replace").decode("ascii")
            print(
                f"[{i:3d}/{total}] {pct:5.1f}%  "
                f"ok={ok}  sin_prod={sin_prod}  errores={errors}  "
                f"ultima: {name_safe} -> {result['status']}"
            )
            import sys; sys.stdout.flush()
        time.sleep(DELAY)

    print(f"\nFinalizado. {OUTPUT_FILE} actualizado.")
    print(f"  Leads mejorados (antes sin productos): {updated}")
    print(f"  Siguen sin productos: {sin_prod}")
    print(f"  Siguen con error:     {errors}")


if __name__ == "__main__":
    main()
