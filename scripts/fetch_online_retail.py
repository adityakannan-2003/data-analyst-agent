"""Download the UCI Online Retail dataset and save it as data/online_retail.csv.

    uv run python scripts/fetch_online_retail.py

Real transactions from a UK online gift shop, 1 December 2010 to 9 December 2011
(541,909 rows). Source: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5BW33. License: CC BY 4.0.

The CSV is about 45 MB, so it isn't stored in the repo. The data is saved unchanged:
cleaning it is the agent's job.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SHA256 = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"  # of the zip; checked so the eval's answer key always sees the same data
OUT = Path(__file__).resolve().parent.parent / "data" / "online_retail.csv"


def main() -> None:
    if OUT.exists():
        print(f"{OUT.name} is already there.")
        return
    print(f"Downloading {URL} (about 23 MB)...")
    with urllib.request.urlopen(URL, timeout=300) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if SHA256 and digest != SHA256:
        sys.exit(f"The download doesn't match the expected file (sha256 {digest}).")
    print("Converting from Excel to CSV (takes a minute)...")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        workbook = io.BytesIO(archive.read("Online Retail.xlsx"))
    # Invoice numbers like "C536379" and stock codes like "85123A" must stay text.
    df = pd.read_excel(workbook, dtype={"InvoiceNo": str, "StockCode": str})
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT.relative_to(OUT.parent.parent)} (sha256 of the zip: {digest})")


if __name__ == "__main__":
    main()
