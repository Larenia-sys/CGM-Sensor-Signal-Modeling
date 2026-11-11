"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/3hbcscwz44"
UA = {"User-Agent": "Mozilla/5.0 (research; cgm-sensor-project)"}


def fetch_json(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as exc:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {exc}")
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def main(dst: str = "data/raw") -> None:
    meta = fetch_json(API)
    files = [f for f in meta["files"] if "free_style_sensor" in f["filename"]]
    print(f"共 {len(files)} 个 CGM 传感器文件，"
          f"许可 {meta.get('data_licence', '见数据集页面')}")

    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        path = out / f["filename"]
        if path.exists() and path.stat().st_size == f["size"]:
            continue
        url = f["content_details"]["download_url"]
        req = urllib.request.Request(url, headers=UA)
        data = urllib.request.urlopen(req, timeout=120).read()
        path.write_bytes(data)
        print(f"  {f['filename']}  {f['size'] / 1e6:.2f} MB")
    print("完成")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")"""