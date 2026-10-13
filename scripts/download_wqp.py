"""Download a bounded public Water Quality Portal query with provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=["Result", "Station"], default="Result")
    parser.add_argument("--provider", choices=["NWIS", "EPA"])
    parser.add_argument("--characteristic", default="Biochemical oxygen demand, standard conditions")
    parser.add_argument("--start", default="01-01-2017")
    parser.add_argument("--end", default="12-31-2023")
    parser.add_argument("--state")
    parser.add_argument("--site")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("Output exists; choose a new file")
    params = {"characteristicName": args.characteristic, "startDateLo": args.start,
              "startDateHi": args.end, "mimeType": "csv", "sorted": "no", "sampleMedia": "Water", "countrycode": "US"}
    if args.provider:
        params["providers"] = args.provider
    if args.state:
        params["statecode"] = args.state
    if args.site:
        params["siteid"] = args.site
    else:
        params["siteType"] = "Stream"
    url = f"https://www.waterqualitydata.us/data/{args.service}/search"
    print(f"Downloading WQP {args.service}: {args.characteristic}; provider={args.provider or 'all'}", flush=True)
    response = requests.get(url, params=params, timeout=(20, 90), stream=True)
    response.raise_for_status()
    chunks, size = [], 0
    for chunk in response.iter_content(65536):
        size += len(chunk)
        if size > 25 * 1024 * 1024:
            raise ValueError("Query exceeded 25MiB; narrow dates/sites")
        chunks.append(chunk)
    content = b"".join(chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(content)
    metadata = {"request_url": response.url, "query": params, "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type"), "bytes": size,
                "sha256": hashlib.sha256(content).hexdigest(), "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "service_guide": "https://www.waterqualitydata.us/webservices_documentation/",
                "note": "WQP aggregates USGS NWIS and other agencies; keep originating organization, units, qualifiers and methods."}
    out.with_suffix(".download.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {size} bytes to {out}")


if __name__ == "__main__":
    main()
