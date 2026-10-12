"""Explicit Earth Engine authentication and bounded metadata connection check."""

import argparse
import os


def main() -> None:
    """Check catalog access without downloading imagery or starting login implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("EARTH_ENGINE_PROJECT"),
                        help="Registered Google Cloud project ID; or set EARTH_ENGINE_PROJECT")
    parser.add_argument("--authenticate", action="store_true",
                        help="Start interactive Google account authentication")
    args = parser.parse_args()
    if not args.project or not args.project.strip():
        parser.error("Provide --project with your registered Earth Engine Cloud project ID")
    try:
        import ee
    except ImportError:
        parser.exit(1, "Install requirements-earthengine.txt using this Python environment first.\n")
    try:
        if args.authenticate:
            ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=args.project.strip())
        collection_id = "COPERNICUS/S2_SR_HARMONIZED"
        count = ee.ImageCollection(collection_id).limit(1).size().getInfo()
        if count != 1:
            parser.exit(1, "Collection accessible but no image found; catalog check incomplete.\n")
    except Exception:
        # Avoid exposing credential-bearing exception text in console logs.
        parser.exit(1, "Earth Engine check failed. Verify login, project registration, API enablement, and permissions. Use --authenticate if login is needed.\n")
    print(f"Earth Engine connection successful: {collection_id}")
    print("Bounded catalog check only; no imagery downloaded or scientific observations generated.")


if __name__ == "__main__":
    main()
