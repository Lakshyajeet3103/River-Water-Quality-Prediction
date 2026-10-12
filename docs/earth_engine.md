# First-time Earth Engine setup

The optional connection checker accesses `COPERNICUS/S2_SR_HARMONIZED` and queries
at most one image's existence. It does not extract spectra or download scenes.
Google sign-in, account consent, and project registration require your participation.

1. Sign in at https://code.earthengine.google.com/ and follow the Cloud project
   setup prompts. Create/select a project, enable the Earth Engine API, and complete
   the applicable Earth Engine registration/eligibility process. Review the usage
   category yourself; this repository does not create projects or change billing.
2. Note the **project ID**, not its display name or project number.
3. From the repository root, install optional dependencies if not already installed:

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-earthengine.txt
```

4. Replace YOUR_PROJECT_ID with that ID and run this in your own VS Code terminal:

```powershell
.\.venv-check\Scripts\python.exe scripts/test_earth_engine.py --project YOUR_PROJECT_ID --authenticate
```

Complete the Google login/consent in your browser. Credentials are handled by
Google's library, outside the repository. Do not paste passwords or authentication
tokens into chat or commit credentials. Subsequent checks need no login flag:

```powershell
.\.venv-check\Scripts\python.exe scripts/test_earth_engine.py --project YOUR_PROJECT_ID
```

The project ID can alternatively be set in the EARTH_ENGINE_PROJECT environment
variable. A successful check demonstrates catalog access, not available cloud-free
imagery at your stations. Actual extraction still requires verified station
coordinates, sampling dates, quality rules, and documented product scaling.
Nominal revisit frequency does not guarantee usable observations. Do not feed raw
scaled spectral values into a reflectance-labelled CSV without explicit conversion.

A bounded extraction pilot is now implemented and has been run using the supplied
ground files; see [pilot workflow and limitations](cpcb_pilot.md). Source assumptions
remain provisional, and the connection checker itself performs no extraction.

Official guidance: [authentication and initialization](https://developers.google.com/earth-engine/guides/auth),
[project setup](https://developers.google.com/earth-engine/guides/access), and
[harmonized surface-reflectance catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED).
