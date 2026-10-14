# RIVEROS professor demo

This frontend uses the existing FastAPI service and exact saved artifacts from `final-working`. Follow [the dashboard startup instructions](../docs/dashboard.md).

`npm run dev` serves the UI on port 5173 and proxies `/api` to Python on port 8000. No Earth Engine account is needed. `npm run build` and `npm run lint` verify the frontend.
