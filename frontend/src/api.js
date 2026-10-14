const API_BASE_URL = "http://127.0.0.1:8000";

async function getJSON(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    throw new Error(`API request failed (${response.status}): ${path}`);
  }

  return response.json();
}

export const getHealth = () => getJSON("/api/health");
export const getSummary = () => getJSON("/api/summary");
export const getStations = () => getJSON("/api/stations");
export const getObservations = () => getJSON("/api/observations");
export const getSatellite = () => getJSON("/api/satellite");
export const getModelMetrics = () => getJSON("/api/model-metrics");
export const getModelPredictions = () => getJSON("/api/model-predictions");
