import { useEffect, useState } from "react";
import {
  Waves,
  Activity,
  Database,
  BrainCircuit,
  FlaskConical,
  Play,
  Download,
  ArrowRight,
} from "lucide-react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
  Legend,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";
import ModelArchitecture from "./ModelArchitecture";
const fmt = (v, n = 3) => (v == null ? "Unavailable" : Number(v).toFixed(n));
const label = (k) =>
  k
    .replace(/^shap_/, "")
    .replace(/^satellite_/, "")
    .replace(/^ground_/, "")
    .replaceAll("_", " ");
const prefix = (id) =>
  id === "all_components" ? "network_all_components" : id;
import { api } from "./api";
function Recenter({ position }) {
  const map = useMap();
  useEffect(() => {
    map.setView(position, 11);
  }, [map, position]);
  return null;
}
function StationMap({ observation }) {
  if (observation.latitude == null || observation.longitude == null)
    return <p>Station coordinates unavailable</p>;
  const position = [observation.latitude, observation.longitude];
  return (
    <>
      <MapContainer
        center={position}
        zoom={11}
        scrollWheelZoom={false}
        className="map"
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <CircleMarker
          center={position}
          radius={8}
          pathOptions={{ color: "#079980", fillOpacity: 0.9 }}
        >
          <Popup>{observation.station_name}</Popup>
        </CircleMarker>
        <Recenter position={position} />
      </MapContainer>
      <p className="muted">
        OpenStreetMap background · station marker · not satellite imagery
      </p>
    </>
  );
}
function Table({ columns, rows }) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.title}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.observation_id || `${r.model}-${i}`}>
              {columns.map((c) => (
                <td key={c.key}>
                  {c.render ? c.render(r) : (r[c.key] ?? "Unavailable")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function Shap({ result }) {
  const s = result.shap;
  if (!s)
    return (
      <section className="panel">
        <h2>Why this prediction?</h2>
        <p>
          Individual SHAP explanation unavailable for this model and
          observation. Choose an observation marked “SHAP”.
        </p>
        <p className="muted">
          SHAP explains the model’s prediction. It does not change its accuracy
          or prove cause and effect.
        </p>
      </section>
    );
  const features = Object.entries(s)
    .filter(([k]) => k.startsWith("shap_"))
    .map(([k, v]) => ({ feature: label(k), contribution: v }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return (
    <section className="panel">
      <h2>Why this prediction?</h2>
      <p>
        Reference {fmt(s.reference_bod_mg_l)} mg/L + input contributions →{" "}
        {fmt(s.predicted_bod_mg_l)} mg/L
      </p>
      <div className="split">
        <div style={{ height: Math.max(250, features.length * 27) }}>
          <ResponsiveContainer>
            <BarChart
              data={features}
              layout="vertical"
              margin={{ left: 70, right: 25 }}
            >
              <XAxis type="number" unit=" mg/L" />
              <YAxis type="category" dataKey="feature" width={110} />
              <Tooltip />
              <ReferenceLine x={0} stroke="#8ca59d" />
              <Bar dataKey="contribution">
                {features.map((f) => (
                  <Cell
                    key={f.feature}
                    fill={f.contribution >= 0 ? "#d77852" : "#168b79"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3>Inputs pushing BOD upward</h3>
          {features
            .filter((f) => f.contribution > 0)
            .map((f) => (
              <p key={f.feature}>
                {f.feature} <strong>+{fmt(f.contribution)} mg/L</strong>
              </p>
            ))}
          <h3>Inputs pushing BOD downward</h3>
          {features
            .filter((f) => f.contribution < 0)
            .map((f) => (
              <p key={f.feature}>
                {f.feature} <strong>{fmt(f.contribution)} mg/L</strong>
              </p>
            ))}
        </div>
      </div>
      <p>
        Satellite: {fmt(s.satellite_contribution_mg_l)} mg/L · Ground:{" "}
        {fmt(s.ground_contribution_mg_l)} mg/L
        {s.date_gap_contribution_mg_l != null &&
          ` · Date gap: ${fmt(s.date_gap_contribution_mg_l)} mg/L`}
      </p>
      <p className="muted">
        Local feature importance for this observation; seven saved examples per
        model. SHAP explains the model’s prediction. It does not change its
        accuracy or prove cause and effect.
      </p>
    </section>
  );
}
export default function App() {
  const [catalog, setCatalog] = useState(null),
    [station, setStation] = useState(""),
    [oid, setOid] = useState(""),
    [mid, setMid] = useState("selected_four_inputs"),
    [source, setSource] = useState("saved"),
    [page, setPage] = useState("Prediction"),
    [result, setResult] = useState(null),
    [evaluationData, setEvaluation] = useState(null),
    [cohort, setCohort] = useState("familiar_20"),
    [detailData, setDetail] = useState(null),
    [oxygen, setOxygen] = useState(null),
    [days, setDays] = useState(5),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [reload, setReload] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    api("catalog", null, controller.signal)
      .then((c) => {
        setCatalog(c);
        const first =
          c.observations.find((o) => o.has_shap) || c.observations[0];
        setStation(first.station_id);
        setOid(first.observation_id);
        setError("");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [reload]);
  useEffect(() => {
    const controller = new AbortController();
    api("evaluation?cohort=" + cohort, null, controller.signal)
      .then((data) => setEvaluation({ cohort, data }))
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [cohort, reload]);
  useEffect(() => {
    if (!oid) return;
    const controller = new AbortController();
    api("data/" + encodeURIComponent(oid), null, controller.signal)
      .then((data) => setDetail({ oid, data }))
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [oid]);
  const evaluation =
      evaluationData?.cohort === cohort ? evaluationData.data : null,
    detail = detailData?.oid === oid ? detailData.data : null;
  const observations = catalog?.observations || [],
    models = catalog?.models || [],
    observation = observations.find((o) => o.observation_id === oid),
    model = models.find((m) => m.id === mid),
    stations = [
      ...new Map(
        observations.map((o) => [o.station_id, o.station_name]),
      ).entries(),
    ];
  const request = { observation_id: oid, model_id: mid, source };
  async function run() {
    setBusy(true);
    setError("");
    try {
      setResult(await api("predict", request));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function simulate() {
    setBusy(true);
    setError("");
    try {
      setOxygen(await api("oxygen", { ...request, days: Number(days) }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  function reset() {
    setResult(null);
    setOxygen(null);
    setError("");
  }
  function chooseStation(id) {
    setStation(id);
    const items = observations.filter((o) => o.station_id === id);
    setOid((items.find((o) => o.has_shap) || items[0]).observation_id);
    reset();
  }
  const nav = [
    ["Prediction", Activity],
    ["Accuracy", BrainCircuit],
    ["Data & matching", Database],
    ["Oxygen simulation", FlaskConical],
    ["Architecture", BrainCircuit],
  ];
  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <Waves size={30} />
          <span>
            RIVER<b>OS</b>
          </span>
        </div>
        <p className="sidebar-caption">
          RIVER WATER QUALITY
          <br />
          RESEARCH WORKSPACE
        </p>
        <nav>
          {nav.map(([name, Icon]) => (
            <button
              className={page === name ? "active" : ""}
              key={name}
              onClick={() => setPage(name)}
            >
              <Icon size={18} />
              {name}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="dot" /> Historical research demo
          <p>
            Saved observations.
            <br />
            Frozen models.
            <br />
            Transparent evidence.
          </p>
        </div>
      </aside>
      <main>
        <header>
          <span>Research workspace / {page}</span>
          <span className="tag">PROOF OF CONCEPT</span>
        </header>
        <div className="content">
          <div className="eyebrow">SATELLITE + GROUND OBSERVATIONS</div>
          <h1>
            {page === "Prediction"
              ? "A clearer view of river quality."
              : page === "Oxygen simulation"
                ? "Experimental oxygen simulation."
                : page}
          </h1>
          <p className="intro">
            Estimate BOD, inspect the evidence, and understand what the models
            can tell us.
          </p>
          {error && (
            <div role="alert" className="error">
              {error}{" "}
              <button
                disabled={busy}
                onClick={() => {
                  reset();
                  setReload((x) => x + 1);
                  setError("");
                }}
              >
                Retry connection
              </button>
            </div>
          )}
          {!catalog ? (
            <section className="panel" aria-live="polite">
              {error
                ? "Data unavailable. Restore the bundle and start the API."
                : "Loading saved research observations…"}
            </section>
          ) : (
            <>
              <section className="panel selectors">
                <label>
                  Monitoring station
                  <select
                    disabled={busy}
                    value={station}
                    onChange={(e) => chooseStation(e.target.value)}
                  >
                    {stations.map(([id, name]) => (
                      <option key={id} value={id}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Observation / sampling date
                  <select
                    disabled={busy}
                    value={oid}
                    onChange={(e) => {
                      setOid(e.target.value);
                      reset();
                    }}
                  >
                    {observations
                      .filter((o) => o.station_id === station)
                      .map((o) => (
                        <option key={o.observation_id} value={o.observation_id}>
                          {o.sample_date} ·{" "}
                          {o.cohort === "familiar_20" ? "Familiar" : "Unseen"}
                          {o.has_shap ? " · SHAP" : ""}
                        </option>
                      ))}
                  </select>
                </label>
                <label>
                  BOD model
                  <select
                    disabled={busy}
                    value={mid}
                    onChange={(e) => {
                      setMid(e.target.value);
                      reset();
                    }}
                  >
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </label>
              </section>
              {page === "Prediction" && (
                <>
                  <div className="split">
                    <section className="panel">
                      <div className="eyebrow">01 / MODEL INPUTS</div>
                      <h2>{model.name}</h2>
                      <p>{model.trait}</p>
                      <p className="muted">{model.limitation}</p>
                      <div className="input-grid">
                        {model.input_requirements.map((k) => (
                          <div key={k}>
                            <span>{label(k)}</span>
                            <strong>
                              {fmt(
                                observation[k],
                                k.startsWith("satellite_") ? 5 : 2,
                              )}{" "}
                              <small>{catalog.units[k] || "reflectance"}</small>
                            </strong>
                          </div>
                        ))}
                      </div>
                      <p className="muted">
                        BOD is the prediction target, not an input. Ratios and
                        saved preprocessing run in Python.
                      </p>
                      <div className="run-row">
                        <label>
                          Result source
                          <select
                            disabled={busy}
                            value={source}
                            onChange={(e) => {
                              setSource(e.target.value);
                              reset();
                            }}
                          >
                            <option value="saved">
                              Saved evaluation result
                            </option>
                            <option value="live">
                              Run saved checkpoint live
                            </option>
                          </select>
                        </label>
                        <button
                          className="primary"
                          onClick={run}
                          disabled={busy}
                        >
                          <Play size={16} />
                          {busy ? "Running…" : "Run prediction"}
                        </button>
                      </div>
                    </section>
                    <section className="panel">
                      <div className="eyebrow">STATION LOCATION</div>
                      <h2>{observation.station_name}</h2>
                      <StationMap observation={observation} />
                      <p>
                        {observation.latitude}, {observation.longitude}
                      </p>
                    </section>
                  </div>
                  {result ? (
                    <>
                      <section className="panel">
                        <div className="eyebrow">
                          02 / {result.result_source.toUpperCase()}
                        </div>
                        <div className="results">
                          <div>
                            <span>Predicted BOD</span>
                            <strong>
                              {fmt(result.predicted_bod_mg_l)}{" "}
                              <small>mg/L</small>
                            </strong>
                          </div>
                          <div>
                            <span>Measured BOD</span>
                            <strong>
                              {fmt(result.measured_bod_mg_l)}{" "}
                              <small>mg/L</small>
                            </strong>
                          </div>
                          <div>
                            <span>Predicted − measured</span>
                            <strong>
                              {fmt(result.difference_mg_l)} <small>mg/L</small>
                            </strong>
                          </div>
                          <div>
                            <span>BOD-led water-quality category</span>
                            {result.category.available ? (
                              <>
                                <strong
                                  className={
                                    "badge " + result.category.bod_led_category
                                  }
                                >
                                  {result.category.bod_led_category}
                                </strong>
                                <p>
                                  Score {fmt(result.category.custom_score, 1)} /
                                  100
                                </p>
                              </>
                            ) : (
                              <p>{result.category.note}</p>
                            )}
                          </div>
                        </div>
                        <p className="muted">
                          Custom demonstration indicator; not an independently
                          validated full WQI. BOD error and category agreement
                          are separate measures.
                        </p>
                        <details>
                          <summary>
                            Category calculation and chemistry sources
                          </summary>
                          <p>
                            Predicted BOD → demonstration quality score → Best /
                            Good / Bad.
                          </p>
                          <p>
                            BOD has weight 70%; pH, dissolved oxygen and fecal
                            coliform each have weight 10%. Sub-indices use the
                            existing Python calculation and are bounded to
                            0–100. Assumptions: 25°C, 1 atm, salinity 0 ppt.
                          </p>
                          <p>{result.category.thresholds}</p>
                          {Object.entries(result.category.chemistry || {}).map(
                            ([k, v]) => (
                              <p key={k}>
                                {label(k)}: {fmt(v.value, 2)} {catalog.units[k]}{" "}
                                · {v.source} (not the current measurement)
                              </p>
                            ),
                          )}
                          <p>
                            Current measured chemicals are listed in the model
                            inputs and Data & matching view.
                          </p>
                        </details>
                        <details>
                          <summary>Model provenance</summary>
                          <p>
                            {result.model.name} · version / checkpoint SHA-256:{" "}
                            <code>{result.model.version}</code>
                          </p>
                        </details>
                      </section>
                      <Shap result={result} />
                    </>
                  ) : (
                    <section className="panel empty">
                      Select an observation and run a prediction to inspect its
                      result and explanation.
                    </section>
                  )}
                </>
              )}
              {page === "Accuracy" && (
                <>
                  {evaluation ? (
                    <>
                      <section className="panel">
                        <div className="section-head">
                          <h2>Recorded prediction errors</h2>
                          <select
                            aria-label="Evaluation cohort"
                            value={cohort}
                            onChange={(e) => setCohort(e.target.value)}
                          >
                            <option value="familiar_20">
                              Familiar stations · 20 observations
                            </option>
                            <option value="new_stations_strict_26">
                              Unseen stations · 26 observations
                            </option>
                          </select>
                          <a
                            className="button"
                            href={"/api/evaluation.csv?cohort=" + cohort}
                          >
                            <Download size={16} />
                            CSV download
                          </a>
                        </div>
                        <Table
                          columns={[
                            {
                              key: "model",
                              title: "Model",
                              render: (r) =>
                                models.find((m) => prefix(m.id) === r.model)
                                  ?.name || r.model.replaceAll("_", " "),
                            },
                            { key: "n", title: "Observations" },
                            {
                              key: "mae",
                              title: "MAE (mg/L)",
                              render: (r) => fmt(r.mae),
                            },
                            {
                              key: "mape_percent",
                              title: "Average % error (MAPE)",
                              render: (r) => fmt(r.mape_percent, 2) + "%",
                            },
                          ]}
                          rows={evaluation.metrics.filter((r) =>
                            [
                              "selected_four_inputs",
                              "network_all_components",
                              "satellite_ratios",
                              "training_mean",
                            ].includes(r.model),
                          )}
                        />
                        <p>
                          Familiar-station evaluation: 20 held-out observations
                          from stations represented during training.
                        </p>
                        <p>
                          Unseen-station evaluation: 26 observations from four
                          additional stations, excluding satellite acquisitions
                          shared with training.
                        </p>
                        <p className="muted">
                          MAE means the average difference in mg/L. Percentage
                          error is MAPE, not classification accuracy or a
                          guaranteed margin around every prediction. The
                          training-mean baseline uses only training data. Values
                          come directly from saved results; exploratory model
                          selection included repeated test inspection.
                        </p>
                      </section>
                      <section className="panel">
                        <h2>Predicted versus measured BOD</h2>
                        <div className="chart">
                          <ResponsiveContainer>
                            <ScatterChart>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis
                                dataKey="actual_bod_mg_l"
                                type="number"
                                name="Measured BOD"
                                unit=" mg/L"
                              />
                              <YAxis
                                dataKey={prefix(mid) + "_bod_mg_l"}
                                type="number"
                                name="Predicted BOD"
                                unit=" mg/L"
                              />
                              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                              <Scatter
                                name={model.name}
                                data={evaluation.predictions}
                                fill="#168b79"
                              />
                              <ReferenceLine
                                segment={[
                                  { x: 0, y: 0 },
                                  { x: 5, y: 5 },
                                ]}
                                stroke="#a0ada7"
                                ifOverflow="extendDomain"
                              />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                        <p className="muted">
                          Diagonal indicates perfect agreement. Selected model:{" "}
                          {model.name}.
                        </p>
                      </section>
                      <section className="panel">
                        <h2>Per-observation results</h2>
                        <Table
                          columns={[
                            { key: "station_name", title: "Station" },
                            { key: "sample_date", title: "Sample date" },
                            {
                              key: "actual_bod_mg_l",
                              title: "Measured mg/L",
                              render: (r) => fmt(r.actual_bod_mg_l),
                            },
                            {
                              key: "predicted",
                              title: "Predicted mg/L",
                              render: (r) => fmt(r[prefix(mid) + "_bod_mg_l"]),
                            },
                            {
                              key: "difference",
                              title: "Difference mg/L",
                              render: (r) =>
                                fmt(
                                  r[prefix(mid) + "_bod_mg_l"] -
                                    r.actual_bod_mg_l,
                                ),
                            },
                            {
                              key: "training_mean_bod_mg_l",
                              title: "Training mean mg/L",
                              render: (r) => fmt(r.training_mean_bod_mg_l),
                            },
                          ]}
                          rows={evaluation.predictions}
                        />
                      </section>
                    </>
                  ) : (
                    <section className="panel">Loading evaluation…</section>
                  )}
                </>
              )}
              {page === "Data & matching" && (
                <>
                  {detail ? (
                    <>
                      <section className="panel">
                        <h2>From observation to prediction</h2>
                        <div className="pipeline">
                          <span>
                            Station measurement + nearby satellite observation
                          </span>
                          <ArrowRight />
                          <span>Matched record</span>
                          <ArrowRight />
                          <span>Model inputs</span>
                          <ArrowRight />
                          <span>BOD prediction</span>
                        </div>
                        <p>{detail.source}</p>
                        <Table
                          columns={[
                            { key: "station_name", title: "Station" },
                            { key: "sample_date", title: "Ground sample" },
                            {
                              key: "satellite_local_date",
                              title: "Satellite date",
                            },
                            {
                              key: "temporal_offset_days",
                              title: "Date offset (days)",
                            },
                            {
                              key: "split",
                              title: "Split",
                              render: () =>
                                "Test / " +
                                (observation.cohort === "familiar_20"
                                  ? "familiar station"
                                  : "unseen station"),
                            },
                          ]}
                          rows={[observation]}
                        />
                      </section>
                      <div className="split">
                        <section className="panel">
                          <h2>Station & matching metadata</h2>
                          <p>
                            River: {detail.river || "Unavailable"} · latitude{" "}
                            {observation.latitude} · longitude{" "}
                            {observation.longitude}
                          </p>
                          <StationMap observation={observation} />
                          <p className="muted">{detail.extraction_note}</p>
                        </section>
                        <section className="panel">
                          <h2>Sentinel-2 acquisition</h2>
                          <div className="thumbnail-empty">
                            <Database size={32} />
                            <strong>Satellite thumbnail unavailable</strong>
                            <span>
                              Saved spectral measurements remain available
                              below.
                            </span>
                          </div>
                          <p>Acquisition: {observation.acquired_at}</p>
                          <p>
                            Source: Sentinel-2 · scene{" "}
                            <code>{observation.scene_id}</code>
                          </p>
                          <div className="input-grid">
                            {[
                              "cloud_fraction",
                              "water_fraction",
                              "pixel_count",
                              "coverage_fraction",
                              "distance_m",
                              "search_tier",
                            ].map((k) => (
                              <div key={k}>
                                <span>{label(k)}</span>
                                <strong>
                                  {observation[k] ?? "Unavailable"}
                                </strong>
                              </div>
                            ))}
                          </div>
                        </section>
                      </div>
                      <section className="panel">
                        <h2>
                          Measured chemistry, satellite bands & derived ratios
                        </h2>
                        <Table
                          columns={[
                            { key: "feature", title: "Parameter" },
                            { key: "value", title: "Value" },
                            { key: "unit", title: "Unit / provenance" },
                          ]}
                          rows={[
                            ...Object.entries(observation)
                              .filter(
                                ([k]) =>
                                  k.startsWith("ground_") ||
                                  k.startsWith("satellite_B"),
                              )
                              .map(([k, v]) => ({
                                feature: label(k),
                                value: v ?? "Unavailable",
                                unit:
                                  catalog.units[k] ||
                                  (k.startsWith("satellite_")
                                    ? "surface reflectance"
                                    : k === "ground_bod"
                                      ? "mg/L · measured target"
                                      : "count/100 mL"),
                              })),
                            ...Object.entries(detail.ratios).map(([k, v]) => ({
                              feature: label(k),
                              value: fmt(v, 5),
                              unit: "dimensionless · calculated by Python",
                            })),
                          ]}
                        />
                      </section>
                    </>
                  ) : (
                    <section className="panel">Loading matched record…</section>
                  )}
                </>
              )}
              {page === "Oxygen simulation" && (
                <>
                  <section className="panel">
                    <h2>Oxygen consumption and replenishment</h2>
                    <p>
                      Remaining oxygen demand consumes dissolved oxygen;
                      reaeration replenishes oxygen toward the assumed
                      saturation level.
                    </p>
                    <p>
                      This is an additional physics simulation. Its
                      equation-solving accuracy was checked, but future river
                      oxygen measurements were not available for validation.
                    </p>
                    <p>
                      Selected BOD model: {model.name}. Initial measured DO:{" "}
                      {fmt(observation.ground_dissolved_oxygen)} mg/L.
                    </p>
                    {mid === "satellite_ratios" && (
                      <p className="notice">
                        This additional oxygen simulation requires initial
                        dissolved oxygen, even though satellite-only BOD
                        prediction itself does not.
                      </p>
                    )}
                    <div className="run-row">
                      <label>
                        Simulation duration: {days} days
                        <input
                          disabled={busy}
                          type="range"
                          min="0"
                          max="5"
                          step="0.1"
                          value={days}
                          onChange={(e) => {
                            setDays(e.target.value);
                            setOxygen(null);
                          }}
                        />
                      </label>
                      <button
                        className="primary"
                        disabled={busy}
                        onClick={simulate}
                      >
                        {busy ? "Simulating…" : "Run oxygen simulation"}
                      </button>
                    </div>
                  </section>
                  {oxygen && (
                    <section className="panel">
                      <p>
                        Predicted BOD used:{" "}
                        {fmt(oxygen.inputs.predicted_bod_mg_l)} mg/L · Initial
                        measured DO:{" "}
                        {fmt(oxygen.inputs.initial_measured_do_mg_l)} mg/L
                      </p>
                      {oxygen.extrapolation && (
                        <p className="notice">
                          Starting conditions are outside the PINN training
                          range; this curve is an extrapolation.
                        </p>
                      )}
                      {[
                        [
                          "Dissolved oxygen over time",
                          "pinn_oxygen",
                          "reference_oxygen",
                        ],
                        [
                          "Remaining oxygen demand",
                          "pinn_demand",
                          "reference_demand",
                        ],
                      ].map(([title, pinn, reference]) => (
                        <div key={title}>
                          <h2>{title}</h2>
                          <div className="chart">
                            <ResponsiveContainer>
                              <LineChart data={oxygen.curves}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="day" unit=" d" />
                                <YAxis unit=" mg/L" />
                                <Tooltip />
                                <Legend />
                                <Line
                                  dataKey={pinn}
                                  name="PINN"
                                  stroke="#168b79"
                                  dot={false}
                                  strokeWidth={2}
                                />
                                <Line
                                  dataKey={reference}
                                  name="Mathematical reference"
                                  stroke="#d77852"
                                  strokeDasharray="5 5"
                                  dot={false}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      ))}
                      <details>
                        <summary>Assumed rates and conditions</summary>
                        <p>dL/dt = −kd × L; dO/dt = kr × (Cs − O) − kd × L.</p>
                        <p>
                          Decay kd: {oxygen.assumptions.kd_per_day} day⁻¹;
                          replenishment kr: {oxygen.assumptions.kr_per_day}{" "}
                          day⁻¹; saturation Cs: {oxygen.assumptions.Cs_mg_l}{" "}
                          mg/L.
                        </p>
                        <p>
                          {oxygen.assumptions.BOD_conversion}. Constant
                          freshwater conditions near 25°C, sea level; no
                          site-calibrated flow, travel time, mixing or other
                          oxygen loads.
                        </p>
                      </details>
                    </section>
                  )}
                </>
              )}
              {page === "Architecture" && (
                <>
                  <ModelArchitecture />
                  <section className="panel">
                    <div className="eyebrow">TWO-BRANCH FUSION</div>
                    <h2>Two-branch fusion architecture</h2>
                    <div className="pipeline">
                      <span>
                        Satellite branch
                        <br />
                        10 spectral bands
                      </span>
                      <span>+</span>
                      <span>
                        Chemical branch
                        <br />
                        Fecal coliform, pH, DO
                      </span>
                      <ArrowRight />
                      <span>
                        Learned gate
                        <br />
                        date difference
                      </span>
                      <ArrowRight />
                      <span>BOD prediction</span>
                    </div>
                    {result?.gate ? (
                      <p>
                        Actual gate weights for this{" "}
                        {result.result_source.toLowerCase()}: satellite{" "}
                        {fmt(result.gate.satellite * 100, 1)}%; ground{" "}
                        {fmt(result.gate.ground * 100, 1)}%.
                      </p>
                    ) : (
                      <p className="muted">
                        Run the two-branch model on the Prediction screen to see
                        saved gate weights where available; live inference
                        returns weights for every selected observation. No
                        weights are inferred from SHAP contributions.
                      </p>
                    )}
                    <p>
                      The selected-feature model is a single combined-input
                      network. The satellite-only model derives five ratios in
                      Python before using its saved scalers and checkpoint.
                    </p>
                  </section>
                  <section className="panel">
                    <div className="section-head">
                      <h2>Recorded component experiments</h2>
                      <select
                        aria-label="Experiment cohort"
                        value={cohort}
                        onChange={(e) => setCohort(e.target.value)}
                      >
                        <option value="familiar_20">Familiar stations</option>
                        <option value="new_stations_strict_26">
                          Unseen stations
                        </option>
                      </select>
                    </div>
                    {evaluation ? (
                      <Table
                        columns={[
                          {
                            key: "model",
                            title: "Variant",
                            render: (r) =>
                              r.model
                                .replace("network_", "")
                                .replaceAll("_", " "),
                          },
                          {
                            key: "mae",
                            title: "MAE mg/L",
                            render: (r) => fmt(r.mae),
                          },
                          {
                            key: "mape_percent",
                            title: "MAPE",
                            render: (r) => fmt(r.mape_percent, 2) + "%",
                          },
                        ]}
                        rows={evaluation.metrics.filter((r) =>
                          r.model.startsWith("network_"),
                        )}
                      />
                    ) : (
                      <p>Loading experiments…</p>
                    )}
                    <p>
                      The original training constraint penalized negative BOD.
                      The oxygen PINN is a separate equation solver, not this
                      training constraint.
                    </p>
                    <p>
                      Station-adversarial training attempts to reduce
                      station-specific representations. Judge its benefit from
                      the recorded errors above; adding components does not
                      guarantee improvement.
                    </p>
                    <p className="muted">
                      Variants are shown as recorded experiments. This screen
                      does not switch or modify checkpoints.
                    </p>
                  </section>
                </>
              )}
            </>
          )}
          <footer>
            RIVEROS · Research demonstration · BOD in mg/L · Frozen evaluation
            artifacts from final-working
          </footer>
        </div>
      </main>
    </div>
  );
}
