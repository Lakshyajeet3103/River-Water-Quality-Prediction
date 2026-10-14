
import { useEffect, useState } from "react";
import {
  Activity, Waves, LayoutDashboard, Map, BrainCircuit,
  FlaskConical, Settings, Bell, Search, ChevronDown,
  ArrowUpRight, ArrowDownRight, Droplets, Wind,
  Menu, CircleHelp, ExternalLink, Clock3, ShieldCheck,
  Database, Sparkles
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar
} from "recharts";
import "./App.css";
import {
  getHealth,
  getSummary,
  getStations,
  getModelMetrics,
  getModelPredictions,
} from "./api";

const trendData = [
  { month: "Jan", ph: 7.7, oxygen: 8.7, bod: 2.8 },
  { month: "Feb", ph: 7.8, oxygen: 8.5, bod: 3.1 },
  { month: "Mar", ph: 7.9, oxygen: 8.2, bod: 3.0 },
  { month: "Apr", ph: 8.0, oxygen: 7.9, bod: 3.4 },
  { month: "May", ph: 7.8, oxygen: 8.1, bod: 3.2 },
  { month: "Jun", ph: 7.9, oxygen: 8.4, bod: 2.9 },
  { month: "Jul", ph: 7.8, oxygen: 8.6, bod: 2.7 },
  { month: "Aug", ph: 8.0, oxygen: 8.3, bod: 3.0 },
  { month: "Sep", ph: 7.9, oxygen: 8.1, bod: 3.2 },
  { month: "Oct", ph: 7.8, oxygen: 8.5, bod: 2.9 },
  { month: "Nov", ph: 7.9, oxygen: 8.2, bod: 3.1 },
  { month: "Dec", ph: 7.9, oxygen: 8.4, bod: 2.8 },
];

const stationData = [
  { name: "Ganga at Allahabad", region: "Prayagraj, UP", ph: "7.9", oxygen: "8.2", bod: "2.8", status: "Sample" },
  { name: "Ganga at Varanasi", region: "Varanasi, UP", ph: "8.0", oxygen: "7.8", bod: "3.1", status: "Sample" },
  { name: "Ganga at Kanpur", region: "Kanpur, UP", ph: "7.8", oxygen: "8.4", bod: "2.6", status: "Sample" },
  { name: "Ganga at Bijnor", region: "Bijnor, UP", ph: "7.7", oxygen: "8.6", bod: "2.4", status: "Sample" },
];

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "River stations", icon: Map },
  { label: "Water analytics", icon: FlaskConical },
  { label: "AI model lab", icon: BrainCircuit },
];

function MetricCard({ title, value, unit, note, icon: Icon, color, trend }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">{title}</span>
        <span className={`metric-icon ${color}`}><Icon size={19} /></span>
      </div>
      <div className="metric-value">{value}<span>{unit}</span></div>
      <div className="metric-bottom">
        <span className={trend === "up" ? "trend positive" : "trend neutral"}>
          {trend === "up" ? <ArrowUpRight size={14} /> : <Activity size={14} />}
          {note}
        </span>
        <span className="muted tiny">Demo</span>
      </div>
    </div>
  );
}

export default function App() {
  const [active, setActive] = useState("Overview");
  const [period, setPeriod] = useState("Year");
  const [search, setSearch] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);

  const [apiHealth, setApiHealth] = useState(null);
  const [apiSummary, setApiSummary] = useState(null);
  const [apiStations, setApiStations] = useState([]);
  const [apiMetrics, setApiMetrics] = useState([]);
  const [apiPredictions, setApiPredictions] = useState([]);
  const [apiError, setApiError] = useState("");
  const [apiLoading, setApiLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboardData() {
      setApiLoading(true);
      setApiError("");

      const results = await Promise.allSettled([
        getHealth(),
        getSummary(),
        getStations(),
        getModelMetrics(),
        getModelPredictions(),
      ]);

      if (cancelled) return;

      const [health, summary, stations, metrics, predictions] = results;

      if (health.status === "fulfilled") setApiHealth(health.value);
      if (summary.status === "fulfilled") setApiSummary(summary.value);
      if (stations.status === "fulfilled") setApiStations(stations.value.items ?? []);
      if (metrics.status === "fulfilled") setApiMetrics(metrics.value.items ?? []);
      if (predictions.status === "fulfilled") setApiPredictions(predictions.value.items ?? []);

      const errors = results
        .filter((result) => result.status === "rejected")
        .map((result) => result.reason?.message || "Unknown API error");

      if (errors.length) setApiError(errors.join(" | "));
      setApiLoading(false);
    }

    loadDashboardData();

    return () => {
      cancelled = true;
    };
  }, []);

  const filteredStations = apiStations.filter((station) =>
    (station.station_name ?? "").toLowerCase().includes(search.toLowerCase())
  );

  const pageDescriptions = {
    Overview: "A clearer picture of river health, from ground observations to model insights.",
    "River stations": "Explore monitoring stations and review sample measurements.",
    "Water analytics": "Explore water-quality parameters and their historical patterns.",
    "AI model lab": "Review model evaluation metrics and compare experimental approaches.",
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Waves size={25} /></div>
          <div>
            <div className="brand-name">RIVER<span>OS</span></div>
            <div className="brand-sub">INTELLIGENCE PLATFORM</div>
          </div>
          <button className="icon-button mobile-close" onClick={() => setMobileMenu(false)}>Ã—</button>
        </div>

        <div className="workspace">
          <div className="workspace-avatar">G</div>
          <div className="workspace-text">
            <strong>Ganga Research</strong>
            <span>Research workspace</span>
          </div>
          <ChevronDown size={15} className="muted" />
        </div>

        <div className="nav-heading">WORKSPACE</div>
        <nav className="nav-list">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              onClick={() => { setActive(label); setMobileMenu(false); }}
              className={`nav-item ${active === label ? "active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === "AI model lab" && <span className="nav-new">AI</span>}
            </button>
          ))}
        </nav>

        <div className="nav-heading secondary-heading">DATA & SYSTEM</div>
        <div className="nav-list">
          <button className="nav-item" onClick={() => setActive("Data sources")}>
            <Database size={18} /><span>Data sources</span>
          </button>
          <button className="nav-item" onClick={() => setActive("Settings")}>
            <Settings size={18} /><span>Settings</span>
          </button>
        </div>

        <div className="sidebar-spacer" />

        <div className="sidebar-status">
          <div className="status-heading"><span className="status-dot" /> SYSTEM STATUS</div>
          <div className="status-line"><span>Dashboard</span><strong>Operational</strong></div>
          <div className="status-line">
            <span>Data connection</span>
            <strong className={apiError ? "status-demo" : ""}>
              {apiLoading ? "Connecting…" : apiError ? "Partial / unavailable" : apiHealth?.status === "ok" ? "Connected" : "Unknown"}
            </strong>
          </div>
          <div className="status-progress"><span /></div>
          <p>
            {apiLoading
              ? "Connecting to the research API…"
              : apiError
                ? "Some API requests failed. Check the backend and try again."
                : "Connected to the historical pilot dataset; not live river monitoring."}
          </p>
        </div>

        <button className="profile">
          <div className="profile-avatar">S</div>
          <div className="profile-text"><strong>Researcher</strong><span>Project dashboard</span></div>
          <Settings size={16} className="muted" />
        </button>
      </aside>

      {mobileMenu && <div className="mobile-overlay" onClick={() => setMobileMenu(false)} />}

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <button className="mobile-menu-button" onClick={() => setMobileMenu(true)}><Menu size={20} /></button>
            <span>Workspace</span><span className="crumb-divider">/</span>
            <strong>{active}</strong>
          </div>
          <div className="topbar-actions">
            <div className="environment-pill"><span /> DEMO ENVIRONMENT</div>
            <button className="icon-button" aria-label="Notifications"><Bell size={18} /><i /></button>
            <div className="topbar-avatar">S</div>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> RIVER MONITORING PLATFORM</div>
              <h1>{active === "Overview" ? <>Understand your river.<br /><span>Protect its future.</span></> : active}</h1>
              <p className="welcome-description">{pageDescriptions[active] || "Configure and explore your research workspace."}</p>
            </div>
            <div className="date-display"><Clock3 size={15} /><span>Historical research view</span><ChevronDown size={14} /></div>
          </section>

          <div className="notice">
            <div className="notice-icon"><Sparkles size={17} /></div>
            <div><strong>Research preview</strong><span>All dashboard measurements and charts currently use illustrative demo values, not live river readings.</span></div>
            <span className="notice-tag">DEMO DATA</span>
          </div>

          <section className="section-heading">
            <div><h2>River health overview</h2><p>Key indicators Â· Illustrative values only</p></div>
            <button className="outline-button" onClick={() => setActive("Water analytics")}><Activity size={15} /> Explore analytics <ArrowUpRight size={15} /></button>
          </section>

          <div className="metrics-grid">
            <MetricCard
              title="Matched observations"
              value={apiLoading ? "…" : apiSummary?.paired_observations ?? "—"}
              unit=""
              note="Paired ground and satellite records"
              icon={Droplets}
              color="mint"
            />
            <MetricCard
              title="Monitoring stations"
              value={apiLoading ? "…" : apiSummary?.station_count ?? "—"}
              unit=""
              note="Stations in the pilot dataset"
              icon={Map}
              color="purple"
            />
            <MetricCard
              title="Matched events"
              value={apiLoading ? "…" : apiSummary?.matched_events ?? "—"}
              unit=""
              note="Successfully matched events"
              icon={Activity}
              color="blue"
            />
            <MetricCard
              title="Unmatched events"
              value={apiLoading ? "…" : apiSummary?.unmatched_events ?? "—"}
              unit=""
              note="Events without a matched pair"
              icon={Wind}
              color="orange"
            />
          </div>

          <section className="charts-grid">
            <div className="panel trend-panel">
              <div className="panel-header">
                <div><h3>Water quality trends</h3><p>Illustrative monthly profile</p></div>
                <select value={period} onChange={(e) => setPeriod(e.target.value)} className="period-select">
                  <option>Year</option><option>Half-year</option><option>Quarter</option>
                </select>
              </div>
              <div className="chart-legend">
                <span><i className="legend-mint" /> pH</span>
                <span><i className="legend-blue" /> Dissolved oxygen</span>
                <span><i className="legend-orange" /> BOD</span>
              </div>
              <div className="chart-area">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={period === "Quarter" ? trendData.slice(0, 3) : period === "Half-year" ? trendData.slice(0, 6) : trendData} margin={{ top: 12, right: 8, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="phFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#65d9ad" stopOpacity={0.22} /><stop offset="100%" stopColor="#65d9ad" stopOpacity={0} /></linearGradient>
                      <linearGradient id="oxygenFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#70b9f5" stopOpacity={0.15} /><stop offset="100%" stopColor="#70b9f5" stopOpacity={0} /></linearGradient>
                    </defs>
                    <CartesianGrid stroke="#26342f" strokeDasharray="3 5" vertical={false} />
                    <XAxis dataKey="month" tick={{ fill: "#82928a", fontSize: 11 }} axisLine={false} tickLine={false} dy={10} />
                    <YAxis tick={{ fill: "#82928a", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#15221d", border: "1px solid #33483d", borderRadius: 10, color: "#e8f1eb" }} />
                    <Area type="monotone" dataKey="ph" name="pH" stroke="#65d9ad" fill="url(#phFill)" strokeWidth={2.3} />
                    <Area type="monotone" dataKey="oxygen" name="Dissolved oxygen" stroke="#70b9f5" fill="url(#oxygenFill)" strokeWidth={2.3} />
                    <Area type="monotone" dataKey="bod" name="BOD" stroke="#e8a56a" fill="transparent" strokeWidth={2.1} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="panel-footer"><span><ShieldCheck size={14} /> Illustrative values</span><span>Monthly pattern Â· Not observed measurements</span></div>
            </div>

            <div className="panel model-panel">
              <div className="panel-header">
                <div><h3>Model lab</h3><p>Backend evaluation results</p></div>
                <span className="mini-badge">{apiLoading ? "LOADING" : apiError ? "CHECK API" : "PILOT"}</span>
              </div>
              <div className="model-intro">
                <div className="model-symbol"><BrainCircuit size={22} /></div>
                <div>
                  <strong>Saved model evaluation</strong>
                  <span>Target: BOD (mg/L) ? Historical test split</span>
                </div>
              </div>
              {apiMetrics.filter((metric) => metric.split === "test").map((metric) => (
                <div className="model-row" key={`${metric.model}-${metric.split}`}>
                  <div>
                    <span className="model-name">{metric.model}</span>
                    <span className="model-sub">Test samples: {metric.n ?? "?"}</span>
                    <span className="model-sub">RMSE: {metric.rmse == null ? "?" : Number(metric.rmse).toFixed(3)} ? R?: {metric.r2 == null ? "?" : Number(metric.r2).toFixed(3)}</span>
                  </div>
                  <strong>{metric.mae == null ? "?" : Number(metric.mae).toFixed(3)} <small>MAE</small></strong>
                </div>
              ))}
              {apiMetrics.filter((metric) => metric.split === "test").length === 0 && (
                <p className="model-disclaimer">
                  {apiLoading ? "Loading evaluation metrics?" : "No test-split metrics are available from the API."}
                </p>
              )}
              <div className="model-row">
                <div>
                  <span className="model-name">Saved predictions</span>
                  <span className="model-sub">Historical test outputs ? Not live forecasts</span>
                </div>
                <strong>{apiLoading ? "?" : apiPredictions.length}</strong>
              </div>
              <button className="model-link" onClick={() => setActive("AI model lab")}>Explore model experiments <ArrowUpRight size={15} /></button>
              <p className="model-disclaimer">
                {apiMetrics.length > 0 ? "Exploratory pilot only. The test set contains just 3 observations; results do not establish generalization. This is retrospective estimation, not forecasting." : "Metrics are supplied by the backend when available."}
              </p>
            </div>
          </section>

          <section className="panel stations-panel">
            <div className="panel-header stations-header">
              <div><h3>Monitoring stations</h3><p>Sample locations for the Ganga research project</p></div>
              <div className="station-actions">
                <div className="search-box"><Search size={15} /><input placeholder="Search stations..." value={search} onChange={(e) => setSearch(e.target.value)} /></div>
                <button className="outline-button compact" onClick={() => setActive("River stations")}>View stations <ArrowUpRight size={14} /></button>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>STATION</th><th>ATTEMPTED</th><th>MATCHED</th><th>UNMATCHED</th><th>DATA TYPE</th></tr></thead>
                <tbody>
                  {filteredStations.map((station) => (
                    <tr key={station.station_name}>
                      <td><div className="station-name"><span className="station-marker"><Waves size={14} /></span><div><strong>{station.station_name}</strong><span>Ganga monitoring station</span></div></div></td>
                      <td>{station.attempted ?? "—"}</td>
                      <td>{station.matched ?? "—"}</td>
                      <td>{station.unmatched ?? "—"}</td>
                      <td><span className="sample-badge"><span /> Pilot dataset</span></td>
                    </tr>
                  ))}
                  {filteredStations.length === 0 && (
                    <tr><td colSpan="5" className="empty-state">{apiLoading ? "Loading stations…" : apiError ? "Station data unavailable. Check the backend connection." : "No matching stations found."}</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="table-footer"><span>Example station entries Â· Values are illustrative</span><button onClick={() => { setSearch(""); setActive("River stations"); }}>Explore stations <ExternalLink size={13} /></button></div>
          </section>

          <footer className="page-footer"><span><Waves size={15} /> RIVER<span className="footer-brand">OS</span> <span className="footer-copy">Â· River Water Quality Research</span></span><span><CircleHelp size={14} /> Research dashboard preview Â· Built for exploration</span></footer>
        </div>
      </main>
    </div>
  );
}











