import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import fixture from "./demo.test-fixture.json";
vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }) => <div>{children}</div>,
  TileLayer: () => null,
  CircleMarker: ({ children }) => <div>{children}</div>,
  Popup: () => null,
  useMap: () => ({ setView: () => {} }),
}));
vi.mock("recharts", () => {
  const Wrapper = ({ children }) => <div>{children}</div>;
  const Chart = ({ children, data }) => (
    <div data-testid="chart-data" data-rows={JSON.stringify(data)}>
      {children}
    </div>
  );
  const Series = ({ name }) => <span>{name}</span>;
  return {
    ResponsiveContainer: Wrapper,
    ScatterChart: Wrapper,
    Scatter: Series,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    LineChart: Chart,
    Line: Series,
    Legend: () => null,
    ReferenceLine: () => null,
    BarChart: Chart,
    Bar: Wrapper,
    Cell: () => null,
  };
});
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url, options) => {
      let value;
      if (url === "/api/catalog") value = fixture.catalog;
      else if (url.startsWith("/api/evaluation?"))
        value = fixture.evaluations[url.split("cohort=")[1]];
      else if (url.startsWith("/api/data/")) value = fixture.data;
      else if (url === "/api/predict")
        value = fixture.predictions[JSON.parse(options.body).model_id];
      else if (url === "/api/oxygen") value = fixture.oxygen;
      else throw new Error("Unexpected request " + url);
      return { ok: true, json: async () => value };
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
async function ready() {
  render(<App />);
  await screen.findByRole("button", { name: "Run prediction" });
  return userEvent.setup();
}
describe("professor workflow", () => {
  it("switches all three models, clears stale results and aligns SHAP with the selected model", async () => {
    const user = await ready();
    for (const m of fixture.catalog.models) {
      await user.selectOptions(screen.getByLabelText("BOD model"), m.id);
      expect(screen.queryByText("Why this prediction?")).toBeNull();
      await user.click(screen.getByRole("button", { name: "Run prediction" }));
      await screen.findByText("Why this prediction?");
      const last = fetch.mock.calls
        .filter(([url]) => url === "/api/predict")
        .at(-1);
      expect(JSON.parse(last[1].body)).toEqual({
        observation_id: fixture.data.observation.observation_id,
        model_id: m.id,
        source: "saved",
      });
      const features = JSON.parse(
        screen.getByTestId("chart-data").getAttribute("data-rows"),
      );
      expect(features).toHaveLength(m.input_requirements.length);
      expect(screen.getByText("Predicted BOD")).toBeTruthy();
    }
    const satelliteInputs = screen
      .getByText("Satellite-only with band ratios", { selector: "h2" })
      .closest("section").textContent;
    expect(satelliteInputs).not.toContain("fecal coliform");
  });
  it("uses both saved evaluation cohorts and provides their CSV links", async () => {
    const user = await ready();
    await user.click(screen.getByRole("button", { name: "Accuracy" }));
    await screen.findByText("0.185");
    expect(screen.getByText("9.37%")).toBeTruthy();
    await user.selectOptions(
      screen.getByLabelText("Evaluation cohort"),
      "new_stations_strict_26",
    );
    await screen.findByText("0.755");
    expect(screen.getByText("33.67%")).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "CSV download" }).getAttribute("href"),
    ).toContain("new_stations_strict_26");
    expect(screen.getByText("training mean")).toBeTruthy();
  });
  it("discloses missing satellite imagery and renders PINN and reference curves with measured DO", async () => {
    const user = await ready();
    await user.click(screen.getByRole("button", { name: "Data & matching" }));
    await screen.findByText("Satellite thumbnail unavailable");
    await user.selectOptions(
      screen.getByLabelText("BOD model"),
      "satellite_ratios",
    );
    await user.click(screen.getByRole("button", { name: "Oxygen simulation" }));
    expect(
      screen.getByText(
        /additional oxygen simulation requires initial dissolved oxygen/,
      ),
    ).toBeTruthy();
    await user.click(
      screen.getByRole("button", { name: "Run oxygen simulation" }),
    );
    await screen.findByText("Dissolved oxygen over time");
    const charts = screen.getAllByTestId("chart-data");
    expect(charts).toHaveLength(2);
    expect(JSON.parse(charts[0].getAttribute("data-rows"))).toHaveLength(101);
    expect(screen.getAllByText("Mathematical reference")).toHaveLength(2);
  });
  it("shows API failure and recovers through retry", async () => {
    fetch.mockImplementationOnce(async () => ({
      ok: false,
      json: async () => ({ detail: "Saved artifact unavailable" }),
    }));
    render(<App />);
    await screen.findByRole("alert");
    await userEvent.click(
      screen.getByRole("button", { name: "Retry connection" }),
    );
    await screen.findByRole("button", { name: "Run prediction" });
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });
});
