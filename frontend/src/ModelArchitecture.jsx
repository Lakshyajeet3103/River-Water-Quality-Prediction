import { ArrowRight } from "lucide-react";

function Flow({ name, steps }) {
  return (
    <div className="architecture-flow" role="img" aria-label={name}>
      {steps.map(([title, description], index) => (
        <div className="architecture-step" key={title}>
          {index > 0 && (
            <ArrowRight className="architecture-arrow" aria-hidden="true" />
          )}
          <div className="architecture-node">
            <span className="architecture-number">
              {String(index + 1).padStart(2, "0")}
            </span>
            <strong>{title}</strong>
            <p>{description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ModelArchitecture() {
  return (
    <>
      <section className="panel">
        <div className="eyebrow">SELECTED-FEATURE COMBINED · 4 → 8 → 1</div>
        <h2>Selected-feature combined architecture</h2>
        <Flow
          name="Selected-feature combined: three satellite bands and fecal coliform, saved preprocessing, one 8-neuron ReLU hidden layer, output conversion, BOD prediction"
          steps={[
            [
              "4 combined inputs",
              "Sentinel-2 B8, B8A, B11 + measured fecal coliform",
            ],
            [
              "Saved preprocessing",
              "log1p(fecal coliform), then the saved input StandardScaler",
            ],
            [
              "Single neural network",
              "4 inputs → 8 hidden neurons (ReLU) → 1 linear output",
            ],
            [
              "Output conversion",
              "Saved target scaler inverse transform; negative BOD is clipped to zero",
            ],
            ["Predicted BOD", "One result in mg/L"],
          ]}
        />
        <p className="muted">
          The satellite and ground inputs enter the same network. BOD is the
          target; current measured fecal coliform is required.
        </p>
      </section>
      <section className="panel">
        <div className="eyebrow">
          SATELLITE-ONLY WITH BAND RATIOS · 15 → 8 → 1
        </div>
        <h2>Satellite-only with band ratios architecture</h2>
        <Flow
          name="Satellite-only: ten satellite bands, five backend-derived ratios, saved input scaling, one 8-neuron ReLU hidden layer, output conversion, BOD prediction"
          steps={[
            ["10 satellite bands", "B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12"],
            [
              "5 derived ratios",
              "Python combines the 10 raw bands with 5 derived features: 15 inputs total",
            ],
            [
              "Saved input scaling",
              "All 15 features use the saved StandardScaler",
            ],
            [
              "Single neural network",
              "15 inputs → 8 hidden neurons (ReLU) → 1 linear output",
            ],
            [
              "Predicted BOD",
              "Saved target scaler inverse transform; clip negative BOD to zero; output in mg/L",
            ],
          ]}
        />
        <p className="muted">
          BOD prediction requires no current chemical measurements. The separate
          oxygen simulation additionally requires initial measured dissolved
          oxygen.
        </p>
        <details>
          <summary>Backend-derived ratio definitions</summary>
          <p>
            B3 / B4; B8 / B4; B5 / B4; (B3 − B8) / (B3 + B8); (B4 − B3) / (B4 +
            B3).
          </p>
          <p>
            The existing Python preprocessing rejects near-zero denominators.
            The diagram describes the saved checkpoint; it does not recalculate
            features in the browser.
          </p>
        </details>
      </section>
    </>
  );
}
