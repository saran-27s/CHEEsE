import { useState } from "react";
import { api, type Analysis, type Health, type Placement } from "../api";
import { Board } from "./Board";
import { CornerPicker, calibReady, emptyCalib, type Calib } from "./CornerPicker";
import { MetadataForm, ProvenanceTable, type MetaInput } from "./MetadataForm";

interface Props {
  title: string;
  health: Health | null;
  onConfirmed?: (a: Analysis) => void;
  askMetadata?: boolean;
}

/** Upload -> calibrate -> preview -> analyze (model or manual) -> correct -> confirm. */
export function ImageAnalysisPanel({ title, health, onConfirmed, askMetadata = true }: Props) {
  const [image, setImage] = useState<{ id: string; url: string } | null>(null);
  const [calib, setCalib] = useState<Calib>(emptyCalib());
  const [preview, setPreview] = useState<{ url: string | null; errors: string[] } | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [placement, setPlacement] = useState<Placement>({});
  const [meta, setMeta] = useState<MetaInput>({});
  const [manualStart, setManualStart] = useState<"empty" | "standard">("standard");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const automated = health?.mode === "automated";
  const cornerModel = health?.models.find((m) => m.task === "board_corners")?.status === "available";

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try { await fn(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const onFile = (f: File | undefined) => f && run(async () => {
    const r = await api.uploadImage(f);
    setImage({ id: r.id, url: r.url });
    setCalib(emptyCalib()); setPreview(null); setAnalysis(null);
  });

  const calibBody = () => ({ image_id: image!.id, corners: calib.corners, a1_index: calib.a1Index!, camera_side: calib.cameraSide });

  const doPreview = () => run(async () => {
    const r = await api.previewCalibration(calibBody());
    setPreview({ url: r.preview_url, errors: r.errors });
  });

  const doAnalyze = (mode: "auto" | "manual") => run(async () => {
    const a = await api.analyze({ ...calibBody(), mode, manual_start: manualStart });
    setAnalysis(a);
    setPlacement(a.position.placement);
    setPreview({ url: a.preview_url, errors: [] });
  });

  const suggest = () => run(async () => {
    const s = await api.suggestCorners(image!.id);
    setCalib({ ...calib, corners: s.corners, a1Index: s.a1_index });
    setError(`Corner suggestion (confidence ${s.confidence.toFixed(2)}): ${s.note}`);
  });

  const save = () => run(async () => {
    const a = await api.saveCorrection(analysis!.id, placement, meta);
    setAnalysis(a);
    if (a.position.valid) onConfirmed?.(a);
  });

  const obs = analysis?.observation;
  const occluded = obs ? obs.squares.filter((s) => s.occluded).map((s) => s.square) : [];
  return (
    <section className="panel">
      <h3>{title}</h3>
      {error && <div className="error">{error}</div>}
      <input type="file" accept=".jpg,.jpeg,.png,.webp" onChange={(e) => onFile(e.target.files?.[0])} />
      {image && (
        <>
          <CornerPicker src={image.url} value={calib} onChange={(c) => { setCalib(c); setPreview(null); }}
            onSuggest={suggest} suggestAvailable={cornerModel} />
          <div className="row" style={{ marginTop: 8, flexWrap: "wrap" }}>
            <button disabled={!calibReady(calib) || busy} onClick={doPreview}>Preview corrected board</button>
            <button disabled={!calibReady(calib) || busy || !automated} onClick={() => doAnalyze("auto")}
              title={automated ? "" : health?.mode_label}>Recognize pieces (model)</button>
            <button disabled={!calibReady(calib) || busy} onClick={() => doAnalyze("manual")}>Enter pieces manually</button>
            <select value={manualStart} onChange={(e) => setManualStart(e.target.value as "empty" | "standard")}>
              <option value="standard">manual: start from initial position</option>
              <option value="empty">manual: start from empty board</option>
            </select>
            {busy && <span className="muted">working…</span>}
          </div>
          {!automated && <p className="warn small">{health?.mode_label}</p>}
        </>
      )}
      {preview && (
        <div style={{ marginTop: 8 }}>
          {preview.errors.length > 0 && <div className="error">{preview.errors.join("; ")}</div>}
          {preview.url && <img src={preview.url} alt="corrected board" style={{ width: 320, border: "1px solid #888" }} />}
          {preview.url && <p className="muted small">Corrected view: a8 must be top-left. If not, fix the a1 choice.</p>}
        </div>
      )}
      {analysis && (
        <div style={{ marginTop: 10 }}>
          <div className={analysis.mode === "model" ? "tag ok" : "tag warn"}>{analysis.mode_label}
            {analysis.model ? ` — ${analysis.model.id}` : ""}</div>
          <p className="muted small">{analysis.note}</p>
          {obs?.hand_detected && <p className="warn small">A hand was detected; hatched squares are occluded and unknown.</p>}
          {obs && obs.warnings.length > 0 && <p className="warn small">{obs.warnings.join("; ")}</p>}
          <div className="row" style={{ alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
            <Board placement={placement} uncertain={analysis.uncertain_squares} occluded={occluded} editable onChange={setPlacement} />
            <div style={{ minWidth: 300, flex: 1 }}>
              {askMetadata && <MetadataForm value={meta} onChange={setMeta} />}
              <button className="primary" onClick={save} disabled={busy}>Confirm this position</button>
              <p className="muted small">Confirming stores your corrected placement separately from the model observation.</p>
              {analysis.correction && (
                <div>
                  <p className="small">Corrected squares vs observation: {analysis.correction.changed_from_observation.join(", ") || "none"}</p>
                </div>
              )}
              <div className="card">
                <div className="small">FEN ({analysis.position.placement_source === "user_correction" ? "confirmed by you" : "unconfirmed observation"})</div>
                <code className="fen">{analysis.position.fen}</code>
                {!analysis.position.valid && <div className="error small">Not a legal position: {analysis.position.issues.join("; ")}</div>}
                {analysis.position.assumptions.length > 0 && (
                  <ul className="warn small">{analysis.position.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
                )}
                <ProvenanceTable meta={analysis.position.metadata} />
                <a href={`/api/analyses/${analysis.id}/report`} target="_blank" rel="noreferrer">JSON analysis report</a>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
