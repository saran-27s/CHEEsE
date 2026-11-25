import { useState } from "react";
import { api, type Analysis, type CompareResult, type Health, type StartSpec } from "../api";
import { Board } from "../components/Board";
import { ImageAnalysisPanel } from "../components/ImageAnalysisPanel";
import { ResolveView } from "../components/MoveInfo";
import { placementFromFen } from "../fen";

type BeforeKind = "start" | "fen" | "image";

export function ComparePage({ health, openGame }: { health: Health | null; openGame: (id: string) => void }) {
  const [kind, setKind] = useState<BeforeKind>("start");
  const [fen, setFen] = useState("");
  const [before, setBefore] = useState<Analysis | null>(null);
  const [after, setAfter] = useState<Analysis | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const spec = (): StartSpec | null =>
    kind === "start" ? { kind: "start" } : kind === "fen" ? (fen.trim() ? { kind: "fen", fen } : null)
      : before ? { kind: "analysis", analysis_id: before.id } : null;

  const compare = async () => {
    setError(null); setResult(null);
    const s = spec();
    if (!s || !after) return;
    try { setResult(await api.compare(s, after.id)); } catch (e) { setError((e as Error).message); }
  };

  const saveGame = async () => {
    if (!result?.best?.uci) return;
    try {
      const g = await api.createGame({ title: "Image comparison", start: spec()!,
        moves: [{ uci: result.best.uci, provenance: "resolver_image", evidence: result.after.evidence }] });
      openGame(g.id);
    } catch (e) { setError((e as Error).message); }
  };

  return (
    <div>
      <h2>Compare positions</h2>
      <p className="small">Detect a single move between a trusted "before" position and an "after" image. Each image is
        calibrated on its own. If the evidence is insufficient, or several moves may have happened, no move is forced.</p>
      <section className="panel">
        <h3>Before position</h3>
        <div className="row">
          <label><input type="radio" checked={kind === "start"} onChange={() => setKind("start")} /> Standard starting position</label>
          <label><input type="radio" checked={kind === "fen"} onChange={() => setKind("fen")} /> Confirmed FEN</label>
          <label><input type="radio" checked={kind === "image"} onChange={() => setKind("image")} /> Before image</label>
        </div>
        {kind === "fen" && <input style={{ width: "100%" }} placeholder="FEN of the confirmed previous position" value={fen} onChange={(e) => setFen(e.target.value)} />}
        {kind === "image" && (
          <>
            <p className="small">Confirm the before position and set the side to move. The side to move cannot be read from an image.</p>
            <ImageAnalysisPanel title="Before image" health={health} onConfirmed={setBefore} />
            {before && <p className="ok small">Before position confirmed: <code>{before.position.fen}</code></p>}
          </>
        )}
      </section>
      <ImageAnalysisPanel title="After image" health={health} onConfirmed={setAfter} askMetadata={false} />
      <section className="panel">
        <button className="primary" disabled={!after || !spec()} onClick={compare}>Compare</button>
        {!after && <span className="muted small"> Confirm the after image first.</span>}
        {error && <div className="error">{error}</div>}
        {result && (
          <div>
            <div className="row" style={{ alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
              <div><div className="small">Before ({result.before.source})</div><Board placement={placementFromFen(result.before.fen)} size={260}
                highlight={result.best?.changed_squares ?? []} /></div>
              <div><div className="small">After ({result.after.evidence_source})</div><Board placement={placementFromFen(result.after_fen ?? result.before.fen)} size={260}
                highlight={result.status === "confirmed" ? result.best?.changed_squares ?? [] : []} /></div>
            </div>
            {result.status !== "confirmed" && result.status !== "no_move" && (
              <p className="small muted">The "after" board above shows the before position because no move was confirmed.</p>
            )}
            {result.before.assumptions.length > 0 && <ul className="warn small">{result.before.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>}
            <ResolveView r={result} />
            {result.status === "confirmed" && <button onClick={saveGame}>Save as a new game</button>}
          </div>
        )}
      </section>
    </div>
  );
}
