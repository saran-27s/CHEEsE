import type { Candidate, MoveDescription, ResolveResult } from "../api";

export function Flags({ d }: { d: MoveDescription }) {
  const flags = [
    d.is_capture && (d.is_en_passant ? "capture (en passant)" : `capture (${d.captured_piece})`),
    d.is_castling && `castling ${d.castling_side}`,
    d.promotion && `promotion to ${d.promotion}`,
    d.is_checkmate ? "checkmate" : d.is_check && "check",
  ].filter(Boolean) as string[];
  return <>{flags.map((f) => <span key={f} className="tag">{f}</span>)}</>;
}

export function MoveCard({ d }: { d: MoveDescription }) {
  return (
    <div className="card">
      <div className="row" style={{ gap: 16, flexWrap: "wrap" }}>
        <span><span className="muted small">SAN</span> <b style={{ fontSize: 22 }}>{d.san}</b></span>
        <span><span className="muted small">UCI</span> <code>{d.uci}</code></span>
        <span><span className="muted small">Coordinates</span> {d.coordinates}</span>
        <Flags d={d} />
      </div>
      <p style={{ margin: "6px 0 0" }}>{d.english}</p>
    </div>
  );
}

const STATUS_TEXT: Record<string, string> = {
  confirmed: "Confirmed: one legal move explains the evidence",
  no_move: "No move: the evidence matches the starting position",
  ambiguous: "Ambiguous: not enough evidence to choose a move",
  invalid: "Invalid: no single legal move (or no move) explains the evidence",
};

export function ResolveView({ r, onPick }: { r: ResolveResult; onPick?: (c: Candidate) => void }) {
  return (
    <div>
      <div className={`status status-${r.status}`}>{STATUS_TEXT[r.status]}</div>
      <p className="small">{r.reason}</p>
      {r.status === "confirmed" && r.best?.description && <MoveCard d={r.best.description} />}
      <details open={r.status !== "confirmed"}>
        <summary className="small">Ranked candidates (margin {r.margin ?? "–"}; observed {r.observed_squares}, occluded {r.occluded_squares})</summary>
        <table className="small">
          <thead><tr><th>Candidate</th><th>Score</th><th>Behind best</th><th>Changed</th><th>Contradicted</th><th>Unobserved/uncertain</th>{onPick && <th />}</tr></thead>
          <tbody>
            {r.candidates.map((c, i) => (
              <tr key={i}>
                <td>{c.description ? <><b>{c.description.san}</b> <code>{c.uci}</code></> : <i>no move</i>}</td>
                <td>{c.score.toFixed(1)}</td>
                <td>{c.margin_to_best.toFixed(1)}</td>
                <td>{c.changed_squares.join(" ")}</td>
                <td className={c.mismatched_squares.length ? "warn" : ""}>{c.mismatched_squares.join(" ")}</td>
                <td>{[...c.unobserved_changed_squares, ...c.low_confidence_changed_squares].join(" ")}</td>
                {onPick && <td><button onClick={() => onPick(c)}>{c.uci ? "Use this move" : "No move happened"}</button></td>}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
