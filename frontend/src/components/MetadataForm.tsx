import type { Metadata } from "../api";

export type MetaInput = { side_to_move?: "w" | "b"; castling?: string; en_passant?: string; halfmove?: number; fullmove?: number };

/** Fields left blank are NOT sent: the backend then labels them as assumed defaults. */
export function MetadataForm({ value, onChange }: { value: MetaInput; onChange: (v: MetaInput) => void }) {
  const set = (k: keyof MetaInput, v: string) => {
    const next = { ...value } as Record<string, unknown>;
    if (v === "") delete next[k];
    else next[k] = k === "halfmove" || k === "fullmove" ? Number(v) : v;
    onChange(next as MetaInput);
  };
  return (
    <fieldset>
      <legend>Position metadata (not visible in an image)</legend>
      <div className="grid2">
        <label>Side to move
          <select value={value.side_to_move ?? ""} onChange={(e) => set("side_to_move", e.target.value)}>
            <option value="">(not set — assumed White)</option>
            <option value="w">White</option>
            <option value="b">Black</option>
          </select>
        </label>
        <label>Castling rights (e.g. KQkq, or -)
          <input value={value.castling ?? ""} placeholder="not set — assumed -" onChange={(e) => set("castling", e.target.value.trim())} />
        </label>
        <label>En passant square
          <input value={value.en_passant ?? ""} placeholder="not set — assumed -" onChange={(e) => set("en_passant", e.target.value.trim())} />
        </label>
        <label>Halfmove clock
          <input type="number" min={0} value={value.halfmove ?? ""} placeholder="assumed 0" onChange={(e) => set("halfmove", e.target.value)} />
        </label>
        <label>Fullmove number
          <input type="number" min={1} value={value.fullmove ?? ""} placeholder="assumed 1" onChange={(e) => set("fullmove", e.target.value)} />
        </label>
      </div>
      <p className="muted small">Castling rights are never inferred from where the kings and rooks stand.</p>
    </fieldset>
  );
}

export function ProvenanceTable({ meta }: { meta: Metadata }) {
  const rows: [string, string | number][] = [
    ["side_to_move", meta.side_to_move], ["castling", meta.castling], ["en_passant", meta.en_passant],
    ["halfmove", meta.halfmove], ["fullmove", meta.fullmove],
  ];
  return (
    <table className="small">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td>{k}</td><td><code>{v}</code></td>
            <td className={meta.provenance[k] === "assumed_default" ? "warn" : "muted"}>{meta.provenance[k]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
