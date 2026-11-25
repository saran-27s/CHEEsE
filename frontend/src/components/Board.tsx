import { useState } from "react";
import type { Placement } from "../api";

const FILES = "abcdefgh";
const GLYPH: Record<string, string> = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const PALETTE = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "empty"];

export function PieceGlyph({ symbol, size = 34 }: { symbol: string; size?: number }) {
  const white = symbol === symbol.toUpperCase();
  return (
    <span style={{ fontSize: size, lineHeight: 1, color: white ? "#fff" : "#111",
      WebkitTextStroke: white ? "1.2px #222" : "0.4px #000" }}>{GLYPH[symbol.toLowerCase()]}</span>
  );
}

interface Props {
  placement: Placement;
  uncertain?: string[];
  occluded?: string[];
  highlight?: string[];
  editable?: boolean;
  onChange?: (p: Placement) => void;
  size?: number;
}

/** Digital board. Canonical orientation: a8 top-left (White at the bottom); a flip toggle is provided. */
export function Board({ placement, uncertain = [], occluded = [], highlight = [], editable, onChange, size = 360 }: Props) {
  const [flipped, setFlipped] = useState(false);
  const [brush, setBrush] = useState<string>("P");
  const cell = size / 8;

  const squareAt = (col: number, row: number) => {
    const file = flipped ? 7 - col : col;
    const rank = flipped ? row + 1 : 8 - row;
    return `${FILES[file]}${rank}`;
  };

  const click = (sq: string) => {
    if (!editable || !onChange) return;
    const next = { ...placement };
    if (brush === "empty" || next[sq] === brush) delete next[sq];
    else next[sq] = brush;
    onChange(next);
  };

  const cells = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const sq = squareAt(col, row);
      const light = (col + row) % 2 === 0;
      const isOcc = occluded.includes(sq);
      const isUnc = uncertain.includes(sq);
      const isHi = highlight.includes(sq);
      const piece = placement[sq];
      cells.push(
        <div key={sq} title={`${sq}${isOcc ? " (occluded)" : isUnc ? " (uncertain)" : ""}`} onClick={() => click(sq)}
          style={{ position: "absolute", left: col * cell, top: row * cell, width: cell, height: cell,
            background: isHi ? (light ? "#f6e58d" : "#d4b62f") : light ? "#eeeed2" : "#769656",
            display: "flex", alignItems: "center", justifyContent: "center", cursor: editable ? "pointer" : "default",
            boxShadow: isUnc ? "inset 0 0 0 3px #e74c3c" : undefined }}>
          {isOcc && <div style={{ position: "absolute", inset: 0, background: "repeating-linear-gradient(45deg,#0003 0 4px,transparent 4px 8px)" }} />}
          {piece && <PieceGlyph symbol={piece} size={cell * 0.78} />}
          {col === 0 && <span className="coord" style={{ left: 2, top: 1 }}>{sq[1]}</span>}
          {row === 7 && <span className="coord" style={{ right: 2, bottom: 0 }}>{sq[0]}</span>}
        </div>,
      );
    }
  }

  return (
    <div>
      <div style={{ position: "relative", width: size, height: size, border: "2px solid #333" }}>{cells}</div>
      <div className="row" style={{ marginTop: 6 }}>
        <button onClick={() => setFlipped(!flipped)}>Flip view ({flipped ? "Black" : "White"} at bottom)</button>
        <span className="muted small">Red outline = uncertain, hatched = occluded/unknown, yellow = changed squares</span>
      </div>
      {editable && (
        <div className="row" style={{ marginTop: 6, flexWrap: "wrap" }}>
          <span className="small">Brush:</span>
          {PALETTE.map((p) => (
            <button key={p} className={brush === p ? "active" : ""} onClick={() => setBrush(p)} style={{ minWidth: 34 }}>
              {p === "empty" ? "clear" : <PieceGlyph symbol={p} size={20} />}
            </button>
          ))}
          <button onClick={() => onChange?.({})}>Empty board</button>
          <span className="muted small">Click a square to place the brush piece; click again to remove.</span>
        </div>
      )}
    </div>
  );
}
