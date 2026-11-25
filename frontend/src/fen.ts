import type { Placement } from "./api";

/** Piece placement (square -> FEN symbol) from the first field of a FEN string. */
export function placementFromFen(fen: string): Placement {
  const out: Placement = {};
  const rows = fen.split(" ")[0].split("/");
  rows.forEach((row, i) => {
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) file += Number(ch);
      else { out[`${"abcdefgh"[file]}${8 - i}`] = ch; file++; }
    }
  });
  return out;
}
