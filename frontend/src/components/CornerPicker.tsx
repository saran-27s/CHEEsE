import { useRef, useState } from "react";
import type { CameraSide, Point } from "../api";

export interface Calib {
  corners: Point[];
  a1Index: number | null;
  cameraSide: CameraSide;
}

export const emptyCalib = (): Calib => ({ corners: [], a1Index: null, cameraSide: "auto" });

interface Props {
  src: string;
  value: Calib;
  onChange: (c: Calib) => void;
  onSuggest?: () => void;
  suggestAvailable?: boolean;
}

const CAMERA_SIDES: [CameraSide, string][] = [
  ["auto", "Estimate from perspective"],
  ["none", "Top-down / already rectified image"],
  ["rank1", "Camera on White's side (rank 1)"],
  ["rank8", "Camera on Black's side (rank 8)"],
  ["afile", "Camera on the a-file side"],
  ["hfile", "Camera on the h-file side"],
];

/** Click four board corners clockwise on screen, starting at the top-left, then choose which one is a1. */
export function CornerPicker({ src, value, onChange, onSuggest, suggestAvailable }: Props) {
  const img = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState<[number, number] | null>(null);

  const click = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!img.current || value.corners.length >= 4) return;
    const r = img.current.getBoundingClientRect();
    const sx = img.current.naturalWidth / r.width;
    const sy = img.current.naturalHeight / r.height;
    const p: Point = [Math.round((e.clientX - r.left) * sx), Math.round((e.clientY - r.top) * sy)];
    onChange({ ...value, corners: [...value.corners, p] });
  };

  const [w, h] = natural ?? [1, 1];
  const labels = ["1 (top-left)", "2", "3", "4"];
  return (
    <div>
      <p className="small">
        Click the four <b>outer corners of the playing area</b> clockwise as they appear on screen, starting with the
        top-left corner. Then choose which clicked corner is <b>a1</b> (White's queen-side rook corner).
      </p>
      <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
        <img ref={img} src={src} onClick={click} onLoad={(e) => setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
          style={{ maxWidth: 640, width: "100%", cursor: value.corners.length < 4 ? "crosshair" : "default", display: "block" }} alt="calibration source" />
        {natural && (
          <svg viewBox={`0 0 ${w} ${h}`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
            {value.corners.length > 1 && (
              <polygon points={value.corners.map((p) => p.join(",")).join(" ")} fill="#3498db22" stroke="#00e5ff" strokeWidth={w / 300} />
            )}
            {value.corners.map((p, i) => (
              <g key={i}>
                <circle cx={p[0]} cy={p[1]} r={w / 90} fill={value.a1Index === i ? "#e74c3c" : "#00e5ff"} stroke="#000" strokeWidth={w / 600} />
                <text x={p[0] + w / 70} y={p[1] - w / 70} fontSize={w / 30} fill="#fff" stroke="#000" strokeWidth={w / 900}>
                  {i + 1}{value.a1Index === i ? " = a1" : ""}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>
      <div className="row" style={{ marginTop: 6, flexWrap: "wrap" }}>
        <span className="small">{value.corners.length}/4 corners</span>
        <button onClick={() => onChange({ ...value, corners: value.corners.slice(0, -1), a1Index: null })} disabled={!value.corners.length}>Undo</button>
        <button onClick={() => onChange(emptyCalib())}>Reset</button>
        {onSuggest && (
          <button onClick={onSuggest} disabled={!suggestAvailable} title={suggestAvailable ? "" : "corner model not available"}>
            Suggest corners (model)
          </button>
        )}
      </div>
      {value.corners.length === 4 && (
        <div className="row" style={{ marginTop: 6, flexWrap: "wrap" }}>
          <span className="small"><b>a1 corner:</b></span>
          {labels.map((l, i) => (
            <label key={i} className="small">
              <input type="radio" checked={value.a1Index === i} onChange={() => onChange({ ...value, a1Index: i })} /> {l}
            </label>
          ))}
        </div>
      )}
      <div className="row" style={{ marginTop: 6 }}>
        <label className="small">Camera position (piece anchoring):{" "}
          <select value={value.cameraSide} onChange={(e) => onChange({ ...value, cameraSide: e.target.value as CameraSide })}>
            {CAMERA_SIDES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
      </div>
    </div>
  );
}

export const calibReady = (c: Calib) => c.corners.length === 4 && c.a1Index !== null;
