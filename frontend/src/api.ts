// Thin typed client for the local FastAPI backend. All paths are relative (/api/...).

export type Placement = Record<string, string>; // square -> FEN symbol
export type CameraSide = "auto" | "none" | "rank1" | "rank8" | "afile" | "hfile";
export type Point = [number, number];

export interface ModelStatus {
  id: string | null;
  task: string;
  status: string;
  message: string;
  path?: string | null;
  license?: string;
  source_url?: string | null;
  has_hand_class?: boolean;
  injected?: boolean;
}

export interface Health {
  status: string;
  version: string;
  mode: "automated" | "manual";
  mode_label: string;
  piece_detector: ModelStatus;
  occlusion: { hand_detection: boolean; degraded: boolean; note: string; occlusion_source: string } | null;
  models: ModelStatus[];
  manifest_error: string | null;
  runtime_installed: Record<string, boolean>;
  device: { torch_installed: boolean; device: string | null; cuda_available: boolean; torch_version?: string; gpu_name?: string | null };
  worker: { alive: boolean; current_job: string | null; queued: number };
  limits: { max_image_mb: number; max_video_mb: number };
  settings: Record<string, unknown>;
  notes: { single_image: string };
}

export interface SquareObs {
  square: string;
  probs: Record<string, number> | null;
  top: string | null;
  top_prob: number;
  occluded: boolean;
  uncertain: boolean;
  reasons: string[];
  detections: number;
}

export interface Observation {
  source: string;
  squares: SquareObs[];
  hand_detected: boolean;
  warnings: string[];
}

export interface Metadata {
  side_to_move: "w" | "b";
  castling: string;
  en_passant: string;
  halfmove: number;
  fullmove: number;
  provenance: Record<string, string>;
}

export interface PositionCheck {
  placement: Placement;
  placement_source: string;
  fen: string;
  valid: boolean;
  issues: string[];
  assumptions: string[];
  metadata: Metadata;
}

export interface Analysis {
  id: string;
  image_id: string;
  mode: "model" | "manual";
  mode_label: string;
  model: { id: string; license: string } | null;
  calibration: Record<string, unknown>;
  preview_url: string | null;
  observation: Observation;
  correction: { placement: Placement; metadata: Metadata; changed_from_observation: string[] } | null;
  uncertain_squares: string[];
  position: PositionCheck;
  note: string;
}

export interface MoveDescription {
  san: string;
  uci: string;
  from_square: string;
  to_square: string;
  coordinates: string;
  english: string;
  color: string;
  piece: string;
  is_capture: boolean;
  captured_piece: string | null;
  is_en_passant: boolean;
  is_castling: boolean;
  castling_side: string | null;
  promotion: string | null;
  is_check: boolean;
  is_checkmate: boolean;
  fen_after: string;
}

export interface Candidate {
  uci: string | null;
  score: number;
  margin_to_best: number;
  changed_squares: string[];
  mismatched_squares: string[];
  unobserved_changed_squares: string[];
  low_confidence_changed_squares: string[];
  description: MoveDescription | null;
}

export interface ResolveResult {
  status: "no_move" | "confirmed" | "ambiguous" | "invalid";
  reason: string;
  best: Candidate | null;
  candidates: Candidate[];
  margin: number | null;
  before_fen: string;
  after_fen: string | null;
  observed_squares: number;
  occluded_squares: number;
}

export interface CompareResult extends ResolveResult {
  before: { fen: string; source: string; metadata: Metadata; assumptions: string[] };
  after: { analysis_id: string; evidence_source: string; evidence: Observation };
  note: string;
}

export interface GameMove extends MoveDescription {
  ply: number;
  id: string;
  move_number: number;
  provenance: string;
  t_settled: number | null;
  t_accepted: number | null;
  thumbnail_url: string | null;
  has_evidence: boolean;
  fen_before: string;
}

export interface Game {
  id: string;
  title: string | null;
  source: string;
  start_fen: string;
  start_metadata: Metadata;
  start_assumptions: string[];
  moves: GameMove[];
  detached: { id: string; uci: string; provenance: string; reason: string; t_settled: number | null; thumbnail_url: string | null }[];
  final_fen: string;
  pgn: string;
  revalidation?: { detached_now: { uci: string; reason: string }[] };
}

export interface GameSummary {
  id: string;
  title: string | null;
  source: string;
  start_fen: string;
  move_count: number;
  updated_at: string;
}

export interface VideoInfo {
  id: string;
  filename: string;
  meta: { fps: number; frame_count: number; duration: number; width: number; height: number; timestamp_source: string };
}

export interface JobEvent {
  id: string;
  kind: "move" | "review" | "summary";
  t_settled: number | null;
  payload: any;
  thumbnail_url: string | null;
  resolution: any;
}

export interface Job {
  id: string;
  video_id: string;
  game_id: string;
  status: string;
  progress: number;
  media_time: number;
  tracker_state: string | null;
  capabilities: { hand_detection: boolean; degraded: boolean; note: string; model_id: string | null } | null;
  error: string | null;
  pending_review_event_id: string | null;
  events: JobEvent[];
  timestamp_note: string;
}

export type StartSpec = { kind: "start" } | { kind: "fen"; fen: string } | { kind: "analysis"; analysis_id: string };

function detailText(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    const parts = [d.message, ...(Array.isArray(d.errors) ? d.errors : []), ...(Array.isArray(d.issues) ? d.issues : [])];
    const text = parts.filter(Boolean).join("; ");
    return text || JSON.stringify(detail);
  }
  return String(detail);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("Backend not reachable. Start it with: uvicorn chessvision.main:app (see README).");
  }
  const text = await res.text();
  const data = text ? (() => { try { return JSON.parse(text); } catch { return text; } })() : null;
  if (!res.ok) throw new Error(`${res.status}: ${detailText(data && typeof data === "object" ? data.detail : data)}`);
  return data as T;
}

const file = (f: File) => { const fd = new FormData(); fd.append("file", f); return fd; };

export const api = {
  health: () => request<Health>("GET", "/api/health"),
  uploadImage: (f: File) => request<{ id: string; url: string; width: number; height: number }>("POST", "/api/images", file(f)),
  suggestCorners: (imageId: string) =>
    request<{ corners: Point[]; a1_index: number; confidence: number; note: string }>("POST", `/api/images/${imageId}/suggest-corners`),
  previewCalibration: (body: { image_id: string; corners: Point[]; a1_index: number; camera_side: CameraSide }) =>
    request<{ valid: boolean; errors: string[]; preview_url: string | null }>("POST", "/api/calibrations/preview", body),
  analyze: (body: { image_id: string; corners: Point[]; a1_index: number; camera_side: CameraSide; mode: "auto" | "manual"; manual_start: "empty" | "standard" }) =>
    request<Analysis>("POST", "/api/analyses", body),
  saveCorrection: (id: string, placement: Placement, metadata: Partial<Metadata>) =>
    request<Analysis>("PUT", `/api/analyses/${id}/correction`, { placement, metadata }),
  compare: (before: StartSpec, after_analysis_id: string) =>
    request<CompareResult>("POST", "/api/compare", { before, after_analysis_id }),
  games: () => request<GameSummary[]>("GET", "/api/games"),
  game: (id: string) => request<Game>("GET", `/api/games/${id}`),
  createGame: (body: { title?: string; start: StartSpec; moves: { uci: string; provenance?: string; evidence?: unknown }[] }) =>
    request<Game>("POST", "/api/games", body),
  appendMove: (id: string, uci: string) => request<Game>("POST", `/api/games/${id}/moves`, { uci }),
  editMove: (id: string, ply: number, uci: string) => request<Game>("PUT", `/api/games/${id}/moves/${ply}`, { uci }),
  deleteMove: (id: string, ply: number) => request<Game>("DELETE", `/api/games/${id}/moves/${ply}`),
  clearDetached: (id: string) => request<Game>("DELETE", `/api/games/${id}/detached`),
  exportFen: (id: string, ply?: number) =>
    request<{ fen: string; provenance: Record<string, string>; assumptions: string[] }>("GET", `/api/games/${id}/export/fen${ply === undefined ? "" : `?ply=${ply}`}`),
  uploadVideo: (f: File) => request<VideoInfo>("POST", "/api/videos", file(f)),
  previewVideoCalibration: (id: string, body: { t: number; corners: Point[]; a1_index: number; camera_side: CameraSide }) =>
    request<{ valid: boolean; errors: string[]; preview_url: string | null }>("POST", `/api/videos/${id}/calibration-preview`, body),
  startJob: (id: string, body: Record<string, unknown>) => request<Job>("POST", `/api/videos/${id}/jobs`, body),
  job: (id: string) => request<Job>("GET", `/api/jobs/${id}`),
  cancelJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/cancel`),
  reviewJob: (id: string, body: { event_id: string; action: "move" | "no_move" | "stop"; uci?: string; ucis?: string[] }) =>
    request<{ accepted: boolean }>("POST", `/api/jobs/${id}/review`, body),
};

export const fmtTime = (t: number | null | undefined) => {
  if (t === null || t === undefined) return "–";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
};
