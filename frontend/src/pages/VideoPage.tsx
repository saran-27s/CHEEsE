import { useEffect, useState } from "react";
import { api, fmtTime, type Candidate, type Health, type Job, type JobEvent, type ResolveResult, type StartSpec, type VideoInfo } from "../api";
import { Board } from "../components/Board";
import { CornerPicker, calibReady, emptyCalib, type Calib } from "../components/CornerPicker";
import { MoveCard, ResolveView } from "../components/MoveInfo";
import { placementFromFen } from "../fen";

const ACTIVE = new Set(["queued", "running", "paused_for_review"]);

const STATE_TEXT: Record<string, string> = {
  STABLE: "stable: board matches the confirmed position",
  OCCLUDED_OR_MOVING: "occluded or moving: hand or motion over the board, nothing is captured",
  SETTLING: "settling: waiting for the board to stay unchanged long enough",
  VALIDATING: "validating: checking the settled board against legal moves",
  NEEDS_REVIEW: "needs review: the evidence does not identify one legal move",
};

function ReviewPanel({ job, event, onDone }: { job: Job; event: JobEvent; onDone: () => void }) {
  const [uci, setUci] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const result = event.payload.result as ResolveResult;
  const fen = event.payload.fen as string;
  const sequence = event.payload.sequence as
    | { status: string; reason: string; candidates: { ucis: string[]; sans: string[]; score: number; margin_to_best: number }[] }
    | null;

  const send = async (action: "move" | "no_move" | "stop", moves?: string[]) => {
    setBusy(true); setError(null);
    try {
      await api.reviewJob(job.id, { event_id: event.id, action, ucis: moves });
      setUci("");
      onDone();
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const pick = (c: Candidate) => (c.uci ? send("move", [c.uci]) : send("no_move"));
  const typed = uci.split(/[\s,]+/).filter(Boolean);

  return (
    <section className="panel review">
      <h3>Review needed at {fmtTime(event.t_settled)} (board settled)</h3>
      <p className="small">Analysis is paused. The settled board could not be explained by exactly one legal move, so no
        move was recorded. Choose what happened, or stop the analysis here. Moves already recorded are kept.
        If several moves were played before the board was still, enter all of them in order.</p>
      {error && <div className="error">{error}</div>}
      <div className="row" style={{ alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
        {event.thumbnail_url && (
          <div><div className="small">Settled frame</div><img src={event.thumbnail_url} alt="settled frame" style={{ width: 300 }} /></div>
        )}
        <div><div className="small">Confirmed position before this change</div><Board placement={placementFromFen(fen)} size={260} /></div>
      </div>
      <ResolveView r={result} onPick={busy ? undefined : pick} />
      {sequence && sequence.candidates.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="small"><b>Several-move explanations</b> ({sequence.status}: {sequence.reason})</div>
          <ul className="small">
            {sequence.candidates.map((c) => (
              <li key={c.ucis.join(" ")}>
                <button disabled={busy} onClick={() => send("move", c.ucis)}>Record {c.sans.join(" ")}</button>
                <span className="muted"> score {c.score.toFixed(1)}, {c.margin_to_best.toFixed(1)} behind best</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="row" style={{ marginTop: 8, flexWrap: "wrap" }}>
        <input style={{ width: 320 }} placeholder="moves in UCI, in order, e.g. g1f3 b8c6" value={uci}
          onChange={(e) => setUci(e.target.value)} />
        <button disabled={busy || typed.length === 0} onClick={() => send("move", typed)}>
          Record {typed.length > 1 ? `these ${typed.length} moves` : "this move"}</button>
        <button disabled={busy} onClick={() => send("no_move")}>No move happened</button>
        <button disabled={busy} className="danger" onClick={() => send("stop")}>Stop analysis here</button>
      </div>
      <p className="muted small">Position before (FEN): <code className="fen">{fen}</code></p>
    </section>
  );
}

function EventRow({ e }: { e: JobEvent }) {
  if (e.kind === "summary") {
    const p = e.payload as { moves: number; reviews: number; final_state: string; registered?: number; frames?: number;
      max_shift_px?: number; follow_board?: boolean; warnings: string[] };
    return (
      <li className="event">
        <div>
          <div className="small"><b>Summary</b>: {p.moves} move(s) recorded automatically, {p.reviews} review(s), tracker ended {p.final_state}.
            {p.follow_board ? ` Board tracking: ${p.registered}/${p.frames} frames registered, largest board shift ${p.max_shift_px} px.` : ""}</div>
          {p.warnings.map((w) => <div key={w} className="warn small">{w}</div>)}
        </div>
      </li>
    );
  }
  if (e.kind === "move") {
    const seq = e.payload.inferred_sequence as { sans: string[]; index: number } | undefined;
    return (
      <li className="event">
        {e.thumbnail_url && <img src={e.thumbnail_url} alt="" />}
        <div style={{ flex: 1 }}>
          <div className="small muted">board settled at {fmtTime(e.t_settled)} · accepted at {fmtTime(e.payload.t_accepted)} ·
            margin {typeof e.payload.margin === "number" ? e.payload.margin.toFixed(1) : "–"}
            {seq ? ` · move ${seq.index + 1} of ${seq.sans.length} played before the board was still (${seq.sans.join(" ")})` : ""}</div>
          {e.payload.description ? <MoveCard d={e.payload.description} /> : <code>{e.payload.uci}</code>}
        </div>
      </li>
    );
  }
  const res = e.resolution as { action: string; uci?: string; ucis?: string[] } | null;
  const text = !res ? "waiting for your decision"
    : res.action === "move" ? `you chose ${(res.ucis ?? [res.uci]).join(" ")}` : res.action === "no_move" ? "you said no move happened" : "you stopped the analysis";
  return (
    <li className="event">
      {e.thumbnail_url && <img src={e.thumbnail_url} alt="" />}
      <div>
        <div className="small muted">board settled at {fmtTime(e.t_settled)}</div>
        <span className="tag warn">review: {(e.payload.result as ResolveResult).status}</span> {text}
      </div>
    </li>
  );
}

export function VideoPage({ health, openGame }: { health: Health | null; openGame: (id: string) => void }) {
  const [video, setVideo] = useState<VideoInfo | null>(null);
  const [slider, setSlider] = useState(0);
  const [frameT, setFrameT] = useState(0);
  const [calib, setCalib] = useState<Calib>(emptyCalib());
  const [preview, setPreview] = useState<{ url: string | null; errors: string[] } | null>(null);
  const [startKind, setStartKind] = useState<"start" | "fen">("start");
  const [fen, setFen] = useState("");
  const [title, setTitle] = useState("");
  const [sampleFps, setSampleFps] = useState("");
  const [stableSeconds, setStableSeconds] = useState("");
  const [minFrames, setMinFrames] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const automated = health?.mode === "automated";
  const defaults = health?.settings as { video_sample_fps?: number; tracker?: { stable_seconds?: number; min_frames?: number } } | undefined;

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const refreshJob = async (id: string) => {
    try { setJob(await api.job(id)); } catch (e) { setError((e as Error).message); }
  };

  const jobId = job?.id;
  const jobActive = job ? ACTIVE.has(job.status) : false;
  useEffect(() => {
    if (!jobId || !jobActive) return;
    const timer = setInterval(() => { void refreshJob(jobId); }, 1000);
    return () => clearInterval(timer);
  }, [jobId, jobActive]);

  const onFile = (f: File | undefined) => f && run(async () => {
    const v = await api.uploadVideo(f);
    setVideo(v); setSlider(0); setFrameT(0); setCalib(emptyCalib()); setPreview(null); setJob(null);
  });

  const calibBody = () => ({ corners: calib.corners, a1_index: calib.a1Index!, camera_side: calib.cameraSide });

  const doPreview = () => run(async () => {
    const r = await api.previewVideoCalibration(video!.id, { t: frameT, ...calibBody() });
    setPreview({ url: r.preview_url, errors: r.errors });
  });

  const num = (s: string) => (s.trim() === "" ? undefined : Number(s));
  const startSpec = (): StartSpec | null => (startKind === "start" ? { kind: "start" } : fen.trim() ? { kind: "fen", fen: fen.trim() } : null);

  const start = () => run(async () => {
    const spec = startSpec();
    if (!spec) throw new Error("Enter the FEN of the position at the start of the video.");
    const j = await api.startJob(video!.id, {
      ...calibBody(), calibration_t: frameT, start: spec, title: title.trim() || undefined,
      sample_fps: num(sampleFps), stable_seconds: num(stableSeconds), min_frames: num(minFrames),
    });
    setJob(j);
  });

  const cancel = () => run(async () => { setJob(await api.cancelJob(job!.id)); });

  const pending = job?.pending_review_event_id ? job.events.find((e) => e.id === job.pending_review_event_id) : undefined;
  const moves = job?.events.filter((e) => e.kind === "move").length ?? 0;
  const frameUrl = video ? `/api/videos/${video.id}/frame?t=${frameT.toFixed(2)}` : "";

  return (
    <div>
      <h2>Analyze a recorded video</h2>
      <p className="small">The board is calibrated once on a frame you choose. If the board or camera is nudged later, the
        board is re-located on every frame from its texture. A move is recorded only after the board has been unchanged
        for the stability time and the change is explained by legal moves. Frames with a hand over the board are never used.</p>
      {!automated && (
        <p className="warn">{health ? `${health.mode_label}. ` : ""}Video analysis needs the piece detector, so it is unavailable.
          Images can still be analyzed manually.</p>
      )}
      {error && <div className="error">{error}</div>}

      <section className="panel">
        <h3>1. Video</h3>
        <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,video/*" onChange={(e) => onFile(e.target.files?.[0])} disabled={busy} />
        {busy && <span className="muted"> working…</span>}
        {video && (
          <p className="small">{video.filename}: {fmtTime(video.meta.duration)}, {video.meta.fps.toFixed(2)} fps,
            {" "}{video.meta.width}×{video.meta.height}, {video.meta.frame_count} frames.{" "}
            <span className="muted">Timestamps: {video.meta.timestamp_source}</span></p>
        )}
      </section>

      {video && (
        <section className="panel">
          <h3>2. Calibrate on a frame</h3>
          <div className="row" style={{ flexWrap: "wrap" }}>
            <label className="small">Frame time {fmtTime(slider)}{" "}
              <input type="range" min={0} max={Math.max(0, video.meta.duration - 0.05)} step={0.05} value={slider}
                style={{ width: 360 }}
                onChange={(e) => setSlider(Number(e.target.value))}
                onPointerUp={() => setFrameT(slider)} onKeyUp={() => setFrameT(slider)} />
            </label>
            <span className="muted small">Pick a frame where all four corners are visible and no hand covers the board.</span>
          </div>
          <CornerPicker src={frameUrl} value={calib} onChange={(c) => { setCalib(c); setPreview(null); }} />
          <div className="row" style={{ marginTop: 8 }}>
            <button disabled={!calibReady(calib) || busy} onClick={doPreview}>Preview corrected board</button>
          </div>
          {preview && (
            <div style={{ marginTop: 8 }}>
              {preview.errors.length > 0 && <div className="error">{preview.errors.join("; ")}</div>}
              {preview.url && <img src={preview.url} alt="corrected board" style={{ width: 320, border: "1px solid #888" }} />}
              {preview.url && <p className="muted small">Corrected view: a8 must be top-left. If not, fix the a1 choice.</p>}
            </div>
          )}
        </section>
      )}

      {video && (
        <section className="panel">
          <h3>3. Starting position and settings</h3>
          <div className="row" style={{ flexWrap: "wrap" }}>
            <label><input type="radio" checked={startKind === "start"} onChange={() => setStartKind("start")} /> Standard starting position</label>
            <label><input type="radio" checked={startKind === "fen"} onChange={() => setStartKind("fen")} /> Position given as FEN</label>
          </div>
          {startKind === "fen" && (
            <input style={{ width: "100%" }} placeholder="FEN at the start of the video (include side to move and castling rights)"
              value={fen} onChange={(e) => setFen(e.target.value)} />
          )}
          <div className="grid2" style={{ marginTop: 8 }}>
            <label className="small">Game title <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder={`Video ${video.filename}`} /></label>
            <label className="small">Sampled frames per second
              <input type="number" min={0.1} max={60} step={0.5} value={sampleFps} onChange={(e) => setSampleFps(e.target.value)}
                placeholder={String(defaults?.video_sample_fps ?? "")} /></label>
            <label className="small">Stability time (seconds)
              <input type="number" min={0.1} max={30} step={0.1} value={stableSeconds} onChange={(e) => setStableSeconds(e.target.value)}
                placeholder={String(defaults?.tracker?.stable_seconds ?? "")} /></label>
            <label className="small">Minimum frames in the stable window
              <input type="number" min={1} max={100} step={1} value={minFrames} onChange={(e) => setMinFrames(e.target.value)}
                placeholder={String(defaults?.tracker?.min_frames ?? "")} /></label>
          </div>
          <p className="muted small">Leave a setting empty to use the server default shown in grey.</p>
          <button className="primary" disabled={!calibReady(calib) || busy || !automated || jobActive} onClick={start}
            title={automated ? "" : health?.mode_label}>Start analysis</button>
        </section>
      )}

      {job && (
        <section className="panel">
          <h3>4. Progress</h3>
          <div className="row" style={{ flexWrap: "wrap" }}>
            <span className={`tag ${job.status === "failed" ? "error" : job.status === "completed" ? "ok" : job.status === "paused_for_review" ? "warn" : ""}`}>
              {job.status.replace(/_/g, " ")}</span>
            <progress value={job.progress} max={1} style={{ width: 240 }} />
            <span className="small">{Math.round(job.progress * 100)}% · media time {fmtTime(job.media_time)} · {moves} move(s)</span>
            {jobActive && <button className="danger" onClick={cancel} disabled={busy}>Cancel</button>}
            <button onClick={() => openGame(job.game_id)}>Open game for correction and export</button>
          </div>
          {job.tracker_state && <p className="small">Tracker: <b>{job.tracker_state}</b>{STATE_TEXT[job.tracker_state] ? ` (${STATE_TEXT[job.tracker_state]})` : ""}</p>}
          {job.capabilities && (
            <p className={job.capabilities.degraded ? "warn small" : "small"}>
              {job.capabilities.degraded ? "Degraded occlusion handling: " : ""}{job.capabilities.note}
              {job.capabilities.model_id ? ` (model ${job.capabilities.model_id})` : ""}</p>
          )}
          {job.error && <div className="error">{job.error}</div>}
          <p className="muted small">{job.timestamp_note}</p>
        </section>
      )}

      {job && pending && <ReviewPanel key={pending.id} job={job} event={pending} onDone={() => void refreshJob(job.id)} />}

      {job && job.events.length > 0 && (
        <section className="panel">
          <h3>Detected events</h3>
          <ul className="events">{job.events.map((e) => <EventRow key={e.id} e={e} />)}</ul>
        </section>
      )}
    </div>
  );
}
