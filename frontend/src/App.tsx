import { useCallback, useEffect, useState } from "react";
import { api, type Health } from "./api";
import { AnalyzePage } from "./pages/AnalyzePage";
import { ComparePage } from "./pages/ComparePage";
import { GamesPage } from "./pages/GamesPage";
import { HealthPage } from "./pages/HealthPage";
import { VideoPage } from "./pages/VideoPage";

type Tab = "analyze" | "compare" | "video" | "games" | "health";

const TABS: [Tab, string][] = [
  ["analyze", "Analyze image"],
  ["compare", "Compare"],
  ["video", "Analyze video"],
  ["games", "Results and correction"],
  ["health", "Health"],
];

export function App() {
  const [tab, setTab] = useState<Tab>("analyze");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [gameId, setGameId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (e) {
      setHealthError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => { void refresh(); }, 10_000);
    return () => clearInterval(timer);
  }, [refresh]);

  const openGame = (id: string) => { setGameId(id); setTab("games"); };

  return (
    <div className="app">
      <header>
        <h1>chess-vision <span className="muted small">local</span></h1>
        <nav>
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div className="statusbar small">
          {healthError ? <span className="error">Backend unreachable: {healthError}</span>
            : !health ? <span className="muted">Connecting to backend…</span>
            : <>
                <span className={health.mode === "automated" ? "tag ok" : "tag warn"}>{health.mode_label}</span>
                <span>device: {health.device.torch_installed ? health.device.device ?? "unknown" : "torch not installed"}</span>
                <span className={health.worker.alive ? "" : "error"}>
                  worker: {health.worker.alive ? (health.worker.current_job ? "busy" : "idle") : "stopped"}
                  {health.worker.queued ? `, ${health.worker.queued} queued` : ""}</span>
                {health.occlusion?.degraded && <span className="warn" title={health.occlusion.note}>occlusion: motion proxy only</span>}
              </>}
        </div>
      </header>
      <main>
        {/* Input pages stay mounted so switching tabs does not lose work in progress (and video polling continues).
            The games page remounts on each visit so it reloads the latest stored moves. */}
        <div hidden={tab !== "analyze"}><AnalyzePage health={health} /></div>
        <div hidden={tab !== "compare"}><ComparePage health={health} openGame={openGame} /></div>
        <div hidden={tab !== "video"}><VideoPage health={health} openGame={openGame} /></div>
        {tab === "games" && <GamesPage gameId={gameId} setGameId={setGameId} />}
        {tab === "health" && <HealthPage health={health} error={healthError} refresh={() => void refresh()} />}
      </main>
    </div>
  );
}
