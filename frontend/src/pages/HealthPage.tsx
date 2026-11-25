import type { Health } from "../api";

export function HealthPage({ health, error, refresh }: { health: Health | null; error: string | null; refresh: () => void }) {
  return (
    <div>
      <h2>Local health and settings</h2>
      <button onClick={refresh}>Refresh</button>
      {error && <div className="error">{error}</div>}
      {health && (
        <>
          <section className="panel">
            <h3>Status</h3>
            <table className="small"><tbody>
              <tr><td>Backend</td><td className="ok">reachable (v{health.version})</td></tr>
              <tr><td>Recognition mode</td><td className={health.mode === "automated" ? "ok" : "warn"}>{health.mode_label}</td></tr>
              <tr><td>Occlusion handling</td><td className={health.occlusion?.degraded ? "warn" : ""}>{health.occlusion ? health.occlusion.note : "n/a (no piece model)"}</td></tr>
              <tr><td>Inference device</td><td>{health.device.torch_installed ? `${health.device.device}${health.device.cuda_available ? ` (${health.device.gpu_name})` : ""}, torch ${health.device.torch_version}` : "torch not installed"}</td></tr>
              <tr><td>Ultralytics runtime (AGPL-3.0, optional)</td><td>{health.runtime_installed.ultralytics ? "installed" : "not installed"}</td></tr>
              <tr><td>Background worker</td><td className={health.worker.alive ? "ok" : "error"}>{health.worker.alive ? "running" : "stopped"}; current job {health.worker.current_job ?? "none"}; queued {health.worker.queued}</td></tr>
              <tr><td>Upload limits</td><td>images {health.limits.max_image_mb} MB, videos {health.limits.max_video_mb} MB</td></tr>
              {health.manifest_error && <tr><td>Manifest</td><td className="error">{health.manifest_error}</td></tr>}
            </tbody></table>
          </section>
          <section className="panel">
            <h3>Models</h3>
            <table className="small">
              <thead><tr><th>ID</th><th>Task</th><th>Status</th><th>Message</th><th>License</th><th>Hand class</th></tr></thead>
              <tbody>{health.models.map((m) => (
                <tr key={`${m.id}-${m.task}`}>
                  <td>{m.id}{m.injected ? " (injected)" : ""}</td><td>{m.task}</td>
                  <td className={m.status === "available" ? "ok" : "warn"}>{m.status}</td><td>{m.message}</td>
                  <td>{m.license}</td><td>{m.has_hand_class ? "yes" : "no"}</td>
                </tr>))}</tbody>
            </table>
            <p className="muted small">Weights are never downloaded automatically. See README: scripts/fetch_reference_weights.py --accept-agpl.</p>
          </section>
          <section className="panel">
            <h3>Settings (read-only; change with CHESSVISION_* environment variables)</h3>
            <pre className="small">{JSON.stringify(health.settings, null, 2)}</pre>
          </section>
        </>
      )}
    </div>
  );
}
