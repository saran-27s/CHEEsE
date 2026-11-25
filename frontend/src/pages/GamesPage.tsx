import { useEffect, useState } from "react";
import { api, fmtTime, type Game, type GameSummary } from "../api";
import { Flags } from "../components/MoveInfo";
import { ProvenanceTable } from "../components/MetadataForm";

export function GamesPage({ gameId, setGameId }: { gameId: string | null; setGameId: (id: string | null) => void }) {
  const [list, setList] = useState<GameSummary[]>([]);
  const [game, setGame] = useState<Game | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [edit, setEdit] = useState<{ ply: number; uci: string } | null>(null);
  const [append, setAppend] = useState("");
  const [newFen, setNewFen] = useState("");
  const [fenInfo, setFenInfo] = useState<{ fen: string; assumptions: string[] } | null>(null);

  const act = async (fn: () => Promise<Game | void>) => {
    setError(null);
    try { const g = await fn(); if (g) setGame(g); setList(await api.games()); } catch (e) { setError((e as Error).message); }
  };

  useEffect(() => { api.games().then(setList).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    setFenInfo(null);
    if (gameId) api.game(gameId).then(setGame).catch((e) => setError(e.message)); else setGame(null);
  }, [gameId]);

  return (
    <div>
      <h2>Results and correction</h2>
      {error && <div className="error">{error}</div>}
      <section className="panel">
        <div className="row" style={{ flexWrap: "wrap" }}>
          <select value={gameId ?? ""} onChange={(e) => setGameId(e.target.value || null)}>
            <option value="">— select a game —</option>
            {list.map((g) => <option key={g.id} value={g.id}>{g.title ?? "Untitled"} · {g.move_count} moves · {g.source}</option>)}
          </select>
          <button onClick={() => act(async () => { const g = await api.createGame({ title: "Manual game", start: { kind: "start" }, moves: [] }); setGameId(g.id); return g; })}>
            New game (standard start)</button>
          <input placeholder="or start from FEN" value={newFen} onChange={(e) => setNewFen(e.target.value)} style={{ width: 300 }} />
          <button disabled={!newFen.trim()} onClick={() => act(async () => { const g = await api.createGame({ title: "Manual game (FEN)", start: { kind: "fen", fen: newFen }, moves: [] }); setGameId(g.id); return g; })}>
            New game from FEN</button>
        </div>
      </section>
      {game && (
        <section className="panel">
          <h3>{game.title ?? "Untitled"} <span className="muted small">({game.source})</span></h3>
          <div className="small">Start FEN: <code className="fen">{game.start_fen}</code></div>
          <ProvenanceTable meta={game.start_metadata} />
          {game.start_assumptions.length > 0 && <ul className="warn small">{game.start_assumptions.map((a) => <li key={a}>{a}</li>)}</ul>}
          {game.revalidation && game.revalidation.detached_now.length > 0 && (
            <div className="warn">Edit revalidated the game: {game.revalidation.detached_now.map((d) => `${d.uci} (${d.reason})`).join("; ")}</div>
          )}
          <table className="moves">
            <thead><tr><th>#</th><th>SAN</th><th>UCI</th><th>Coordinates</th><th>Description</th><th>Flags</th><th>Source</th><th>Board settled</th><th /><th /></tr></thead>
            <tbody>
              {game.moves.map((m) => (
                <tr key={m.id}>
                  <td>{m.move_number}{m.color === "black" ? "…" : "."}</td>
                  <td><b>{m.san}</b></td>
                  <td><code>{m.uci}</code></td>
                  <td>{m.coordinates}</td>
                  <td className="small">{m.english}</td>
                  <td><Flags d={m} /></td>
                  <td className="small">{m.provenance}{m.has_evidence ? " (evidence)" : ""}</td>
                  <td className="small">{m.t_settled !== null ? fmtTime(m.t_settled) : "–"}</td>
                  <td>{m.thumbnail_url && <img src={m.thumbnail_url} width={72} alt="" />}</td>
                  <td>
                    {edit?.ply === m.ply ? (
                      <>
                        <input value={edit.uci} onChange={(e) => setEdit({ ...edit, uci: e.target.value })} style={{ width: 70 }} />
                        <button onClick={() => act(async () => { const g = await api.editMove(game.id, m.ply, edit.uci.trim()); setEdit(null); return g; })}>Save</button>
                        <button onClick={() => setEdit(null)}>×</button>
                      </>
                    ) : (
                      <>
                        <button onClick={() => setEdit({ ply: m.ply, uci: m.uci })}>Edit</button>
                        <button onClick={() => act(() => api.deleteMove(game.id, m.ply))}>Delete</button>
                        <button onClick={() => api.exportFen(game.id, m.ply).then(setFenInfo).catch((e) => setError(e.message))}>FEN</button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">Editing or deleting a move replays everything after it. Later moves that no longer follow
            (illegal, or their stored evidence no longer confirms them) are detached, never patched over.
            Video times mark when the board settled after the move, not when the piece was touched.</p>
          <div className="row">
            <input placeholder="append move (UCI, e.g. e2e4 or e7e8q)" value={append} onChange={(e) => setAppend(e.target.value)} />
            <button disabled={!append.trim()} onClick={() => act(async () => { const g = await api.appendMove(game.id, append.trim()); setAppend(""); return g; })}>Add move</button>
          </div>
          {game.detached.length > 0 && (
            <div className="card">
              <b>Detached moves (need review)</b>
              <ul className="small">{game.detached.map((d) => <li key={d.id}><code>{d.uci}</code> [{d.provenance}] — {d.reason}</li>)}</ul>
              <button onClick={() => act(() => api.clearDetached(game.id))}>Discard detached moves</button>
            </div>
          )}
          {fenInfo && (
            <div className="card small">FEN: <code className="fen">{fenInfo.fen}</code>
              {fenInfo.assumptions.length > 0 && <ul className="warn">{fenInfo.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>}</div>
          )}
          <div className="card">
            <div className="small">Final FEN: <code className="fen">{game.final_fen}</code></div>
            <div className="row" style={{ marginTop: 6 }}>
              <a href={`/api/games/${game.id}/export/pgn`}>Download PGN</a>
              <a href={`/api/games/${game.id}/export/fen`} target="_blank" rel="noreferrer">FEN + metadata (JSON)</a>
              <a href={`/api/games/${game.id}/export/json`} target="_blank" rel="noreferrer">Full JSON report</a>
            </div>
            <pre className="pgn">{game.pgn}</pre>
          </div>
        </section>
      )}
    </div>
  );
}
