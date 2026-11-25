import type { Health } from "../api";
import { ImageAnalysisPanel } from "../components/ImageAnalysisPanel";

export function AnalyzePage({ health }: { health: Health | null }) {
  return (
    <div>
      <h2>Analyze image</h2>
      <p className="small">A single image gives piece placement only. Side to move, castling rights, en passant and move
        counters must come from you; anything you leave blank is shown as an assumption.</p>
      <ImageAnalysisPanel title="Board image" health={health} />
    </div>
  );
}
