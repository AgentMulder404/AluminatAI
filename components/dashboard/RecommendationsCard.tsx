"use client";

import { useState, useEffect } from "react";

interface Recommendation {
  id: string;
  recommendation_type: string;
  title: string;
  description: string;
  estimated_savings_usd: number | null;
  estimated_savings_pct: number | null;
  context: Record<string, unknown>;
}

interface SavingsAlternative {
  gpu_model: string;
  provider: string;
  instance_type: string | null;
  monthly_cost_usd: number;
  monthly_savings_usd: number;
  savings_pct: number;
  is_spot: boolean;
}

interface SavingsEntry {
  current_gpu: string;
  current_monthly_cost_usd: number;
  alternatives: SavingsAlternative[];
}

interface RecommendationsCardProps {
  clusterParam?: string;
}

const TYPE_ICONS: Record<string, string> = {
  time_shift: "🕐",
  gpu_downsize: "📉",
  spot_pricing: "💰",
  consolidation: "🔗",
};

export default function RecommendationsCard({ clusterParam }: RecommendationsCardProps) {
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [savings, setSavings] = useState<SavingsEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [showSavings, setShowSavings] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);

    const params = new URLSearchParams();
    if (clusterParam) params.set("cluster_tag", clusterParam);

    fetch(`/api/cost/recommendations?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((data) => {
        setRecs(data.recommendations ?? []);
        setSavings(data.savings_calculator ?? []);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [clusterParam]);

  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5 animate-pulse">
        <div className="h-4 bg-neutral-800 rounded w-40 mb-3" />
        <div className="h-20 bg-neutral-800 rounded" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
        <p className="text-red-400 text-sm">Failed to load recommendations.</p>
      </div>
    );
  }

  if (recs.length === 0 && savings.length === 0) return null;

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-neutral-200">Recommendations</h3>
        {savings.length > 0 && (
          <button
            onClick={() => setShowSavings(!showSavings)}
            className="text-xs text-indigo-400 hover:text-indigo-300"
          >
            {showSavings ? "Hide" : "Show"} Savings Calculator
          </button>
        )}
      </div>

      {/* Recommendations */}
      {recs.length > 0 && (
        <div className="space-y-2 mb-4">
          {recs.slice(0, 5).map((rec) => (
            <div
              key={rec.id}
              className="bg-neutral-800/50 rounded-lg px-3 py-2"
            >
              <div className="flex items-start gap-2">
                <span className="text-sm">{TYPE_ICONS[rec.recommendation_type] ?? "💡"}</span>
                <div>
                  <p className="text-xs font-medium text-neutral-200">{rec.title}</p>
                  <p className="text-xs text-neutral-400 mt-0.5">{rec.description}</p>
                  {rec.estimated_savings_usd != null && (
                    <span className="text-xs text-green-400 mt-1 inline-block">
                      Save ~${rec.estimated_savings_usd.toFixed(2)}
                      {rec.estimated_savings_pct != null && ` (${rec.estimated_savings_pct}%)`}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Savings Calculator */}
      {showSavings && savings.length > 0 && (
        <div className="border-t border-neutral-800 pt-3 mt-3">
          <h4 className="text-xs font-medium text-neutral-300 mb-2">
            GPU Savings Calculator (30-day window)
          </h4>
          {savings.map((s) => (
            <div key={s.current_gpu} className="mb-3">
              <p className="text-xs text-neutral-400 mb-1">
                <span className="text-neutral-200">{s.current_gpu}</span> —
                ${s.current_monthly_cost_usd.toFixed(2)}/mo
              </p>
              <div className="space-y-1 pl-3">
                {s.alternatives.slice(0, 3).map((alt, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className="text-green-400">
                      ↓ ${alt.monthly_savings_usd.toFixed(2)}/mo ({alt.savings_pct}%)
                    </span>
                    <span className="text-neutral-500">
                      → {alt.gpu_model} on {alt.provider}
                      {alt.is_spot && " (spot)"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
