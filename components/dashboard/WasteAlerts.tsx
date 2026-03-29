"use client";

import { useState, useEffect, useCallback } from "react";

interface WasteEvent {
  id: string;
  gpu_uuid: string;
  gpu_name: string | null;
  job_id: string | null;
  team_id: string | null;
  waste_type: string;
  avg_utilization_pct: number | null;
  duration_hours: number;
  estimated_waste_usd: number;
  detected_at: string;
}

interface WasteAlertsProps {
  clusterParam?: string;
}

const WASTE_LABELS: Record<string, string> = {
  idle_gpu: "Idle GPU",
  low_utilization: "Low Utilization",
  oversized_gpu: "Oversized GPU",
  long_idle_between_jobs: "Long Idle Gap",
};

export default function WasteAlerts({ clusterParam }: WasteAlertsProps) {
  const [events, setEvents] = useState<WasteEvent[]>([]);
  const [totalWaste, setTotalWaste] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchData = useCallback(() => {
    setError(false);
    const params = new URLSearchParams();
    if (clusterParam) params.set("cluster_tag", clusterParam);

    fetch(`/api/cost/waste?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((data) => {
        setEvents(data.events ?? []);
        setTotalWaste(data.total_waste_usd ?? 0);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [clusterParam]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const dismissEvent = async (eventId: string) => {
    await fetch("/api/cost/waste/dismiss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId }),
    });
    setEvents((prev) => prev.filter((e) => e.id !== eventId));
  };

  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5 animate-pulse">
        <div className="h-4 bg-neutral-800 rounded w-32 mb-3" />
        <div className="space-y-2">
          <div className="h-12 bg-neutral-800 rounded" />
          <div className="h-12 bg-neutral-800 rounded" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
        <p className="text-red-400 text-sm">Failed to load waste data.</p>
        <button onClick={fetchData} className="text-xs text-indigo-400 hover:text-indigo-300 mt-1">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-neutral-200">Waste Detected</h3>
        {totalWaste > 0 && (
          <span className="text-xs font-medium text-red-400 bg-red-400/10 px-2 py-0.5 rounded-full">
            ${totalWaste.toFixed(2)} wasted
          </span>
        )}
      </div>

      {events.length === 0 ? (
        <p className="text-neutral-500 text-sm text-center py-6">No waste detected. Nice!</p>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {events.slice(0, 20).map((e) => (
            <div
              key={e.id}
              className="flex items-start justify-between gap-2 bg-neutral-800/50 rounded-lg px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-amber-400">
                    {WASTE_LABELS[e.waste_type] ?? e.waste_type}
                  </span>
                  <span className="text-xs text-neutral-500">
                    {e.gpu_name ?? e.gpu_uuid.slice(0, 8)}
                  </span>
                </div>
                <p className="text-xs text-neutral-400 mt-0.5">
                  {e.avg_utilization_pct != null && `${e.avg_utilization_pct}% avg util · `}
                  {e.duration_hours.toFixed(1)}h · ${e.estimated_waste_usd.toFixed(2)}
                  {e.job_id && ` · Job: ${e.job_id.slice(0, 12)}`}
                </p>
              </div>
              <button
                onClick={() => dismissEvent(e.id)}
                className="text-xs text-neutral-500 hover:text-neutral-300 shrink-0"
                title="Dismiss"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
