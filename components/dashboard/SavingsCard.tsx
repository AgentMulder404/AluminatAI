"use client";

import { useState, useEffect, useCallback } from "react";

interface SavingsData {
  total_savings_usd: number;
  savings_pct: number;
  total_actual_usd: number;
  total_cloud_equivalent_usd: number;
  by_gpu: Array<{
    gpu_name: string;
    savings_usd: number;
    reference_provider: string;
  }>;
}

export default function SavingsCard() {
  const [data, setData] = useState<SavingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchData = useCallback(() => {
    fetch("/api/cost/savings?days=30")
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((d) => {
        setData(d);
        setError(false);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-green-900/50 p-5 animate-pulse">
        <div className="h-4 bg-neutral-800 rounded w-32 mb-3" />
        <div className="h-8 bg-neutral-800 rounded w-20" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
        <p className="text-sm text-neutral-400 mb-1">Savings vs Cloud</p>
        <p className="text-sm text-neutral-500">No data</p>
        <button
          onClick={fetchData}
          className="text-xs text-indigo-400 hover:text-indigo-300 mt-1"
        >
          Retry
        </button>
      </div>
    );
  }

  const savings = data.total_savings_usd;
  const isPositive = savings > 0;

  function fmtUsd(n: number): string {
    if (Math.abs(n) >= 1000) return `$${(n / 1000).toFixed(1)}k`;
    return `$${n.toFixed(2)}`;
  }

  return (
    <div
      className={`rounded-xl border p-5 ${
        isPositive
          ? "bg-green-950/30 border-green-800/50"
          : "bg-neutral-900 border-neutral-800"
      }`}
    >
      <p className="text-sm text-neutral-400 mb-1">Savings vs Cloud (30d)</p>
      <div className="flex items-baseline gap-2">
        <span
          className={`text-2xl font-semibold ${
            isPositive ? "text-green-400" : "text-neutral-300"
          }`}
        >
          {isPositive ? "+" : ""}
          {fmtUsd(savings)}
        </span>
        {data.savings_pct > 0 && (
          <span className="text-xs font-medium text-green-500">
            {Math.round(data.savings_pct)}% less
          </span>
        )}
      </div>
      <p className="text-xs text-neutral-500 mt-2">
        Your cost: {fmtUsd(data.total_actual_usd)} &middot; Cloud equivalent:{" "}
        {fmtUsd(data.total_cloud_equivalent_usd)}
      </p>
    </div>
  );
}
