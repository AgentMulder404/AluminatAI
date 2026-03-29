"use client";

import { useState, useEffect, useCallback } from "react";

type Format = "currency" | "number" | "percent" | "co2";

interface StatCardProps {
  title: string;
  fetchUrl: string;
  valueKey: string;
  format: Format;
  suffix?: string;
  refreshInterval?: number;
}

function formatValue(value: number | null, format: Format, suffix?: string): string {
  if (value == null) return "—";
  switch (format) {
    case "currency":
      return value >= 1000
        ? `$${(value / 1000).toFixed(1)}k`
        : `$${value.toFixed(2)}`;
    case "number":
      return value >= 1000
        ? `${(value / 1000).toFixed(1)}k`
        : `${Math.round(value)}`;
    case "percent":
      return `${Math.round(value)}%`;
    case "co2":
      if (value >= 1000) return `${(value / 1000).toFixed(1)} kg`;
      return `${Math.round(value)} g`;
  }
  return `${value}${suffix ?? ""}`;
}

export default function StatCard({
  title,
  fetchUrl,
  valueKey,
  format,
  suffix,
  refreshInterval = 60000,
}: StatCardProps) {
  const [value, setValue] = useState<number | null>(null);
  const [prevValue, setPrevValue] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchData = useCallback(() => {
    fetch(fetchUrl)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((data) => {
        setPrevValue(value);
        setValue(data[valueKey] ?? null);
        setError(false);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [fetchUrl, valueKey, value]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchUrl, valueKey, refreshInterval]); // eslint-disable-line react-hooks/exhaustive-deps

  const trend =
    prevValue != null && value != null && prevValue !== 0
      ? ((value - prevValue) / Math.abs(prevValue)) * 100
      : null;

  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5 animate-pulse">
        <div className="h-4 bg-neutral-800 rounded w-24 mb-3" />
        <div className="h-8 bg-neutral-800 rounded w-16" />
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <p className="text-sm text-neutral-400 mb-1">{title}</p>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-semibold text-neutral-100">
          {error ? "Error" : formatValue(value, format, suffix)}
        </span>
        {trend != null && !error && (
          <span
            className={`text-xs font-medium ${
              trend > 0 ? "text-red-400" : trend < 0 ? "text-green-400" : "text-neutral-500"
            }`}
          >
            {trend > 0 ? "↑" : trend < 0 ? "↓" : "→"}{" "}
            {Math.abs(Math.round(trend))}%
          </span>
        )}
      </div>
      {error && (
        <button
          onClick={fetchData}
          className="text-xs text-indigo-400 hover:text-indigo-300 mt-1"
        >
          Retry
        </button>
      )}
    </div>
  );
}
