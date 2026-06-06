"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import CarbonExportButton from "@/components/dashboard/CarbonExportButton";
import NotificationPreferences from "@/components/dashboard/NotificationPreferences";
import { PLAN_DISPLAY, type PlanTier } from "@/lib/plans";

export default function SettingsPage() {
  const [optIn, setOptIn] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [gridZone, setGridZone] = useState<string | null>(null);
  const [carbonGPerKwh, setCarbonGPerKwh] = useState<number | null>(null);
  const [apiKeyMasked, setApiKeyMasked] = useState<string | null>(null);
  const [apiKeyFull, setApiKeyFull] = useState<string | null>(null);
  const [keyCopied, setKeyCopied] = useState(false);
  const [keyVisible, setKeyVisible] = useState(false);
  const [keyRevealing, setKeyRevealing] = useState(false);

  // Billing state
  const [plan, setPlan] = useState<PlanTier>("free");
  const [periodEnd, setPeriodEnd] = useState<string | null>(null);
  const [cancelAtPeriodEnd, setCancelAtPeriodEnd] = useState(false);
  const [portalLoading, setPortalLoading] = useState(false);

  useEffect(() => {
    fetch("/api/user/profile")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d) {
          setOptIn(Boolean(d.benchmark_opt_in));
          if (d.api_key_masked) setApiKeyMasked(d.api_key_masked);
          if (d.plan) setPlan(d.plan as PlanTier);
          if (d.plan_period_end) setPeriodEnd(d.plan_period_end);
          if (d.plan_cancel_at_period_end) setCancelAtPeriodEnd(true);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));

    // Check for billing callback
    const params = new URLSearchParams(window.location.search);
    if (params.get("billing") === "success") {
      // Refresh to get updated plan
      setTimeout(() => window.location.replace("/dashboard/settings"), 1500);
    }
  }, []);

  async function revealKey() {
    if (apiKeyFull) {
      setKeyVisible((v) => !v);
      return;
    }
    setKeyRevealing(true);
    try {
      const res = await fetch("/api/user/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reveal_key" }),
      });
      const d = await res.json();
      if (d.api_key) {
        setApiKeyFull(d.api_key);
        setKeyVisible(true);
      }
    } catch {
      /* ignore */
    } finally {
      setKeyRevealing(false);
    }
  }

  function copyKey() {
    const key = apiKeyFull || apiKeyMasked;
    if (!key) return;
    // If we don't have the full key yet, fetch it first
    if (!apiKeyFull) {
      fetch("/api/user/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "reveal_key" }),
      })
        .then((r) => r.json())
        .then((d) => {
          if (d.api_key) {
            setApiKeyFull(d.api_key);
            navigator.clipboard.writeText(d.api_key).then(() => {
              setKeyCopied(true);
              setTimeout(() => setKeyCopied(false), 2000);
            });
          }
        })
        .catch(() => {});
      return;
    }
    navigator.clipboard.writeText(apiKeyFull).then(() => {
      setKeyCopied(true);
      setTimeout(() => setKeyCopied(false), 2000);
    });
  }

  useEffect(() => {
    // Fetch today-cost to get the grid_zone in use, then look up current intensity
    fetch("/api/dashboard/today-cost")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.grid_zone) {
          setGridZone(d.grid_zone);
          // Fetch live intensity for that zone
          return fetch(`/api/carbon/intensity?zone=${encodeURIComponent(d.grid_zone)}`);
        }
        return null;
      })
      .then((r) => (r?.ok ? r.json() : null))
      .then((d) => {
        if (d?.carbon_g_per_kwh != null) setCarbonGPerKwh(d.carbon_g_per_kwh);
      })
      .catch(() => {});
  }, []);

  async function handleToggle() {
    const next = !optIn;
    setOptIn(next);
    setSaving(true);
    setSaved(false);
    try {
      await fetch("/api/user/profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ benchmark_opt_in: next }),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      setOptIn(!next); // revert
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6">
      <h1 className="text-lg font-semibold mb-6">Settings</h1>

      {/* Billing & Plan */}
      <section className="border border-neutral-800 rounded-lg p-5 max-w-lg mb-4">
        <h2 className="text-sm font-semibold text-neutral-300 mb-1">Plan & Billing</h2>
        <p className="text-xs text-neutral-500 mb-4">
          Manage your subscription and billing details.
        </p>

        {loading ? (
          <div className="h-8 w-full bg-neutral-800 rounded animate-pulse" />
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded ${
                  plan === "enterprise" ? "bg-purple-900 text-purple-300" :
                  plan === "team" ? "bg-blue-900 text-blue-300" :
                  "bg-neutral-800 text-neutral-400"
                }`}>
                  {PLAN_DISPLAY[plan].name}
                </span>
                {cancelAtPeriodEnd && (
                  <span className="ml-2 text-xs text-amber-400">Cancels at period end</span>
                )}
              </div>
              {plan !== "free" && periodEnd && (
                <span className="text-xs text-neutral-500">
                  {cancelAtPeriodEnd ? "Expires" : "Renews"}{" "}
                  {new Date(periodEnd).toLocaleDateString()}
                </span>
              )}
            </div>

            <div className="flex gap-2">
              {plan === "free" ? (
                <Link
                  href="/pricing"
                  className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded transition"
                >
                  Upgrade Plan
                </Link>
              ) : (
                <button
                  onClick={async () => {
                    setPortalLoading(true);
                    try {
                      const res = await fetch("/api/stripe/portal", {
                        method: "POST",
                      });
                      const data = await res.json();
                      if (data.url) window.location.href = data.url;
                      else alert(data.error || "Failed to open billing portal");
                    } catch {
                      alert("Something went wrong");
                    } finally {
                      setPortalLoading(false);
                    }
                  }}
                  disabled={portalLoading}
                  className="text-xs bg-neutral-800 hover:bg-neutral-700 text-neutral-200 px-4 py-2 rounded transition disabled:opacity-50"
                >
                  {portalLoading ? "Opening..." : "Manage Billing"}
                </button>
              )}
              <Link
                href="/pricing"
                className="text-xs text-neutral-400 hover:text-neutral-200 px-4 py-2 border border-neutral-700 rounded transition"
              >
                View Plans
              </Link>
            </div>
          </div>
        )}
      </section>

      {/* API Key section */}
      <section className="border border-neutral-800 rounded-lg p-5 max-w-lg mb-4">
        <h2 className="text-sm font-semibold text-neutral-300 mb-1">API Key</h2>
        <p className="text-xs text-neutral-500 mb-4">
          Use this key with the <code className="text-neutral-400">ALUMINATAI_API_KEY</code> environment variable when running the agent.
        </p>
        {loading ? (
          <div className="h-8 w-full bg-neutral-800 rounded animate-pulse" />
        ) : apiKeyMasked ? (
          <div className="flex items-center gap-2">
            <code className="flex-1 bg-neutral-900 border border-neutral-700 rounded px-3 py-2 text-xs text-neutral-200 font-mono truncate">
              {keyVisible && apiKeyFull ? apiKeyFull : apiKeyMasked}
            </code>
            <button
              onClick={revealKey}
              disabled={keyRevealing}
              className="text-xs text-neutral-400 hover:text-neutral-200 px-2 py-2 border border-neutral-700 rounded disabled:opacity-50"
            >
              {keyRevealing ? "..." : keyVisible ? "Hide" : "Show"}
            </button>
            <button
              onClick={copyKey}
              className="text-xs text-neutral-400 hover:text-neutral-200 px-2 py-2 border border-neutral-700 rounded"
            >
              {keyCopied ? "Copied!" : "Copy"}
            </button>
          </div>
        ) : (
          <p className="text-xs text-neutral-500">No API key found. Contact support.</p>
        )}
      </section>

      {/* Benchmarking section */}
      <section className="border border-neutral-800 rounded-lg p-5 max-w-lg">
        <h2 className="text-sm font-semibold text-neutral-300 mb-1">
          Benchmarking
        </h2>
        <p className="text-xs text-neutral-500 mb-4">
          Share anonymized GPU efficiency data to see how you rank vs peers.
          Your user ID is never stored — only a one-way hash is used.
        </p>

        <label className="flex items-center gap-3 cursor-pointer select-none">
          <button
            type="button"
            role="switch"
            aria-checked={optIn}
            disabled={loading || saving}
            onClick={handleToggle}
            className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full border-2 border-transparent transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-neutral-950 ${
              optIn ? "bg-green-500" : "bg-neutral-700"
            } ${loading || saving ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition-transform ${
                optIn ? "translate-x-4" : "translate-x-0"
              }`}
            />
          </button>
          <span className="text-sm text-neutral-300">
            {optIn
              ? "Sharing efficiency data (opt-in active)"
              : "Not sharing efficiency data"}
          </span>
        </label>

        {saved && (
          <p className="mt-2 text-xs text-green-400">Saved.</p>
        )}
      </section>

      {/* Notification Preferences */}
      <section className="border border-neutral-800 rounded-lg p-5 max-w-lg mt-4">
        <h2 className="text-sm font-semibold text-neutral-300 mb-1">
          Notification Preferences
        </h2>
        <p className="text-xs text-neutral-500 mb-4">
          Choose which channels receive budget alerts, waste detection, and other notifications.
        </p>
        <NotificationPreferences />
      </section>

      {/* Carbon tracking section */}
      <section className="border border-neutral-800 rounded-lg p-5 max-w-lg mt-4">
        <h2 className="text-sm font-semibold text-neutral-300 mb-1">Carbon Tracking</h2>
        <p className="text-xs text-neutral-500 mb-4">
          Real-time CO₂e stamped on every GPU metric using Electricity Maps carbon intensity data.
          Set <code className="text-neutral-400">ALUMINATAI_GRID_ZONE</code> in your agent environment to enable.
        </p>

        {gridZone ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-green-400" />
              <span className="text-sm text-neutral-300 font-mono">{gridZone}</span>
              <span className="text-xs text-neutral-500">active zone</span>
            </div>
            {carbonGPerKwh != null && (
              <p className="text-xs text-neutral-400">
                Current intensity:{" "}
                <span className="text-neutral-200 font-medium">{carbonGPerKwh.toFixed(0)} gCO₂eq/kWh</span>
              </p>
            )}
            <Link href="/carbon" className="text-xs text-green-400 hover:underline">
              View carbon leaderboard →
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-neutral-600" />
              <span className="text-sm text-neutral-500">No grid zone configured</span>
            </div>
            <p className="text-xs text-neutral-600">
              Add <code className="text-neutral-500">ALUMINATAI_GRID_ZONE=US-CAL-CISO</code> to your agent .env file.{" "}
              <a
                href="https://www.electricitymaps.com/map"
                target="_blank"
                rel="noopener noreferrer"
                className="text-neutral-400 hover:underline"
              >
                Find your zone →
              </a>
            </p>
          </div>
        )}
      </section>

      {/* Carbon report export */}
      <div className="mt-4 max-w-lg">
        <CarbonExportButton />
      </div>
    </div>
  );
}
