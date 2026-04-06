"use client";

import { useState, useEffect, useCallback } from "react";

interface Preferences {
  in_app: boolean;
  email: boolean;
  slack: boolean;
  pagerduty: boolean;
  opsgenie: boolean;
}

const CHANNEL_LABELS: Record<keyof Preferences, { label: string; description: string }> = {
  in_app: {
    label: "In-App Notifications",
    description: "Bell icon notifications in the dashboard",
  },
  email: {
    label: "Email",
    description: "Budget alerts and waste detection via email",
  },
  slack: {
    label: "Slack",
    description: "Messages to configured Slack channels",
  },
  pagerduty: {
    label: "PagerDuty",
    description: "Alerts via PagerDuty Events API",
  },
  opsgenie: {
    label: "OpsGenie",
    description: "Alerts via OpsGenie Alert API",
  },
};

export default function NotificationPreferences() {
  const [prefs, setPrefs] = useState<Preferences | null>(null);
  const [saving, setSaving] = useState(false);

  const fetchPrefs = useCallback(() => {
    fetch("/api/notifications/preferences")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setPrefs(data);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchPrefs();
  }, [fetchPrefs]);

  const toggle = async (channel: keyof Preferences) => {
    if (!prefs) return;
    const newValue = !prefs[channel];
    setPrefs({ ...prefs, [channel]: newValue });
    setSaving(true);

    try {
      await fetch("/api/notifications/preferences", {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ [channel]: newValue }),
      });
    } catch {
      // Revert on failure
      setPrefs({ ...prefs, [channel]: !newValue });
    } finally {
      setSaving(false);
    }
  };

  if (!prefs) {
    return (
      <div className="animate-pulse space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-10 bg-neutral-800 rounded" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {(Object.keys(CHANNEL_LABELS) as (keyof Preferences)[]).map((ch) => (
        <div
          key={ch}
          className="flex items-center justify-between bg-neutral-900 rounded-lg px-4 py-3"
        >
          <div>
            <p className="text-sm text-neutral-200">
              {CHANNEL_LABELS[ch].label}
            </p>
            <p className="text-xs text-neutral-500">
              {CHANNEL_LABELS[ch].description}
            </p>
          </div>
          <button
            onClick={() => toggle(ch)}
            disabled={saving}
            className={`relative w-10 h-5 rounded-full transition-colors ${
              prefs[ch] ? "bg-indigo-600" : "bg-neutral-700"
            }`}
            aria-label={`Toggle ${CHANNEL_LABELS[ch].label}`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                prefs[ch] ? "translate-x-5" : ""
              }`}
            />
          </button>
        </div>
      ))}
    </div>
  );
}
