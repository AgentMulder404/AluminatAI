"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  read: boolean;
  created_at: string;
}

const TYPE_ICONS: Record<string, string> = {
  budget_alert: "💰",
  waste_detected: "⚠️",
  agent_offline: "🔴",
  system: "ℹ️",
  team: "👥",
};

export default function NotificationBell() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const fetchNotifications = useCallback(() => {
    setLoading(true);
    fetch("/api/notifications?limit=10")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) {
          setNotifications(data.notifications);
          setUnreadCount(data.unread_count);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchNotifications();
    const interval = setInterval(fetchNotifications, 30000);
    return () => clearInterval(interval);
  }, [fetchNotifications]);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const markAllRead = () => {
    fetch("/api/notifications", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ all: true }),
    }).then(() => {
      setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
      setUnreadCount(0);
    });
  };

  function timeAgo(dateStr: string): string {
    const ms = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(ms / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => {
          setOpen(!open);
          if (!open) fetchNotifications();
        }}
        className="relative p-2 rounded-lg hover:bg-neutral-800 transition-colors"
        aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ""}`}
        aria-expanded={open}
        aria-haspopup="true"
      >
        <svg
          className="w-5 h-5 text-neutral-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 bg-red-500 text-white text-[10px] font-bold rounded-full h-4 w-4 flex items-center justify-center">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div
          role="menu"
          aria-label="Notifications"
          className="absolute right-0 top-full mt-2 w-80 bg-neutral-900 border border-neutral-800 rounded-xl shadow-2xl z-50 overflow-hidden"
        >
          <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800">
            <h3 className="text-sm font-medium text-neutral-200">Notifications</h3>
            {unreadCount > 0 && (
              <button
                onClick={markAllRead}
                className="text-xs text-indigo-400 hover:text-indigo-300"
              >
                Mark all read
              </button>
            )}
          </div>

          <div className="max-h-80 overflow-y-auto">
            {loading && notifications.length === 0 ? (
              <div className="p-4 space-y-3 animate-pulse">
                <div className="h-12 bg-neutral-800 rounded" />
                <div className="h-12 bg-neutral-800 rounded" />
              </div>
            ) : notifications.length === 0 ? (
              <p className="text-neutral-500 text-sm text-center py-8">
                No notifications yet
              </p>
            ) : (
              notifications.map((n) => (
                <div
                  key={n.id}
                  className={`px-4 py-3 border-b border-neutral-800/50 ${
                    !n.read ? "bg-neutral-800/30" : ""
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <span className="text-sm mt-0.5">
                      {TYPE_ICONS[n.type] ?? "📌"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-neutral-200 truncate">
                        {n.title}
                      </p>
                      <p className="text-xs text-neutral-500 truncate mt-0.5">
                        {n.message}
                      </p>
                      <p className="text-[10px] text-neutral-600 mt-1">
                        {timeAgo(n.created_at)}
                      </p>
                    </div>
                    {!n.read && (
                      <span className="w-2 h-2 rounded-full bg-indigo-500 mt-1.5 shrink-0" />
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
