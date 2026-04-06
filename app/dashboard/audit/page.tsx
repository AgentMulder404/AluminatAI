"use client";

import { useState, useEffect, useCallback } from "react";

interface AuditEntry {
  id: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string | null;
  metadata: Record<string, unknown>;
  ip_address: string | null;
  created_at: string;
}

export default function AuditLogPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [actionFilter, setActionFilter] = useState("");
  const [resourceFilter, setResourceFilter] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const limit = 25;

  const fetchEntries = useCallback(() => {
    setLoading(true);
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (actionFilter) params.set("action", actionFilter);
    if (resourceFilter) params.set("resource_type", resourceFilter);

    fetch(`/api/audit-log?${params}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) {
          setEntries(data.entries);
          setTotal(data.total);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [offset, actionFilter, resourceFilter]);

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6">
      <h1 className="text-xl font-semibold mb-6">Audit Log</h1>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3 mb-4">
        <input
          type="text"
          value={actionFilter}
          onChange={(e) => {
            setActionFilter(e.target.value);
            setOffset(0);
          }}
          placeholder="Filter by action (e.g. budget.create)"
          className="px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 w-full sm:w-64"
        />
        <input
          type="text"
          value={resourceFilter}
          onChange={(e) => {
            setResourceFilter(e.target.value);
            setOffset(0);
          }}
          placeholder="Filter by resource type"
          className="px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 w-full sm:w-64"
        />
      </div>

      {/* Table */}
      <div className="border border-neutral-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-neutral-900 text-neutral-400 text-xs">
                <th className="text-left px-4 py-3 font-medium">Time</th>
                <th className="text-left px-4 py-3 font-medium">Action</th>
                <th className="text-left px-4 py-3 font-medium">Resource</th>
                <th className="text-left px-4 py-3 font-medium">IP</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-neutral-500">
                    Loading...
                  </td>
                </tr>
              ) : entries.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-neutral-500">
                    No audit entries found
                  </td>
                </tr>
              ) : (
                entries.map((entry) => (
                  <>
                    <tr
                      key={entry.id}
                      onClick={() =>
                        setExpandedId(expandedId === entry.id ? null : entry.id)
                      }
                      className="border-t border-neutral-800/50 hover:bg-neutral-900/50 cursor-pointer"
                    >
                      <td className="px-4 py-2.5 text-neutral-400 text-xs whitespace-nowrap">
                        {new Date(entry.created_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-2.5 text-neutral-200 font-mono text-xs">
                        {entry.action}
                      </td>
                      <td className="px-4 py-2.5 text-neutral-400 text-xs">
                        {entry.resource_type}
                        {entry.resource_id && (
                          <span className="text-neutral-600 ml-1">
                            {entry.resource_id.slice(0, 8)}...
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-neutral-500 text-xs">
                        {entry.ip_address ?? "—"}
                      </td>
                    </tr>
                    {expandedId === entry.id && (
                      <tr key={`${entry.id}-meta`} className="border-t border-neutral-800/30">
                        <td colSpan={4} className="px-4 py-3 bg-neutral-900/30">
                          <pre className="text-xs text-neutral-400 overflow-x-auto">
                            {JSON.stringify(entry.metadata, null, 2)}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4">
          <p className="text-xs text-neutral-500">
            Page {currentPage} of {totalPages} ({total} entries)
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              className="px-3 py-1 text-xs bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded disabled:opacity-50"
            >
              Previous
            </button>
            <button
              onClick={() => setOffset(offset + limit)}
              disabled={offset + limit >= total}
              className="px-3 py-1 text-xs bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
