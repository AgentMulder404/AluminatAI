"use client";

import { useState, useEffect, useRef } from "react";

interface Org {
  org_id: string;
  name: string;
  slug: string;
  plan: string;
  role: string;
}

interface OrgSwitcherProps {
  onOrgChange?: (orgId: string | null) => void;
}

export default function OrgSwitcher({ onOrgChange }: OrgSwitcherProps) {
  const [orgs, setOrgs] = useState<Org[]>([]);
  const [activeOrg, setActiveOrg] = useState<Org | null>(null);
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/organizations")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.organizations?.length > 0) {
          setOrgs(data.organizations);
          setActiveOrg(data.organizations[0]);
          onOrgChange?.(data.organizations[0].org_id);
        }
      })
      .catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setCreating(false);
      }
    }
    if (open) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const switchOrg = (org: Org) => {
    setActiveOrg(org);
    onOrgChange?.(org.org_id);
    setOpen(false);
  };

  const createOrg = async () => {
    if (!newName.trim()) return;
    try {
      const res = await fetch("/api/organizations", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: newName.trim() }),
      });
      if (res.ok) {
        const org = await res.json();
        const newOrg: Org = { org_id: org.id, name: org.name, slug: org.slug, plan: "free", role: "owner" };
        setOrgs((prev) => [...prev, newOrg]);
        switchOrg(newOrg);
        setNewName("");
        setCreating(false);
      }
    } catch {
      // Ignore
    }
  };

  if (orgs.length === 0) return null;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition text-sm"
        aria-label={`Organization: ${activeOrg?.name ?? "Select Org"}`}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className="text-neutral-200 font-medium truncate max-w-[120px]">
          {activeOrg?.name ?? "Select Org"}
        </span>
        <svg
          className={`w-3 h-3 text-neutral-500 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div
          role="listbox"
          aria-label="Organizations"
          className="absolute left-0 top-full mt-1 w-56 bg-neutral-900 border border-neutral-800 rounded-xl shadow-2xl z-50 overflow-hidden"
        >
          {orgs.map((org) => (
            <button
              key={org.org_id}
              role="option"
              aria-selected={activeOrg?.org_id === org.org_id}
              onClick={() => switchOrg(org)}
              className={`w-full text-left px-4 py-2.5 text-sm hover:bg-neutral-800 transition ${
                activeOrg?.org_id === org.org_id ? "bg-neutral-800/50" : ""
              }`}
            >
              <span className="text-neutral-200">{org.name}</span>
              <span className="text-xs text-neutral-600 ml-2">{org.role}</span>
            </button>
          ))}

          <div className="border-t border-neutral-800">
            {creating ? (
              <div className="p-3">
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && createOrg()}
                  placeholder="Organization name"
                  className="w-full px-3 py-1.5 text-sm bg-neutral-800 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600"
                  autoFocus
                />
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={createOrg}
                    className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1 rounded transition"
                  >
                    Create
                  </button>
                  <button
                    onClick={() => setCreating(false)}
                    className="text-xs text-neutral-400 hover:text-neutral-200 px-3 py-1"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setCreating(true)}
                className="w-full text-left px-4 py-2.5 text-sm text-indigo-400 hover:bg-neutral-800 transition"
              >
                + Create Organization
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
