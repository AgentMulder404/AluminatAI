"use client";

import { useState, useEffect } from "react";

interface OrgMember {
  user_id: string;
  role: string;
  joined_at: string;
}

interface OrgDetail {
  id: string;
  name: string;
  slug: string;
  plan: string;
  created_at: string;
}

export default function OrgManagementPage() {
  const [orgs, setOrgs] = useState<Array<{ org_id: string; name: string; slug: string; role: string }>>([]);
  const [selectedOrg, setSelectedOrg] = useState<OrgDetail | null>(null);
  const [members, setMembers] = useState<OrgMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [inviteUserId, setInviteUserId] = useState("");
  const [inviteRole, setInviteRole] = useState("member");

  useEffect(() => {
    fetch("/api/organizations")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.organizations?.length > 0) {
          setOrgs(data.organizations);
          loadOrg(data.organizations[0].org_id);
        } else {
          setLoading(false);
        }
      })
      .catch(() => setLoading(false));
  }, []);

  const loadOrg = async (orgId: string) => {
    setLoading(true);
    const [orgRes, membersRes] = await Promise.all([
      fetch(`/api/organizations/${orgId}`),
      fetch(`/api/organizations/${orgId}/members`),
    ]);
    if (orgRes.ok) setSelectedOrg(await orgRes.json());
    if (membersRes.ok) {
      const data = await membersRes.json();
      setMembers(data.members ?? []);
    }
    setLoading(false);
  };

  const inviteMember = async () => {
    if (!selectedOrg || !inviteUserId.trim()) return;
    const res = await fetch(`/api/organizations/${selectedOrg.id}/members`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ user_id: inviteUserId.trim(), role: inviteRole }),
    });
    if (res.ok) {
      setInviteUserId("");
      loadOrg(selectedOrg.id);
    }
  };

  const removeMember = async (userId: string) => {
    if (!selectedOrg) return;
    await fetch(`/api/organizations/${selectedOrg.id}/members/${userId}`, {
      method: "DELETE",
    });
    loadOrg(selectedOrg.id);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-neutral-950 p-6 animate-pulse">
        <div className="h-8 bg-neutral-800 rounded w-48 mb-4" />
        <div className="h-48 bg-neutral-800 rounded" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6 max-w-3xl">
      <h1 className="text-xl font-semibold mb-6">Organization Settings</h1>

      {orgs.length === 0 ? (
        <p className="text-neutral-500 text-sm">
          No organizations yet. Create one from the dashboard.
        </p>
      ) : (
        <>
          {/* Org selector */}
          {orgs.length > 1 && (
            <div className="mb-6">
              <select
                value={selectedOrg?.id ?? ""}
                onChange={(e) => loadOrg(e.target.value)}
                className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-200"
              >
                {orgs.map((o) => (
                  <option key={o.org_id} value={o.org_id}>
                    {o.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {selectedOrg && (
            <>
              {/* Org info */}
              <section className="border border-neutral-800 rounded-lg p-5 mb-6">
                <h2 className="text-sm font-semibold text-neutral-300 mb-3">
                  Organization Details
                </h2>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-neutral-500 text-xs">Name</p>
                    <p className="text-neutral-200">{selectedOrg.name}</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-xs">Slug</p>
                    <p className="text-neutral-200 font-mono">{selectedOrg.slug}</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-xs">Plan</p>
                    <p className="text-neutral-200 capitalize">{selectedOrg.plan}</p>
                  </div>
                  <div>
                    <p className="text-neutral-500 text-xs">Created</p>
                    <p className="text-neutral-200">
                      {new Date(selectedOrg.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </section>

              {/* Members */}
              <section className="border border-neutral-800 rounded-lg p-5 mb-6">
                <h2 className="text-sm font-semibold text-neutral-300 mb-3">
                  Members ({members.length})
                </h2>
                <div className="space-y-2 mb-4">
                  {members.map((m) => (
                    <div
                      key={m.user_id}
                      className="flex items-center justify-between bg-neutral-900 rounded-lg px-3 py-2"
                    >
                      <div>
                        <span className="text-sm text-neutral-200 font-mono">
                          {m.user_id.slice(0, 8)}...
                        </span>
                        <span className="text-xs text-neutral-500 ml-2 capitalize">
                          {m.role}
                        </span>
                      </div>
                      {m.role !== "owner" && (
                        <button
                          onClick={() => removeMember(m.user_id)}
                          className="text-xs text-red-400 hover:text-red-300"
                        >
                          Remove
                        </button>
                      )}
                    </div>
                  ))}
                </div>

                {/* Invite */}
                <div className="flex items-end gap-2">
                  <div className="flex-1">
                    <label className="text-xs text-neutral-500 block mb-1">
                      User ID
                    </label>
                    <input
                      type="text"
                      value={inviteUserId}
                      onChange={(e) => setInviteUserId(e.target.value)}
                      placeholder="Paste user ID"
                      className="w-full px-3 py-2 text-sm bg-neutral-900 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600"
                    />
                  </div>
                  <select
                    value={inviteRole}
                    onChange={(e) => setInviteRole(e.target.value)}
                    className="bg-neutral-900 border border-neutral-700 rounded px-3 py-2 text-sm text-neutral-200"
                  >
                    <option value="member">Member</option>
                    <option value="admin">Admin</option>
                  </select>
                  <button
                    onClick={inviteMember}
                    className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded transition"
                  >
                    Invite
                  </button>
                </div>
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}
