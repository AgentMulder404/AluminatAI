"use client";

import { useState, useEffect, useCallback, useRef } from "react";

/* ── Types ── */
interface Prospect {
  title: string;
  url: string;
  description: string;
  snippet: string;
  category: string;
  query: string;
  status: "new" | "contacted" | "replied" | "not_interested";
  notes: string;
  savedAt?: string;
}

interface RunInfo {
  runId: string;
  datasetId: string;
  keyword?: string;
  domain?: string;
  companyName?: string;
  linkedinUrl?: string;
  industry?: string;
  employeeCount?: number;
  companyDescription?: string;
  location?: string;
}

interface AssembledProspect {
  company_name: string;
  company_url?: string;
  linkedin_url?: string;
  domain?: string;
  industry?: string;
  employee_count?: number;
  location?: string;
  description?: string;
  contact_name?: string;
  contact_email?: string;
  contact_title?: string;
  contact_phone?: string;
  contact_linkedin?: string;
  email_verified?: boolean;
  email_status?: string;
  source_query?: string;
  category?: string;
}

/* ── Constants ── */
const CATEGORIES = [
  { value: "ai-startups", label: "AI Startups" },
  { value: "ml-teams", label: "ML Engineering Teams" },
  { value: "data-centers", label: "Data Centers" },
  { value: "research-labs", label: "Research Labs" },
  { value: "cloud-users", label: "Cloud GPU Users" },
  { value: "custom", label: "Custom Search" },
];

const STATUS_COLORS: Record<string, string> = {
  new: "bg-blue-500/20 text-blue-400",
  contacted: "bg-yellow-500/20 text-yellow-400",
  replied: "bg-green-500/20 text-green-400",
  not_interested: "bg-neutral-500/20 text-neutral-500",
};

const STORAGE_KEY = "alum_outreach_prospects_v1";

const DEFAULT_KEYWORDS = [
  "GPU cloud",
  "AI infrastructure",
  "machine learning",
  "deep learning infrastructure",
  "AI startup GPU",
];

const COMPANY_SIZES = [
  { value: "1-10", label: "1-10" },
  { value: "11-50", label: "11-50" },
  { value: "51-200", label: "51-200" },
  { value: "201-500", label: "201-500" },
  { value: "501-1000", label: "501-1K" },
  { value: "1001-5000", label: "1K-5K" },
  { value: "5001-10000", label: "5K-10K" },
  { value: "10001+", label: "10K+" },
];

type PipelineStep = "idle" | "discovering" | "enriching" | "verifying" | "loading" | "done";

async function safeJson(res: Response): Promise<Record<string, unknown>> {
  const text = await res.text();
  if (!text || text.trim() === "") throw new Error("Empty response from server");
  return JSON.parse(text);
}

/* ── Pipeline Tab Component ── */
function PipelineTab() {
  const [step, setStep] = useState<PipelineStep>("idle");
  const [keywords, setKeywords] = useState<string[]>([...DEFAULT_KEYWORDS]);
  const [newKeyword, setNewKeyword] = useState("");
  const [selectedSizes, setSelectedSizes] = useState<string[]>(["11-50", "51-200", "201-500", "501-1000"]);
  const [locationInput, setLocationInput] = useState("");
  const [maxItems, setMaxItems] = useState(20);

  const [discoverRuns, setDiscoverRuns] = useState<RunInfo[]>([]);
  const [discoverStatus, setDiscoverStatus] = useState<Record<string, string>>({});
  const [enrichRuns, setEnrichRuns] = useState<RunInfo[]>([]);
  const [enrichStatus, setEnrichStatus] = useState<Record<string, string>>({});
  const [verifyRun, setVerifyRun] = useState<RunInfo | null>(null);
  const [verifyStatus, setVerifyStatus] = useState("");

  const [assembledProspects, setAssembledProspects] = useState<AssembledProspect[]>([]);
  const [loadResult, setLoadResult] = useState<{ inserted: number; total: number } | null>(null);
  const [error, setError] = useState("");
  const [log, setLog] = useState<string[]>([]);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  function toggleSize(size: string) {
    setSelectedSizes((prev) =>
      prev.includes(size) ? prev.filter((s) => s !== size) : [...prev, size]
    );
  }

  function addKeyword() {
    const k = newKeyword.trim();
    if (k && !keywords.includes(k)) {
      setKeywords((prev) => [...prev, k]);
      setNewKeyword("");
    }
  }

  function removeKeyword(k: string) {
    setKeywords((prev) => prev.filter((x) => x !== k));
  }

  async function pollRuns(
    runs: RunInfo[],
    setStatusMap: React.Dispatch<React.SetStateAction<Record<string, string>>>
  ): Promise<RunInfo[]> {
    return new Promise((resolve) => {
      const completedDatasets: RunInfo[] = [];
      const statusMap: Record<string, string> = {};
      runs.forEach((r) => (statusMap[r.runId] = "RUNNING"));
      setStatusMap({ ...statusMap });

      const interval = setInterval(async () => {
        let allDone = true;
        for (const run of runs) {
          if (statusMap[run.runId] === "SUCCEEDED" || statusMap[run.runId] === "FAILED") continue;
          try {
            const res = await fetch(`/api/admin/prospect-agents/status?runId=${run.runId}`);
            if (!res.ok) {
              statusMap[run.runId] = "ERROR";
              continue;
            }
            const data = await safeJson(res);
            statusMap[run.runId] = data.status as string;
            if (data.status === "SUCCEEDED") {
              completedDatasets.push({ ...run, datasetId: data.datasetId as string });
            }
          } catch {
            statusMap[run.runId] = "ERROR";
          }
          if (statusMap[run.runId] !== "SUCCEEDED" && statusMap[run.runId] !== "FAILED" && statusMap[run.runId] !== "ERROR") {
            allDone = false;
          }
        }
        setStatusMap({ ...statusMap });
        if (allDone) {
          clearInterval(interval);
          resolve(completedDatasets);
        }
      }, 5000);
      pollRef.current = interval;
    });
  }

  async function runFullPipeline() {
    setError("");
    setLog([]);
    setAssembledProspects([]);
    setLoadResult(null);

    // Step 1: Discover
    setStep("discovering");
    addLog(`Starting discovery with ${keywords.length} keywords...`);
    try {
      const res = await fetch("/api/admin/prospect-agents/discover", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          keywords,
          companySize: selectedSizes,
          locations: locationInput ? [locationInput] : [],
          maxItems,
        }),
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { msg = (await res.json()).error || msg; } catch { /* non-JSON response */ }
        throw new Error(msg);
      }
      const data = await safeJson(res);
      const runs = (data.runs ?? []) as RunInfo[];
      setDiscoverRuns(runs);
      addLog(`Started ${runs.length} discovery runs. Polling...`);

      const completedDiscovery = await pollRuns(runs, setDiscoverStatus);
      addLog(`Discovery complete. ${completedDiscovery.length}/${runs.length} succeeded.`);

      if (completedDiscovery.length === 0) {
        setError("All discovery runs failed");
        setStep("idle");
        return;
      }

      // Step 2: Enrich
      setStep("enriching");
      addLog("Starting enrichment — finding decision makers...");
      const enrichRes = await fetch("/api/admin/prospect-agents/enrich", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasetIds: completedDiscovery.map((r) => r.datasetId),
          maxLeads: 3,
        }),
      });
      if (!enrichRes.ok) {
        let msg = "Enrichment failed";
        try { msg = (await enrichRes.json()).error || msg; } catch { /* non-JSON */ }
        throw new Error(msg);
      }
      const enrichData = await safeJson(enrichRes);
      const enrichRunsList = (enrichData.enrichRuns ?? []) as RunInfo[];
      setEnrichRuns(enrichRunsList);
      addLog(`Found ${enrichData.companiesFound ?? 0} companies. Started ${enrichRunsList.length} enrichment runs.`);

      if (enrichRunsList.length === 0) {
        addLog("No companies with valid domains found. Pipeline complete.");
        setStep("done");
        return;
      }

      const completedEnrich = await pollRuns(enrichRunsList, setEnrichStatus);
      addLog(`Enrichment complete. ${completedEnrich.length} succeeded.`);

      // Fetch contacts from each completed enrichment run
      const assembled: AssembledProspect[] = [];
      for (const run of completedEnrich) {
        const meta = enrichRunsList.find((r: RunInfo) => r.runId === run.runId);
        if (!meta) continue;

        try {
          const contactsRes = await fetch(
            `/api/admin/prospect-agents/enrich?datasetId=${run.datasetId}`
          );
          if (!contactsRes.ok) continue;
          const contactsData = await safeJson(contactsRes);
          const contacts = (contactsData.contacts ?? []) as Record<string, string>[];

          if (contacts && contacts.length > 0) {
            for (const c of contacts) {
              assembled.push({
                company_name: meta.companyName || meta.domain || "Unknown",
                company_url: meta.website,
                linkedin_url: meta.linkedinUrl,
                domain: meta.domain,
                industry: meta.industry,
                employee_count: meta.employeeCount,
                location: meta.location,
                description: meta.companyDescription,
                contact_name: c["01_Name"] || undefined,
                contact_email: c["04_Email"] || undefined,
                contact_title: c["07_Title"] || undefined,
                contact_phone: c["05_Phone_number"] || undefined,
                contact_linkedin: c["06_Linkedin_url"] || undefined,
                source_query: keywords.join(", "),
              });
            }
          } else {
            assembled.push({
              company_name: meta.companyName || meta.domain || "Unknown",
              company_url: meta.website,
              linkedin_url: meta.linkedinUrl,
              domain: meta.domain,
              industry: meta.industry,
              employee_count: meta.employeeCount,
              location: meta.location,
              description: meta.companyDescription,
              source_query: keywords.join(", "),
            });
          }
        } catch {
          assembled.push({
            company_name: meta.companyName || meta.domain || "Unknown",
            linkedin_url: meta.linkedinUrl,
            domain: meta.domain,
            industry: meta.industry,
            source_query: keywords.join(", "),
          });
        }
      }

      setAssembledProspects(assembled);
      addLog(`Assembled ${assembled.length} prospects from ${completedEnrich.length} companies.`);

      // Step 3: Verify emails
      const allEmails = assembled.filter((p) => p.contact_email).map((p) => p.contact_email!);

      if (allEmails.length > 0) {
        setStep("verifying");
        addLog(`Verifying ${allEmails.length} emails...`);
        const verifyRes = await fetch("/api/admin/prospect-agents/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ emails: allEmails }),
        });
        if (verifyRes.ok) {
          const vData = await safeJson(verifyRes) as unknown as RunInfo;
          setVerifyRun(vData);
          const verifyCompleted = await pollRuns([vData], (m) =>
            setVerifyStatus(Object.values(m)[0] || "RUNNING")
          );
          if (verifyCompleted.length > 0) {
            const vResultsRes = await fetch(
              `/api/admin/prospect-agents/enrich?datasetId=${verifyCompleted[0].datasetId}`
            );
            if (vResultsRes.ok) {
              const vResultsData = await safeJson(vResultsRes);
              const vResults = (vResultsData.contacts ?? []) as Array<{ Validation: string; Email: string }>;
              const verifiedSet = new Set(
                vResults
                  .filter((v) => v.Validation === "Valid")
                  .map((v) => v.Email)
              );
              for (const p of assembled) {
                if (p.contact_email) {
                  p.email_verified = verifiedSet.has(p.contact_email);
                  p.email_status = verifiedSet.has(p.contact_email) ? "valid" : "unverified";
                }
              }
              addLog(`Email verification complete. ${verifiedSet.size}/${allEmails.length} valid.`);
            }
          }
        }
      } else {
        addLog("No emails found to verify — skipping verification step.");
      }

      // Step 4: Load
      setStep("loading");
      addLog(`Loading ${assembled.length} prospects into CRM...`);
      const loadRes = await fetch("/api/admin/prospect-agents/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prospects: assembled }),
      });
      if (!loadRes.ok) {
        let msg = "Load failed";
        try { msg = (await loadRes.json()).error || msg; } catch { /* non-JSON */ }
        throw new Error(msg);
      }
      const loadData = await safeJson(loadRes);
      setLoadResult({ inserted: loadData.inserted as number, total: loadData.total as number });
      addLog(`Loaded ${loadData.inserted} prospects into database.`);
      setStep("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Pipeline failed");
      addLog(`ERROR: ${e instanceof Error ? e.message : "Unknown error"}`);
      setStep("idle");
    }
  }

  const isRunning = step !== "idle" && step !== "done";

  return (
    <div className="space-y-6">
      {/* Pipeline Steps Indicator */}
      <div className="flex items-center gap-2 mb-6">
        {(["discovering", "enriching", "verifying", "loading"] as const).map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                step === s
                  ? "bg-green-600 text-white animate-pulse"
                  : step === "done" || (["discovering", "enriching", "verifying", "loading"].indexOf(step) > i)
                  ? "bg-green-600/30 text-green-400"
                  : "bg-neutral-800 text-neutral-500"
              }`}
            >
              <span className="w-5 h-5 flex items-center justify-center rounded-full bg-black/30 text-xs">
                {i + 1}
              </span>
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </div>
            {i < 3 && <div className="w-8 h-px bg-neutral-700" />}
          </div>
        ))}
      </div>

      {/* Config Panel */}
      <div className="bg-neutral-900 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Discovery Configuration</h3>

        {/* Keywords */}
        <div className="mb-4">
          <label className="block text-sm text-neutral-400 mb-2">Search Keywords</label>
          <div className="flex flex-wrap gap-2 mb-2">
            {keywords.map((k) => (
              <span
                key={k}
                className="flex items-center gap-1 bg-neutral-800 text-neutral-300 px-3 py-1 rounded-full text-sm"
              >
                {k}
                <button
                  onClick={() => removeKeyword(k)}
                  disabled={isRunning}
                  className="text-neutral-500 hover:text-red-400 ml-1"
                >
                  x
                </button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={newKeyword}
              onChange={(e) => setNewKeyword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addKeyword()}
              placeholder="Add keyword..."
              disabled={isRunning}
              className="flex-1 bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
            />
            <button
              onClick={addKeyword}
              disabled={isRunning || !newKeyword.trim()}
              className="px-4 py-2 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 rounded-lg text-sm"
            >
              Add
            </button>
          </div>
        </div>

        {/* Company Size */}
        <div className="mb-4">
          <label className="block text-sm text-neutral-400 mb-2">Company Size</label>
          <div className="flex flex-wrap gap-2">
            {COMPANY_SIZES.map((s) => (
              <button
                key={s.value}
                onClick={() => toggleSize(s.value)}
                disabled={isRunning}
                className={`px-3 py-1 rounded-full text-sm ${
                  selectedSizes.includes(s.value)
                    ? "bg-green-600 text-white"
                    : "bg-neutral-800 text-neutral-400 hover:text-white"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Location + Max */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm text-neutral-400 mb-1">Location (optional)</label>
            <input
              type="text"
              value={locationInput}
              onChange={(e) => setLocationInput(e.target.value)}
              placeholder="e.g. United States"
              disabled={isRunning}
              className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
            />
          </div>
          <div>
            <label className="block text-sm text-neutral-400 mb-1">Max companies per keyword</label>
            <input
              type="number"
              min={5}
              max={50}
              value={maxItems}
              onChange={(e) => setMaxItems(Number(e.target.value))}
              disabled={isRunning}
              className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
            />
          </div>
        </div>

        <button
          onClick={runFullPipeline}
          disabled={isRunning || keywords.length === 0}
          className="px-6 py-3 bg-green-600 hover:bg-green-500 disabled:bg-neutral-700 disabled:text-neutral-500 rounded-lg font-medium transition-colors text-lg"
        >
          {isRunning ? `Running — ${step}...` : "Run Full Pipeline"}
        </button>
      </div>

      {/* Run Status */}
      {discoverRuns.length > 0 && (
        <div className="bg-neutral-900 rounded-lg p-4">
          <h4 className="font-medium mb-2">Discovery Runs</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {discoverRuns.map((r) => (
              <div key={r.runId} className="flex items-center gap-2 text-sm bg-neutral-800 rounded px-3 py-2">
                <span className={`w-2 h-2 rounded-full ${
                  discoverStatus[r.runId] === "SUCCEEDED" ? "bg-green-500" :
                  discoverStatus[r.runId] === "FAILED" ? "bg-red-500" :
                  "bg-yellow-500 animate-pulse"
                }`} />
                <span className="text-neutral-300 truncate">{r.keyword}</span>
                <span className="text-neutral-500 text-xs ml-auto">{discoverStatus[r.runId] || "queued"}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {enrichRuns.length > 0 && (
        <div className="bg-neutral-900 rounded-lg p-4">
          <h4 className="font-medium mb-2">Enrichment Runs ({enrichRuns.length} companies)</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {enrichRuns.map((r) => (
              <div key={r.runId} className="flex items-center gap-2 text-sm bg-neutral-800 rounded px-3 py-2">
                <span className={`w-2 h-2 rounded-full ${
                  enrichStatus[r.runId] === "SUCCEEDED" ? "bg-green-500" :
                  enrichStatus[r.runId] === "FAILED" ? "bg-red-500" :
                  "bg-yellow-500 animate-pulse"
                }`} />
                <span className="text-neutral-300 truncate">{r.companyName || r.domain}</span>
                <span className="text-neutral-500 text-xs ml-auto">{enrichStatus[r.runId] || "queued"}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {loadResult && (
        <div className="bg-green-500/10 border border-green-500/30 text-green-400 rounded-lg p-4">
          Pipeline complete! Loaded {loadResult.inserted} new prospects into the CRM ({loadResult.total} total processed).
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg p-4">
          {error}
        </div>
      )}

      {/* Log */}
      {log.length > 0 && (
        <div className="bg-neutral-900 rounded-lg p-4">
          <h4 className="font-medium mb-2 text-neutral-400">Pipeline Log</h4>
          <div className="bg-black rounded p-3 font-mono text-xs text-neutral-400 max-h-60 overflow-y-auto space-y-1">
            {log.map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── CRM Database Tab ── */
interface DbProspect {
  id: number;
  company_name: string;
  company_url?: string;
  domain?: string;
  industry?: string;
  company_size?: string;
  category?: string;
  description?: string;
  notes?: string;
  status?: string;
  contact_name?: string;
  contact_email?: string;
  contact_title?: string;
  location?: string;
  discovered_at?: string;
}

function CrmTab() {
  const [prospects, setProspects] = useState<DbProspect[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loadingCrm, setLoadingCrm] = useState(true);
  const [crmError, setCrmError] = useState("");
  const [search, setSearch] = useState("");
  const [filterCat, setFilterCat] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");
  const [filterSize, setFilterSize] = useState("all");
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const fetchProspects = useCallback(async () => {
    setLoadingCrm(true);
    setCrmError("");
    try {
      const params = new URLSearchParams({ limit: "500" });
      if (filterStatus !== "all") params.set("status", filterStatus);
      const res = await fetch(`/api/admin/prospect-agents/load?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setProspects(data.prospects ?? []);
      setTotalCount(data.count ?? 0);
    } catch (e: unknown) {
      setCrmError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoadingCrm(false);
    }
  }, [filterStatus]);

  useEffect(() => { fetchProspects(); }, [fetchProspects]);

  const filtered = prospects.filter((p) => {
    if (filterCat !== "all" && p.category !== filterCat) return false;
    if (filterSize !== "all" && p.company_size !== filterSize) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        p.company_name?.toLowerCase().includes(q) ||
        p.industry?.toLowerCase().includes(q) ||
        p.description?.toLowerCase().includes(q) ||
        p.notes?.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const catCounts: Record<string, number> = {};
  for (const p of prospects) {
    const c = p.category || "uncategorized";
    catCounts[c] = (catCounts[c] || 0) + 1;
  }

  const statusCounts: Record<string, number> = {};
  for (const p of prospects) {
    const s = p.status || "new";
    statusCounts[s] = (statusCounts[s] || 0) + 1;
  }

  return (
    <div>
      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        <div className="bg-neutral-900 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-white">{totalCount}</div>
          <div className="text-sm text-neutral-500">Total Prospects</div>
        </div>
        <div className="bg-neutral-900 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-400">{statusCounts["new"] ?? 0}</div>
          <div className="text-sm text-neutral-500">New</div>
        </div>
        <div className="bg-neutral-900 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-yellow-400">{statusCounts["contacted"] ?? 0}</div>
          <div className="text-sm text-neutral-500">Contacted</div>
        </div>
        <div className="bg-neutral-900 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-400">{statusCounts["replied"] ?? 0}</div>
          <div className="text-sm text-neutral-500">Replied</div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-neutral-900 rounded-lg p-4 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search companies..."
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          />
          <select
            value={filterCat}
            onChange={(e) => setFilterCat(e.target.value)}
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          >
            <option value="all">All Categories ({totalCount})</option>
            {CATEGORIES.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label} ({catCounts[c.value] ?? 0})
              </option>
            ))}
          </select>
          <select
            value={filterSize}
            onChange={(e) => setFilterSize(e.target.value)}
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          >
            <option value="all">All Sizes</option>
            {COMPANY_SIZES.map((s) => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          >
            <option value="all">All Statuses</option>
            <option value="new">New</option>
            <option value="contacted">Contacted</option>
            <option value="replied">Replied</option>
            <option value="not_interested">Not Interested</option>
          </select>
        </div>
      </div>

      {crmError && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg p-4 mb-6">
          {crmError}
        </div>
      )}

      {loadingCrm ? (
        <div className="text-center py-12 text-neutral-400 animate-pulse">Loading prospects...</div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-12 text-neutral-500">No prospects match your filters.</div>
      ) : (
        <>
          <div className="text-sm text-neutral-500 mb-3">
            Showing {filtered.length} of {totalCount} prospects
          </div>
          <div className="space-y-2">
            {filtered.map((p) => (
              <div
                key={p.id}
                className="bg-neutral-900 rounded-lg border border-neutral-800 overflow-hidden"
              >
                <div
                  className="p-4 cursor-pointer hover:bg-neutral-800/50 transition-colors"
                  onClick={() => setExpandedId(expandedId === p.id ? null : p.id)}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 flex-wrap">
                        <span className="text-green-400 font-medium">{p.company_name}</span>
                        {p.category && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-neutral-700 text-neutral-300">
                            {CATEGORIES.find((c) => c.value === p.category)?.label ?? p.category}
                          </span>
                        )}
                        {p.company_size && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-400">
                            {p.company_size}
                          </span>
                        )}
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          STATUS_COLORS[p.status ?? "new"] ?? STATUS_COLORS["new"]
                        }`}>
                          {(p.status ?? "new").replace("_", " ")}
                        </span>
                      </div>
                      {p.industry && (
                        <div className="text-xs text-neutral-500 mt-1">{p.industry}</div>
                      )}
                      {p.description && (
                        <p className="text-neutral-400 text-sm mt-1 line-clamp-1">{p.description}</p>
                      )}
                    </div>
                    <span className="text-neutral-600 text-sm shrink-0">
                      {expandedId === p.id ? "▲" : "▼"}
                    </span>
                  </div>
                </div>
                {expandedId === p.id && (
                  <div className="px-4 pb-4 border-t border-neutral-800 pt-3 space-y-3">
                    {p.description && (
                      <div>
                        <div className="text-xs text-neutral-500 uppercase tracking-wider mb-1">Description</div>
                        <p className="text-sm text-neutral-300">{p.description}</p>
                      </div>
                    )}
                    {p.notes && (
                      <div>
                        <div className="text-xs text-neutral-500 uppercase tracking-wider mb-1">Why They Fit</div>
                        <p className="text-sm text-neutral-300">{p.notes}</p>
                      </div>
                    )}
                    <div className="flex flex-wrap gap-4 text-sm text-neutral-400">
                      {p.contact_name && <span>Contact: {p.contact_name}</span>}
                      {p.contact_email && <span>Email: {p.contact_email}</span>}
                      {p.contact_title && <span>Title: {p.contact_title}</span>}
                      {p.location && <span>Location: {p.location}</span>}
                      {p.domain && <span>Domain: {p.domain}</span>}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ── Main Page ── */
export default function OutreachPage() {
  const [category, setCategory] = useState("ai-startups");
  const [customQuery, setCustomQuery] = useState("");
  const [maxResults, setMaxResults] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Prospect[]>([]);
  const [saved, setSaved] = useState<Prospect[]>([]);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"crm" | "search" | "saved" | "pipeline">("crm");
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) setSaved(JSON.parse(stored));
  }, []);

  function persistSaved(list: Prospect[]) {
    setSaved(list);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  }

  async function handleSearch() {
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const body: Record<string, unknown> = { maxResults };
      if (category === "custom") {
        body.query = customQuery;
      } else {
        body.category = category;
      }
      const res = await fetch("/api/admin/prospect-scraper", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { msg = (await res.json()).error || msg; } catch { /* non-JSON */ }
        throw new Error(msg);
      }
      const data = await safeJson(res);
      setResults(
        ((data.prospects ?? []) as Array<Omit<Prospect, "status" | "notes">>).map((p) => ({
          ...p,
          status: "new" as const,
          notes: "",
        }))
      );
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  function saveProspect(prospect: Prospect) {
    const exists = saved.some((s) => s.url === prospect.url);
    if (exists) return;
    persistSaved([...saved, { ...prospect, savedAt: new Date().toISOString() }]);
  }

  function updateStatus(url: string, status: Prospect["status"]) {
    persistSaved(saved.map((s) => (s.url === url ? { ...s, status } : s)));
  }

  function updateNotes(url: string, notes: string) {
    persistSaved(saved.map((s) => (s.url === url ? { ...s, notes } : s)));
  }

  function removeProspect(url: string) {
    persistSaved(saved.filter((s) => s.url !== url));
  }

  const stats = {
    total: saved.length,
    new: saved.filter((s) => s.status === "new").length,
    contacted: saved.filter((s) => s.status === "contacted").length,
    replied: saved.filter((s) => s.status === "replied").length,
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Prospect Pipeline</h1>
      <p className="text-neutral-400 mb-6">
        Find and enrich potential NemulAI clients using Apify-powered agents
      </p>

      {/* Stats bar */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        {[
          { label: "Saved", value: stats.total, color: "text-white" },
          { label: "New", value: stats.new, color: "text-blue-400" },
          { label: "Contacted", value: stats.contacted, color: "text-yellow-400" },
          { label: "Replied", value: stats.replied, color: "text-green-400" },
        ].map((s) => (
          <div key={s.label} className="bg-neutral-900 rounded-lg p-4 text-center">
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-sm text-neutral-500">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {(["crm", "pipeline", "search", "saved"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg font-medium capitalize ${
              tab === t
                ? "bg-green-600 text-white"
                : "bg-neutral-800 text-neutral-400 hover:text-white"
            }`}
          >
            {t === "pipeline" ? "Apify Pipeline" : t === "crm" ? "CRM" : t}{" "}
            {t === "saved" && `(${saved.length})`}
          </button>
        ))}
      </div>

      {tab === "crm" && <CrmTab />}

      {tab === "pipeline" && <PipelineTab />}

      {tab === "search" && (
        <>
          <div className="bg-neutral-900 rounded-lg p-6 mb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-neutral-400 mb-1">Category</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white"
                >
                  {CATEGORIES.map((c) => (
                    <option key={c.value} value={c.value}>{c.label}</option>
                  ))}
                </select>
              </div>
              {category === "custom" && (
                <div className="md:col-span-2">
                  <label className="block text-sm text-neutral-400 mb-1">Search Query</label>
                  <input
                    type="text"
                    value={customQuery}
                    onChange={(e) => setCustomQuery(e.target.value)}
                    placeholder='e.g. "AI company GPU cloud costs too high"'
                    className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white"
                  />
                </div>
              )}
              <div>
                <label className="block text-sm text-neutral-400 mb-1">Max Results per Query</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={maxResults}
                  onChange={(e) => setMaxResults(Number(e.target.value))}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white"
                />
              </div>
            </div>
            <button
              onClick={handleSearch}
              disabled={loading || (category === "custom" && !customQuery.trim())}
              className="mt-4 px-6 py-2 bg-green-600 hover:bg-green-500 disabled:bg-neutral-700 disabled:text-neutral-500 rounded-lg font-medium transition-colors"
            >
              {loading ? "Searching..." : "Search for Prospects"}
            </button>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg p-4 mb-6">
              {error}
            </div>
          )}

          {loading && (
            <div className="text-center py-12 text-neutral-400">
              <div className="animate-pulse text-lg">Scraping the web for prospects...</div>
              <div className="text-sm mt-2">This may take 15-30 seconds per query</div>
            </div>
          )}

          {results.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Found {results.length} prospects</h2>
              <div className="space-y-3">
                {results.map((p, i) => {
                  const isSaved = saved.some((s) => s.url === p.url);
                  return (
                    <div key={p.url} className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-green-300 font-medium text-lg">
                            {p.title}
                          </a>
                          <div className="text-xs text-neutral-500 mt-1 truncate">{p.url}</div>
                          <p className="text-neutral-300 text-sm mt-2">{p.description}</p>
                        </div>
                        <button
                          onClick={() => saveProspect(p)}
                          disabled={isSaved}
                          className={`shrink-0 px-4 py-2 rounded-lg text-sm font-medium ${
                            isSaved ? "bg-neutral-700 text-neutral-500" : "bg-green-600 hover:bg-green-500 text-white"
                          }`}
                        >
                          {isSaved ? "Saved" : "Save"}
                        </button>
                      </div>
                      <button
                        onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                        className="text-xs text-neutral-500 hover:text-neutral-300 mt-2"
                      >
                        {expandedIdx === i ? "Hide" : "Show"} snippet
                      </button>
                      {expandedIdx === i && p.snippet && (
                        <pre className="mt-2 text-xs text-neutral-500 bg-neutral-800 rounded p-3 whitespace-pre-wrap max-h-40 overflow-y-auto">
                          {p.snippet}
                        </pre>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      {tab === "saved" && (
        <div>
          {saved.length === 0 ? (
            <div className="text-center py-12 text-neutral-500">
              No saved prospects yet. Use the Search tab to find some.
            </div>
          ) : (
            <div className="space-y-3">
              {saved.map((p) => (
                <div key={p.url} className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3">
                        <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-green-300 font-medium">
                          {p.title}
                        </a>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLORS[p.status]}`}>
                          {p.status.replace("_", " ")}
                        </span>
                      </div>
                      <div className="text-xs text-neutral-500 mt-1 truncate">{p.url}</div>
                      <p className="text-neutral-400 text-sm mt-1">{p.description}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <select
                        value={p.status}
                        onChange={(e) => updateStatus(p.url, e.target.value as Prospect["status"])}
                        className="bg-neutral-800 border border-neutral-700 rounded px-2 py-1 text-sm text-white"
                      >
                        <option value="new">New</option>
                        <option value="contacted">Contacted</option>
                        <option value="replied">Replied</option>
                        <option value="not_interested">Not Interested</option>
                      </select>
                      <button
                        onClick={() => removeProspect(p.url)}
                        className="text-red-400 hover:text-red-300 text-sm px-2 py-1"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                  <textarea
                    value={p.notes}
                    onChange={(e) => updateNotes(p.url, e.target.value)}
                    placeholder="Add notes..."
                    rows={2}
                    className="mt-3 w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 resize-none"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
