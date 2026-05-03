"use client";

import { useState, useEffect, useCallback } from "react";

/* ── Types ── */
type TabId =
  | "contracts"
  | "redliner"
  | "questionnaire"
  | "compliance"
  | "policies"
  | "outreach_review";

interface SavedDocument {
  id: string;
  type: string;
  title: string;
  content: string;
  createdAt: string;
}

interface RedlineFlag {
  clause: string;
  riskLevel: "high" | "medium" | "low";
  category: string;
  explanation: string;
  suggestedLanguage: string;
}

interface QAAnswer {
  question: string;
  answer: string;
  confidence: "high" | "medium" | "low";
}

interface ComplianceItem {
  requirement: string;
  status: "pass" | "fail" | "needs_attention";
  notes: string;
}

interface ComplianceFramework {
  name: string;
  score?: string;
  items: ComplianceItem[];
}

interface OutreachIssue {
  regulation: string;
  issue: string;
  severity: "violation" | "warning" | "suggestion";
  fix: string;
  originalText?: string;
}

/* ── Constants ── */
const TABS: { id: TabId; label: string }[] = [
  { id: "contracts", label: "Contract Generator" },
  { id: "redliner", label: "Redliner" },
  { id: "questionnaire", label: "Security Q&A" },
  { id: "compliance", label: "Compliance Scan" },
  { id: "policies", label: "Policy Generator" },
  { id: "outreach_review", label: "Outreach Review" },
];

const STORAGE_KEY = "alum_lawyer_history_v1";

const CONTRACT_TYPES = [
  { value: "MSA", label: "MSA" },
  { value: "DPA", label: "DPA" },
  { value: "SOW", label: "SOW" },
  { value: "NDA", label: "NDA" },
];

const POLICY_TYPES = [
  { value: "terms", label: "Terms of Service" },
  { value: "privacy", label: "Privacy Policy" },
  { value: "acceptable_use", label: "Acceptable Use" },
];

const FRAMEWORKS = ["GDPR", "CCPA", "SOC2"];

const RISK_COLORS: Record<string, string> = {
  high: "bg-red-500/20 text-red-400 border-red-500/30",
  medium: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  low: "bg-blue-500/20 text-blue-400 border-blue-500/30",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-500/20 text-green-400",
  medium: "bg-yellow-500/20 text-yellow-400",
  low: "bg-red-500/20 text-red-400",
};

const STATUS_ICONS: Record<string, { icon: string; color: string }> = {
  pass: { icon: "✓", color: "text-green-400" },
  fail: { icon: "✗", color: "text-red-400" },
  needs_attention: { icon: "!", color: "text-yellow-400" },
};

const SEVERITY_COLORS: Record<string, string> = {
  violation: "bg-red-500/20 text-red-400",
  warning: "bg-yellow-500/20 text-yellow-400",
  suggestion: "bg-blue-500/20 text-blue-400",
};

/* ── Main Page ── */
export default function LawyerPage() {
  const [tab, setTab] = useState<TabId>("contracts");

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">AI Lawyer</h1>
        <p className="text-neutral-400 mb-6">
          Generate contracts, review legal documents, and ensure compliance with
          AI assistance
        </p>

        {/* Tab bar */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                tab === t.id
                  ? "bg-green-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "contracts" && <ContractGeneratorTab />}
        {tab === "redliner" && <ContractRedlinerTab />}
        {tab === "questionnaire" && <SecurityQuestionnaireTab />}
        {tab === "compliance" && <ComplianceScannerTab />}
        {tab === "policies" && <PolicyGeneratorTab />}
        {tab === "outreach_review" && <OutreachReviewTab />}
      </div>
    </div>
  );
}

/* ── Shared Helpers ── */
function useLawyerApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const callApi = useCallback(
    async (body: Record<string, unknown>) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch("/api/admin/lawyer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(
            (data as Record<string, string>).error || `HTTP ${res.status}`
          );
        }
        return await res.json();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Request failed");
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { loading, error, callApi };
}

function useHistory() {
  const [history, setHistory] = useState<SavedDocument[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setHistory(JSON.parse(stored));
    } catch {}
  }, []);

  const saveToHistory = useCallback(
    (type: string, title: string, content: string) => {
      const doc: SavedDocument = {
        id: crypto.randomUUID(),
        type,
        title,
        content,
        createdAt: new Date().toISOString(),
      };
      const updated = [doc, ...history].slice(0, 50);
      setHistory(updated);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    },
    [history]
  );

  return { history, saveToHistory };
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className="px-3 py-1.5 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm font-medium transition-colors"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function ErrorBanner({ error }: { error: string }) {
  if (!error) return null;
  return (
    <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg p-4 mb-6">
      {error}
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div className="text-center py-12 text-neutral-400">
      <div className="inline-block w-6 h-6 border-2 border-neutral-600 border-t-green-500 rounded-full animate-spin mb-3" />
      <p className="animate-pulse text-lg">Generating with AI...</p>
    </div>
  );
}

function DocumentResult({
  document,
  onSave,
}: {
  document: string;
  onSave?: () => void;
}) {
  return (
    <div className="mt-6 space-y-3">
      <div className="flex gap-2">
        <CopyButton text={document} />
        {onSave && (
          <button
            onClick={onSave}
            className="px-3 py-1.5 rounded-lg bg-green-600 hover:bg-green-500 text-sm font-medium transition-colors"
          >
            Save to History
          </button>
        )}
        <button
          onClick={() => {
            const blob = new Blob([document], { type: "text/markdown" });
            const url = URL.createObjectURL(blob);
            const a = Object.assign(window.document.createElement("a"), {
              href: url,
              download: `document-${Date.now()}.md`,
            });
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="px-3 py-1.5 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm font-medium transition-colors"
        >
          Download .md
        </button>
      </div>
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 whitespace-pre-wrap text-sm leading-relaxed font-mono max-h-[600px] overflow-y-auto">
        {document}
      </div>
    </div>
  );
}

/* ── Tab 1: Contract Generator ── */
function ContractGeneratorTab() {
  const [contractType, setContractType] = useState("MSA");
  const [companyName, setCompanyName] = useState("");
  const [dealTerms, setDealTerms] = useState("");
  const [specialClauses, setSpecialClauses] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const { loading, error, callApi } = useLawyerApi();
  const { saveToHistory } = useHistory();

  async function generate() {
    const data = await callApi({
      action: "generate_contract",
      contractType,
      companyName,
      dealTerms,
      specialClauses,
      jurisdiction,
    });
    if (data?.document) setResult(data.document);
  }

  return (
    <div className="space-y-6">
      <ErrorBanner error={error} />

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold">Generate Contract</h2>

        {/* Contract type selector */}
        <div>
          <label className="block text-sm text-neutral-400 mb-2">
            Contract Type
          </label>
          <div className="flex gap-2">
            {CONTRACT_TYPES.map((t) => (
              <button
                key={t.value}
                onClick={() => setContractType(t.value)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  contractType === t.value
                    ? "bg-green-600 text-white"
                    : "bg-neutral-800 text-neutral-400 hover:text-white"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Company name */}
        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Counterparty Company Name
          </label>
          <input
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="Acme Corp, Inc."
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          />
        </div>

        {/* Jurisdiction */}
        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Governing Law / Jurisdiction
          </label>
          <input
            value={jurisdiction}
            onChange={(e) => setJurisdiction(e.target.value)}
            placeholder="State of Delaware, USA (default)"
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          />
        </div>

        {/* Deal terms */}
        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Deal Terms & Specifics
          </label>
          <textarea
            value={dealTerms}
            onChange={(e) => setDealTerms(e.target.value)}
            placeholder="e.g., Enterprise plan, 100 GPUs, $5/GPU/month, 12-month commitment..."
            rows={3}
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y"
          />
        </div>

        {/* Special clauses */}
        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Special Clauses (optional)
          </label>
          <textarea
            value={specialClauses}
            onChange={(e) => setSpecialClauses(e.target.value)}
            placeholder="e.g., Custom SLA requirements, data residency, specific IP terms..."
            rows={3}
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y"
          />
        </div>

        <button
          onClick={generate}
          disabled={loading || !companyName.trim()}
          className="px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Generating..." : `Generate ${contractType}`}
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {result && !loading && (
        <DocumentResult
          document={result}
          onSave={() =>
            saveToHistory(
              "contract",
              `${contractType} — ${companyName}`,
              result
            )
          }
        />
      )}
    </div>
  );
}

/* ── Tab 2: Contract Redliner ── */
function ContractRedlinerTab() {
  const [contractText, setContractText] = useState("");
  const [flags, setFlags] = useState<RedlineFlag[]>([]);
  const [summary, setSummary] = useState("");
  const { loading, error, callApi } = useLawyerApi();

  async function analyze() {
    const data = await callApi({
      action: "redline_contract",
      contractText,
    });
    if (data) {
      setFlags(data.flags || []);
      setSummary(data.summary || "");
    }
  }

  const riskCounts = {
    high: flags.filter((f) => f.riskLevel === "high").length,
    medium: flags.filter((f) => f.riskLevel === "medium").length,
    low: flags.filter((f) => f.riskLevel === "low").length,
  };

  return (
    <div className="space-y-6">
      <ErrorBanner error={error} />

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold">Contract Redliner</h2>
        <p className="text-sm text-neutral-400">
          Paste a customer contract to identify risky clauses and get suggested
          counter-language.
        </p>
        <textarea
          value={contractText}
          onChange={(e) => setContractText(e.target.value)}
          placeholder="Paste the full contract text here..."
          rows={12}
          className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y font-mono"
        />
        <button
          onClick={analyze}
          disabled={loading || !contractText.trim()}
          className="px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Analyzing..." : "Analyze Contract"}
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {summary && !loading && (
        <>
          {/* Summary */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
            <div className="flex items-center gap-4 mb-3">
              <h3 className="font-semibold">Analysis Summary</h3>
              <div className="flex gap-2 text-sm">
                {riskCounts.high > 0 && (
                  <span className="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400">
                    {riskCounts.high} High
                  </span>
                )}
                {riskCounts.medium > 0 && (
                  <span className="px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400">
                    {riskCounts.medium} Medium
                  </span>
                )}
                {riskCounts.low > 0 && (
                  <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400">
                    {riskCounts.low} Low
                  </span>
                )}
              </div>
            </div>
            <p className="text-sm text-neutral-300">{summary}</p>
          </div>

          {/* Flags */}
          <div className="space-y-4">
            {flags.map((flag, i) => (
              <div
                key={i}
                className={`border rounded-lg p-4 space-y-3 ${RISK_COLORS[flag.riskLevel]}`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-semibold uppercase ${RISK_COLORS[flag.riskLevel]}`}
                  >
                    {flag.riskLevel}
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-xs bg-neutral-700 text-neutral-300">
                    {flag.category}
                  </span>
                </div>

                <div className="bg-neutral-950/50 rounded-lg p-3 text-sm font-mono text-neutral-300">
                  &ldquo;{flag.clause}&rdquo;
                </div>

                <p className="text-sm text-neutral-200">{flag.explanation}</p>

                <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-semibold text-green-400 uppercase">
                      Suggested Counter-Language
                    </span>
                    <CopyButton text={flag.suggestedLanguage} />
                  </div>
                  <p className="text-sm text-green-200">
                    {flag.suggestedLanguage}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ── Tab 3: Security Questionnaire Auto-Fill ── */
function SecurityQuestionnaireTab() {
  const [questionnaireText, setQuestionnaireText] = useState("");
  const [answers, setAnswers] = useState<QAAnswer[]>([]);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const { loading, error, callApi } = useLawyerApi();

  async function autoFill() {
    const data = await callApi({
      action: "fill_questionnaire",
      questionnaireText,
    });
    if (data?.answers) setAnswers(data.answers);
  }

  function updateAnswer(idx: number, newAnswer: string) {
    setAnswers((prev) =>
      prev.map((a, i) => (i === idx ? { ...a, answer: newAnswer } : a))
    );
  }

  function exportAsText() {
    const text = answers
      .map(
        (a, i) =>
          `Q${i + 1}: ${a.question}\nA: ${a.answer}\n`
      )
      .join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(window.document.createElement("a"), {
      href: url,
      download: `security-questionnaire-${Date.now()}.txt`,
    });
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6">
      <ErrorBanner error={error} />

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold">Security Questionnaire Auto-Fill</h2>
        <p className="text-sm text-neutral-400">
          Paste a vendor security questionnaire and AluminatAI&apos;s security posture
          will be used to auto-fill answers.
        </p>
        <textarea
          value={questionnaireText}
          onChange={(e) => setQuestionnaireText(e.target.value)}
          placeholder="Paste the questionnaire here — one question per line, or numbered questions..."
          rows={10}
          className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y font-mono"
        />
        <button
          onClick={autoFill}
          disabled={loading || !questionnaireText.trim()}
          className="px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Auto-Filling..." : "Auto-Fill Answers"}
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {answers.length > 0 && !loading && (
        <>
          <div className="flex gap-2">
            <CopyButton
              text={answers
                .map((a, i) => `Q${i + 1}: ${a.question}\nA: ${a.answer}`)
                .join("\n\n")}
            />
            <button
              onClick={exportAsText}
              className="px-3 py-1.5 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm font-medium transition-colors"
            >
              Export as .txt
            </button>
          </div>

          <div className="space-y-3">
            {answers.map((qa, i) => (
              <div
                key={i}
                className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 space-y-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-medium text-neutral-200">
                    Q{i + 1}: {qa.question}
                  </p>
                  <span
                    className={`shrink-0 px-2 py-0.5 rounded-full text-xs ${CONFIDENCE_COLORS[qa.confidence]}`}
                  >
                    {qa.confidence}
                  </span>
                </div>

                {editingIdx === i ? (
                  <div className="space-y-2">
                    <textarea
                      value={qa.answer}
                      onChange={(e) => updateAnswer(i, e.target.value)}
                      rows={3}
                      className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y"
                    />
                    <button
                      onClick={() => setEditingIdx(null)}
                      className="px-3 py-1 rounded-lg bg-green-600 hover:bg-green-500 text-sm font-medium"
                    >
                      Done
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => setEditingIdx(i)}
                    className="text-sm text-neutral-300 bg-neutral-800/50 rounded-lg p-3 cursor-pointer hover:bg-neutral-800 transition-colors"
                  >
                    {qa.answer}
                    <span className="text-neutral-500 text-xs ml-2">
                      (click to edit)
                    </span>
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

/* ── Tab 4: Compliance Scanner ── */
function ComplianceScannerTab() {
  const [selectedFrameworks, setSelectedFrameworks] = useState<string[]>([
    "GDPR",
  ]);
  const [results, setResults] = useState<ComplianceFramework[]>([]);
  const { loading, error, callApi } = useLawyerApi();

  function toggleFramework(fw: string) {
    setSelectedFrameworks((prev) =>
      prev.includes(fw) ? prev.filter((f) => f !== fw) : [...prev, fw]
    );
  }

  async function scan() {
    const data = await callApi({
      action: "scan_compliance",
      frameworks: selectedFrameworks,
    });
    if (data?.frameworks) setResults(data.frameworks);
  }

  return (
    <div className="space-y-6">
      <ErrorBanner error={error} />

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold">Compliance Scanner</h2>
        <p className="text-sm text-neutral-400">
          Evaluate AluminatAI&apos;s compliance posture against selected frameworks.
        </p>

        <div>
          <label className="block text-sm text-neutral-400 mb-2">
            Select Frameworks
          </label>
          <div className="flex gap-2">
            {FRAMEWORKS.map((fw) => (
              <button
                key={fw}
                onClick={() => toggleFramework(fw)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedFrameworks.includes(fw)
                    ? "bg-green-600 text-white"
                    : "bg-neutral-800 text-neutral-400 hover:text-white"
                }`}
              >
                {fw}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={scan}
          disabled={loading || selectedFrameworks.length === 0}
          className="px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Scanning..." : "Run Compliance Scan"}
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {results.length > 0 && !loading && (
        <div className="space-y-6">
          {results.map((fw) => {
            const counts = {
              pass: fw.items.filter((i) => i.status === "pass").length,
              fail: fw.items.filter((i) => i.status === "fail").length,
              needs_attention: fw.items.filter(
                (i) => i.status === "needs_attention"
              ).length,
            };
            return (
              <div
                key={fw.name}
                className="bg-neutral-900 border border-neutral-800 rounded-lg p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">{fw.name}</h3>
                  <div className="flex gap-3 text-sm">
                    <span className="text-green-400">
                      {counts.pass} passed
                    </span>
                    <span className="text-red-400">{counts.fail} failed</span>
                    <span className="text-yellow-400">
                      {counts.needs_attention} attention
                    </span>
                  </div>
                </div>

                {fw.score && (
                  <p className="text-sm text-neutral-400 mb-4">{fw.score}</p>
                )}

                <div className="space-y-2">
                  {fw.items.map((item, i) => {
                    const si = STATUS_ICONS[item.status];
                    return (
                      <div
                        key={i}
                        className="flex items-start gap-3 py-2 border-b border-neutral-800 last:border-0"
                      >
                        <span
                          className={`text-lg font-bold mt-0.5 ${si.color}`}
                        >
                          {si.icon}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-neutral-200">
                            {item.requirement}
                          </p>
                          <p className="text-xs text-neutral-400 mt-0.5">
                            {item.notes}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── Tab 5: Policy Generator ── */
function PolicyGeneratorTab() {
  const [policyType, setPolicyType] = useState("terms");
  const [companyDetails, setCompanyDetails] = useState("");
  const [productDescription, setProductDescription] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const { loading, error, callApi } = useLawyerApi();
  const { saveToHistory } = useHistory();

  async function generate() {
    const data = await callApi({
      action: "generate_policy",
      policyType,
      companyDetails,
      productDescription,
      jurisdiction,
    });
    if (data?.document) setResult(data.document);
  }

  const policyLabel =
    POLICY_TYPES.find((p) => p.value === policyType)?.label || "Policy";

  return (
    <div className="space-y-6">
      <ErrorBanner error={error} />

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold">Policy Generator</h2>

        <div>
          <label className="block text-sm text-neutral-400 mb-2">
            Policy Type
          </label>
          <div className="flex gap-2">
            {POLICY_TYPES.map((t) => (
              <button
                key={t.value}
                onClick={() => setPolicyType(t.value)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  policyType === t.value
                    ? "bg-green-600 text-white"
                    : "bg-neutral-800 text-neutral-400 hover:text-white"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Additional Company Details (optional)
          </label>
          <textarea
            value={companyDetails}
            onChange={(e) => setCompanyDetails(e.target.value)}
            placeholder="e.g., Incorporated in Delaware, 10 employees, Series A..."
            rows={2}
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y"
          />
        </div>

        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Product Description (optional)
          </label>
          <textarea
            value={productDescription}
            onChange={(e) => setProductDescription(e.target.value)}
            placeholder="e.g., Additional product features or data handling specifics..."
            rows={2}
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y"
          />
        </div>

        <div>
          <label className="block text-sm text-neutral-400 mb-1">
            Jurisdiction
          </label>
          <input
            value={jurisdiction}
            onChange={(e) => setJurisdiction(e.target.value)}
            placeholder="United States (default)"
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm"
          />
        </div>

        <button
          onClick={generate}
          disabled={loading}
          className="px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Generating..." : `Generate ${policyLabel}`}
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {result && !loading && (
        <DocumentResult
          document={result}
          onSave={() => saveToHistory("policy", policyLabel, result)}
        />
      )}
    </div>
  );
}

/* ── Tab 6: Outreach Compliance Review ── */
function OutreachReviewTab() {
  const [emailText, setEmailText] = useState("");
  const [targetRegions, setTargetRegions] = useState<string[]>([
    "US",
    "EU",
    "Canada",
  ]);
  const [result, setResult] = useState<{
    overallCompliant: boolean;
    summary: string;
    issues: OutreachIssue[];
  } | null>(null);
  const { loading, error, callApi } = useLawyerApi();

  function toggleRegion(region: string) {
    setTargetRegions((prev) =>
      prev.includes(region)
        ? prev.filter((r) => r !== region)
        : [...prev, region]
    );
  }

  async function review() {
    const data = await callApi({
      action: "review_outreach",
      emailText,
      targetRegions,
    });
    if (data) setResult(data);
  }

  return (
    <div className="space-y-6">
      <ErrorBanner error={error} />

      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold">Outreach Compliance Review</h2>
        <p className="text-sm text-neutral-400">
          Check outreach emails for CAN-SPAM, GDPR, and CASL compliance before
          sending.
        </p>

        <textarea
          value={emailText}
          onChange={(e) => setEmailText(e.target.value)}
          placeholder="Paste your outreach email here..."
          rows={8}
          className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm resize-y font-mono"
        />

        <div>
          <label className="block text-sm text-neutral-400 mb-2">
            Target Regions
          </label>
          <div className="flex gap-2">
            {["US", "EU", "Canada"].map((region) => (
              <button
                key={region}
                onClick={() => toggleRegion(region)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  targetRegions.includes(region)
                    ? "bg-green-600 text-white"
                    : "bg-neutral-800 text-neutral-400 hover:text-white"
                }`}
              >
                {region}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={review}
          disabled={loading || !emailText.trim()}
          className="px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Checking..." : "Check Compliance"}
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {result && !loading && (
        <div className="space-y-4">
          {/* Overall status */}
          <div
            className={`rounded-lg p-4 border ${
              result.overallCompliant
                ? "bg-green-500/10 border-green-500/30"
                : "bg-red-500/10 border-red-500/30"
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span
                className={`text-lg font-bold ${
                  result.overallCompliant
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >
                {result.overallCompliant ? "Compliant" : "Issues Found"}
              </span>
            </div>
            <p
              className={`text-sm ${
                result.overallCompliant
                  ? "text-green-300"
                  : "text-red-300"
              }`}
            >
              {result.summary}
            </p>
          </div>

          {/* Issues */}
          {result.issues.length > 0 && (
            <div className="space-y-3">
              {result.issues.map((issue, i) => (
                <div
                  key={i}
                  className="bg-neutral-900 border border-neutral-800 rounded-lg p-4 space-y-2"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-semibold ${SEVERITY_COLORS[issue.severity]}`}
                    >
                      {issue.severity}
                    </span>
                    <span className="px-2 py-0.5 rounded-full text-xs bg-neutral-700 text-neutral-300">
                      {issue.regulation}
                    </span>
                  </div>

                  <p className="text-sm text-neutral-200">{issue.issue}</p>

                  {issue.originalText && (
                    <div className="bg-neutral-950/50 rounded-lg p-3 text-sm font-mono text-neutral-400">
                      &ldquo;{issue.originalText}&rdquo;
                    </div>
                  )}

                  <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                    <span className="text-xs font-semibold text-green-400 uppercase">
                      Suggested Fix
                    </span>
                    <p className="text-sm text-green-200 mt-1">{issue.fix}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
