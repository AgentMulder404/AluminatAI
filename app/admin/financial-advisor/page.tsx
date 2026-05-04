"use client";

import { useState, useEffect, useCallback, useMemo } from "react";

// ── Types ──

type TabId =
  | "revenue"
  | "expenses"
  | "investors"
  | "valuation"
  | "pricing"
  | "reports";

interface RevenueEntry {
  id: string;
  date: string;
  source: "subscriptions" | "consulting" | "grants" | "other";
  label: string;
  amount: number;
  recurring: boolean;
  notes: string;
}

interface EmployeeEntry {
  id: string;
  name: string;
  role: string;
  type: "employee" | "contractor";
  monthlySalary: number;
  startDate: string;
  equityPercent: number;
  active: boolean;
}

interface ExpenseEntry {
  id: string;
  date: string;
  category:
    | "infrastructure"
    | "tools"
    | "services"
    | "marketing"
    | "legal"
    | "other";
  label: string;
  amount: number;
  recurring: boolean;
  notes: string;
}

interface InvestorEntry {
  id: string;
  name: string;
  type: "angel" | "vc" | "grant" | "safe" | "convertible_note";
  amount: number;
  date: string;
  equityPercent: number;
  round: "pre-seed" | "seed" | "series-a" | "series-b" | "grant";
  terms: string;
  notes: string;
}

interface FounderEntry {
  id: string;
  name: string;
  role: string;
  equityPercent: number;
  vestingMonths: number;
  cliffMonths: number;
}

interface CompetitorEntry {
  id: string;
  name: string;
  freePrice: number;
  proPrice: number;
  enterprisePrice: number;
  notes: string;
}

interface AnalysisResult {
  content: string;
  generatedAt: string;
  action: string;
}

// ── Constants ──

const TABS: { id: TabId; label: string }[] = [
  { id: "revenue", label: "Revenue & Income" },
  { id: "expenses", label: "Expenses & Payroll" },
  { id: "investors", label: "Investors & Cap Table" },
  { id: "valuation", label: "Valuation Engine" },
  { id: "pricing", label: "Pricing Advisor" },
  { id: "reports", label: "Financial Reports" },
];

const REVENUE_SOURCES: { value: RevenueEntry["source"]; label: string }[] = [
  { value: "subscriptions", label: "Subscriptions" },
  { value: "consulting", label: "Consulting" },
  { value: "grants", label: "Grants" },
  { value: "other", label: "Other" },
];

const EXPENSE_CATEGORIES: { value: ExpenseEntry["category"]; label: string }[] =
  [
    { value: "infrastructure", label: "Infrastructure" },
    { value: "tools", label: "Tools & Software" },
    { value: "services", label: "Professional Services" },
    { value: "marketing", label: "Marketing" },
    { value: "legal", label: "Legal" },
    { value: "other", label: "Other" },
  ];

const INVESTOR_TYPES: { value: InvestorEntry["type"]; label: string }[] = [
  { value: "angel", label: "Angel" },
  { value: "vc", label: "VC" },
  { value: "grant", label: "Grant" },
  { value: "safe", label: "SAFE" },
  { value: "convertible_note", label: "Conv. Note" },
];

const FUNDING_ROUNDS: { value: InvestorEntry["round"]; label: string }[] = [
  { value: "pre-seed", label: "Pre-Seed" },
  { value: "seed", label: "Seed" },
  { value: "series-a", label: "Series A" },
  { value: "series-b", label: "Series B" },
  { value: "grant", label: "Grant" },
];

const SOURCE_COLORS: Record<string, string> = {
  subscriptions: "bg-green-500/20 text-green-400",
  consulting: "bg-blue-500/20 text-blue-400",
  grants: "bg-purple-500/20 text-purple-400",
  other: "bg-neutral-500/20 text-neutral-400",
};

const CATEGORY_COLORS: Record<string, string> = {
  infrastructure: "bg-red-500/20 text-red-400",
  tools: "bg-blue-500/20 text-blue-400",
  services: "bg-purple-500/20 text-purple-400",
  marketing: "bg-yellow-500/20 text-yellow-400",
  legal: "bg-orange-500/20 text-orange-400",
  other: "bg-neutral-500/20 text-neutral-400",
};

const INPUT_CLS =
  "w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm";
const SMALL_INPUT_CLS =
  "bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-white text-sm";

// ── Hooks ──

function useFinanceApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const callApi = useCallback(
    async (body: Record<string, unknown>) => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch("/api/admin/financial-advisor", {
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
        return (await res.json()) as Record<string, unknown>;
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

function useFinanceData<T>(storageKey: string, fallback: T) {
  const [data, setData] = useState<T>(fallback);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) setData(JSON.parse(stored));
    } catch {
      /* ignore */
    }
  }, [storageKey]);

  const save = useCallback(
    (updated: T) => {
      setData(updated);
      try {
        localStorage.setItem(storageKey, JSON.stringify(updated));
      } catch {
        /* ignore */
      }
    },
    [storageKey]
  );

  return { data, save };
}

function gatherAllFinanceData() {
  const read = <T,>(key: string, fallback: T): T => {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : fallback;
    } catch {
      return fallback;
    }
  };
  return {
    revenue: read<RevenueEntry[]>("alum_finance_revenue_v1", []),
    employees: read<EmployeeEntry[]>("alum_finance_employees_v1", []),
    expenses: read<ExpenseEntry[]>("alum_finance_expenses_v1", []),
    investors: read<InvestorEntry[]>("alum_finance_investors_v1", []),
    founders: read<FounderEntry[]>("alum_finance_founders_v1", []),
    competitors: read<CompetitorEntry[]>("alum_finance_competitors_v1", []),
  };
}

// ── Compute Functions ──

function computeMRR(revenue: RevenueEntry[]): number {
  const recurring = revenue.filter((r) => r.recurring);
  if (recurring.length === 0) return 0;
  const months = [...new Set(recurring.map((r) => r.date))].sort();
  const latest = months[months.length - 1];
  return recurring
    .filter((r) => r.date === latest)
    .reduce((sum, r) => sum + r.amount, 0);
}

function computeARR(revenue: RevenueEntry[]): number {
  return computeMRR(revenue) * 12;
}

function computeMonthlyBurn(
  employees: EmployeeEntry[],
  expenses: ExpenseEntry[]
): number {
  const payroll = employees
    .filter((e) => e.active)
    .reduce((sum, e) => sum + e.monthlySalary, 0);
  const opex = expenses
    .filter((e) => e.recurring)
    .reduce((sum, e) => sum + e.amount, 0);
  return payroll + opex;
}

function computeRunway(totalCash: number, monthlyBurn: number): number {
  if (monthlyBurn <= 0) return Infinity;
  return Math.round(totalCash / monthlyBurn);
}

function computeMRRGrowth(revenue: RevenueEntry[]): number | null {
  const recurring = revenue.filter((r) => r.recurring);
  const months = [...new Set(recurring.map((r) => r.date))].sort();
  if (months.length < 2) return null;
  const latest = months[months.length - 1];
  const prior = months[months.length - 2];
  const latestMRR = recurring
    .filter((r) => r.date === latest)
    .reduce((s, r) => s + r.amount, 0);
  const priorMRR = recurring
    .filter((r) => r.date === prior)
    .reduce((s, r) => s + r.amount, 0);
  if (priorMRR === 0) return null;
  return ((latestMRR - priorMRR) / priorMRR) * 100;
}

function computeTotalRaised(investors: InvestorEntry[]): number {
  return investors.reduce((sum, i) => sum + i.amount, 0);
}

function computeTotalEquity(
  founders: FounderEntry[],
  investors: InvestorEntry[]
): number {
  return (
    founders.reduce((s, f) => s + f.equityPercent, 0) +
    investors.reduce((s, i) => s + i.equityPercent, 0)
  );
}

function computeSimpleDCF(
  currentARR: number,
  growthRate: number,
  discountRate: number,
  years: number = 5
) {
  const projections: {
    year: number;
    revenue: number;
    discounted: number;
  }[] = [];
  let totalNPV = 0;
  for (let y = 1; y <= years; y++) {
    const revenue = currentARR * Math.pow(1 + growthRate / 100, y);
    const discounted = revenue / Math.pow(1 + discountRate / 100, y);
    projections.push({ year: y, revenue, discounted });
    totalNPV += discounted;
  }
  const terminalRevenue = currentARR * Math.pow(1 + growthRate / 100, years);
  const terminalValue =
    (terminalRevenue * 3) / Math.pow(1 + discountRate / 100, years);
  totalNPV += terminalValue;
  return { projections, npv: totalNPV, terminalValue };
}

function computeGrossMargin(
  revenue: RevenueEntry[],
  expenses: ExpenseEntry[]
): number {
  const totalRev = revenue.reduce((s, r) => s + r.amount, 0);
  if (totalRev === 0) return 0;
  const infra = expenses
    .filter((e) => e.category === "infrastructure")
    .reduce((s, e) => s + e.amount, 0);
  return ((totalRev - infra) / totalRev) * 100;
}

// ── Format Helpers ──

function fmt(n: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(n);
}

function fmtPct(n: number | null): string {
  if (n === null) return "N/A";
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

function uid(): string {
  return crypto.randomUUID();
}

// ── Shared UI Components ──

function ErrorBanner({ error }: { error: string }) {
  if (!error) return null;
  return (
    <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 text-sm">
      {error}
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-12 gap-3">
      <div className="w-6 h-6 border-2 border-neutral-600 border-t-green-500 rounded-full animate-spin" />
      <span className="text-neutral-400 text-sm animate-pulse">
        Analyzing...
      </span>
    </div>
  );
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
      className="bg-neutral-700 hover:bg-neutral-600 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function DocumentResult({
  document,
  title,
}: {
  document: string;
  title?: string;
}) {
  const handleDownload = () => {
    const blob = new Blob([document], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = globalThis.document.createElement("a");
    a.href = url;
    a.download = `${title ?? "report"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mt-6 space-y-3">
      <div className="flex gap-2">
        <CopyButton text={document} />
        <button
          onClick={handleDownload}
          className="bg-neutral-700 hover:bg-neutral-600 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
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

function MetricCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
      <div className="text-neutral-400 text-xs mb-1">{label}</div>
      <div className={`text-xl font-bold ${color ?? "text-white"}`}>
        {value}
      </div>
      {sub && <div className="text-neutral-500 text-xs mt-1">{sub}</div>}
    </div>
  );
}

// ── Tab Components ──

function RevenueTab() {
  const { data: entries, save } = useFinanceData<RevenueEntry[]>(
    "alum_finance_revenue_v1",
    []
  );
  const { loading, error, callApi } = useFinanceApi();
  const [analysis, setAnalysis] = useState<string | null>(null);

  const [date, setDate] = useState(
    new Date().toISOString().slice(0, 7)
  );
  const [source, setSource] = useState<RevenueEntry["source"]>("subscriptions");
  const [label, setLabel] = useState("");
  const [amount, setAmount] = useState("");
  const [recurring, setRecurring] = useState(true);
  const [notes, setNotes] = useState("");

  const mrr = computeMRR(entries);
  const arr = computeARR(entries);
  const growth = computeMRRGrowth(entries);

  const monthlyBreakdown = useMemo(() => {
    const months = [...new Set(entries.map((e) => e.date))].sort().slice(-6);
    return months.map((m) => ({
      month: m,
      total: entries
        .filter((e) => e.date === m)
        .reduce((s, e) => s + e.amount, 0),
    }));
  }, [entries]);

  const maxMonthly = Math.max(...monthlyBreakdown.map((m) => m.total), 1);

  const addEntry = () => {
    if (!label.trim() || !amount) return;
    const entry: RevenueEntry = {
      id: uid(),
      date,
      source,
      label: label.trim(),
      amount: parseFloat(amount),
      recurring,
      notes: notes.trim(),
    };
    save([entry, ...entries]);
    setLabel("");
    setAmount("");
    setNotes("");
  };

  const deleteEntry = (id: string) => save(entries.filter((e) => e.id !== id));

  const analyze = async () => {
    const res = await callApi({ action: "analyze_revenue", revenue: entries });
    if (res) setAnalysis(res.analysis as string);
  };

  return (
    <div className="space-y-6">
      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="MRR" value={fmt(mrr)} color="text-green-400" />
        <MetricCard label="ARR" value={fmt(arr)} color="text-green-400" />
        <MetricCard
          label="MRR Growth"
          value={fmtPct(growth)}
          color={
            growth === null
              ? "text-neutral-400"
              : growth >= 0
                ? "text-green-400"
                : "text-red-400"
          }
        />
      </div>

      {/* Trend */}
      {monthlyBreakdown.length > 0 && (
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
          <div className="text-sm text-neutral-400 mb-3">Revenue Trend (Last 6 Months)</div>
          <div className="space-y-2">
            {monthlyBreakdown.map((m) => (
              <div key={m.month} className="flex items-center gap-3 text-sm">
                <span className="w-20 text-neutral-400">{m.month}</span>
                <div className="flex-1 bg-neutral-800 rounded-full h-3">
                  <div
                    className="bg-green-600 h-3 rounded-full transition-all"
                    style={{
                      width: `${(m.total / maxMonthly) * 100}%`,
                    }}
                  />
                </div>
                <span className="w-24 text-right text-white">{fmt(m.total)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Add Entry Form */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Add Revenue Entry</div>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Month</label>
            <input
              type="month"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className={SMALL_INPUT_CLS + " w-full"}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Source</label>
            <div className="flex gap-1 flex-wrap">
              {REVENUE_SOURCES.map((s) => (
                <button
                  key={s.value}
                  onClick={() => setSource(s.value)}
                  className={`px-3 py-1.5 rounded-lg text-xs transition-colors ${
                    source === s.value
                      ? "bg-green-600 text-white"
                      : "bg-neutral-800 text-neutral-400 hover:text-white"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Label</label>
            <input
              type="text"
              placeholder="e.g. Pro subscriptions"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Amount ($)</label>
            <input
              type="number"
              placeholder="0"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Notes</label>
            <input
              type="text"
              placeholder="Optional"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-neutral-300 cursor-pointer">
            <input
              type="checkbox"
              checked={recurring}
              onChange={(e) => setRecurring(e.target.checked)}
              className="accent-green-600"
            />
            Recurring (MRR)
          </label>
          <button
            onClick={addEntry}
            disabled={!label.trim() || !amount}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
          >
            Add Entry
          </button>
        </div>
      </div>

      {/* Entries Table */}
      {entries.length > 0 && (
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                <th className="px-4 py-3">Date</th>
                <th className="px-4 py-3">Source</th>
                <th className="px-4 py-3">Label</th>
                <th className="px-4 py-3 text-right">Amount</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr
                  key={e.id}
                  className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                >
                  <td className="px-4 py-2 text-neutral-300">{e.date}</td>
                  <td className="px-4 py-2">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${SOURCE_COLORS[e.source]}`}
                    >
                      {e.source}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-white">{e.label}</td>
                  <td className="px-4 py-2 text-right text-white">
                    {fmt(e.amount)}
                  </td>
                  <td className="px-4 py-2 text-neutral-400">
                    {e.recurring ? "Recurring" : "One-time"}
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() => deleteEntry(e.id)}
                      className="text-neutral-600 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* AI Analysis */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <button
          onClick={analyze}
          disabled={loading || entries.length === 0}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Analyze Revenue Trends
        </button>
        <ErrorBanner error={error} />
        {loading && <LoadingSpinner />}
        {analysis && <DocumentResult document={analysis} title="revenue-analysis" />}
      </div>
    </div>
  );
}

function ExpensesTab() {
  const { data: employees, save: saveEmployees } = useFinanceData<
    EmployeeEntry[]
  >("alum_finance_employees_v1", []);
  const { data: expenses, save: saveExpenses } = useFinanceData<ExpenseEntry[]>(
    "alum_finance_expenses_v1",
    []
  );
  const { loading, error, callApi } = useFinanceApi();
  const [analysis, setAnalysis] = useState<string | null>(null);

  // Employee form
  const [empName, setEmpName] = useState("");
  const [empRole, setEmpRole] = useState("");
  const [empType, setEmpType] = useState<EmployeeEntry["type"]>("employee");
  const [empSalary, setEmpSalary] = useState("");
  const [empStart, setEmpStart] = useState("");
  const [empEquity, setEmpEquity] = useState("");

  // Expense form
  const [expDate, setExpDate] = useState(
    new Date().toISOString().slice(0, 7)
  );
  const [expCategory, setExpCategory] =
    useState<ExpenseEntry["category"]>("infrastructure");
  const [expLabel, setExpLabel] = useState("");
  const [expAmount, setExpAmount] = useState("");
  const [expRecurring, setExpRecurring] = useState(true);
  const [expNotes, setExpNotes] = useState("");

  const totalPayroll = employees
    .filter((e) => e.active)
    .reduce((s, e) => s + e.monthlySalary, 0);
  const totalOpex = expenses
    .filter((e) => e.recurring)
    .reduce((s, e) => s + e.amount, 0);
  const burn = totalPayroll + totalOpex;

  const allData = useMemo(gatherAllFinanceData, [employees, expenses]);
  const totalRaised = computeTotalRaised(allData.investors);
  const totalExpensesAllTime =
    employees.reduce(
      (s, e) => {
        if (!e.active || !e.startDate) return s;
        const months = Math.max(
          1,
          Math.round(
            (Date.now() - new Date(e.startDate).getTime()) / (30 * 86400000)
          )
        );
        return s + e.monthlySalary * months;
      },
      0
    ) + expenses.reduce((s, e) => s + e.amount, 0);
  const estimatedCash = Math.max(0, totalRaised - totalExpensesAllTime);
  const runway = computeRunway(estimatedCash, burn);

  const addEmployee = () => {
    if (!empName.trim() || !empSalary) return;
    const entry: EmployeeEntry = {
      id: uid(),
      name: empName.trim(),
      role: empRole.trim(),
      type: empType,
      monthlySalary: parseFloat(empSalary),
      startDate: empStart,
      equityPercent: parseFloat(empEquity) || 0,
      active: true,
    };
    saveEmployees([entry, ...employees]);
    setEmpName("");
    setEmpRole("");
    setEmpSalary("");
    setEmpStart("");
    setEmpEquity("");
  };

  const addExpense = () => {
    if (!expLabel.trim() || !expAmount) return;
    const entry: ExpenseEntry = {
      id: uid(),
      date: expDate,
      category: expCategory,
      label: expLabel.trim(),
      amount: parseFloat(expAmount),
      recurring: expRecurring,
      notes: expNotes.trim(),
    };
    saveExpenses([entry, ...expenses]);
    setExpLabel("");
    setExpAmount("");
    setExpNotes("");
  };

  const analyze = async () => {
    const res = await callApi({
      action: "optimize_costs",
      employees,
      expenses,
      revenue: allData.revenue,
    });
    if (res) setAnalysis(res.analysis as string);
  };

  return (
    <div className="space-y-6">
      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Monthly Payroll" value={fmt(totalPayroll)} />
        <MetricCard label="Monthly OpEx" value={fmt(totalOpex)} />
        <MetricCard
          label="Burn Rate"
          value={fmt(burn)}
          color="text-red-400"
        />
        <MetricCard
          label="Runway"
          value={runway === Infinity ? "N/A" : `${runway} mo`}
          sub={runway !== Infinity ? `~${fmt(estimatedCash)} remaining` : "Enter funding data"}
          color={
            runway === Infinity
              ? "text-neutral-400"
              : runway > 12
                ? "text-green-400"
                : runway > 6
                  ? "text-yellow-400"
                  : "text-red-400"
          }
        />
      </div>

      {/* Payroll Section */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Team & Payroll</div>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Name</label>
            <input
              type="text"
              placeholder="Full name"
              value={empName}
              onChange={(e) => setEmpName(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Role</label>
            <input
              type="text"
              placeholder="e.g. Full-Stack Engineer"
              value={empRole}
              onChange={(e) => setEmpRole(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Type</label>
            <div className="flex gap-1">
              {(["employee", "contractor"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setEmpType(t)}
                  className={`flex-1 px-3 py-2 rounded-lg text-xs transition-colors ${
                    empType === t
                      ? "bg-green-600 text-white"
                      : "bg-neutral-800 text-neutral-400 hover:text-white"
                  }`}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Monthly Salary ($)
            </label>
            <input
              type="number"
              placeholder="0"
              value={empSalary}
              onChange={(e) => setEmpSalary(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Start Date
            </label>
            <input
              type="date"
              value={empStart}
              onChange={(e) => setEmpStart(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Equity (%)
            </label>
            <input
              type="number"
              placeholder="0"
              value={empEquity}
              onChange={(e) => setEmpEquity(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <button
          onClick={addEmployee}
          disabled={!empName.trim() || !empSalary}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Add Team Member
        </button>

        {employees.length > 0 && (
          <table className="w-full text-sm mt-4">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                <th className="px-4 py-2">Name</th>
                <th className="px-4 py-2">Role</th>
                <th className="px-4 py-2">Type</th>
                <th className="px-4 py-2 text-right">Salary/mo</th>
                <th className="px-4 py-2 text-right">Equity</th>
                <th className="px-4 py-2">Status</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {employees.map((e) => (
                <tr
                  key={e.id}
                  className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                >
                  <td className="px-4 py-2 text-white">{e.name}</td>
                  <td className="px-4 py-2 text-neutral-300">{e.role}</td>
                  <td className="px-4 py-2 text-neutral-400">{e.type}</td>
                  <td className="px-4 py-2 text-right text-white">
                    {fmt(e.monthlySalary)}
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {e.equityPercent}%
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() =>
                        saveEmployees(
                          employees.map((x) =>
                            x.id === e.id ? { ...x, active: !x.active } : x
                          )
                        )
                      }
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        e.active
                          ? "bg-green-500/20 text-green-400"
                          : "bg-neutral-500/20 text-neutral-400"
                      }`}
                    >
                      {e.active ? "Active" : "Inactive"}
                    </button>
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() =>
                        saveEmployees(employees.filter((x) => x.id !== e.id))
                      }
                      className="text-neutral-600 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Expenses Section */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Operating Expenses</div>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Month</label>
            <input
              type="month"
              value={expDate}
              onChange={(e) => setExpDate(e.target.value)}
              className={SMALL_INPUT_CLS + " w-full"}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Category
            </label>
            <div className="flex gap-1 flex-wrap">
              {EXPENSE_CATEGORIES.map((c) => (
                <button
                  key={c.value}
                  onClick={() => setExpCategory(c.value)}
                  className={`px-2 py-1 rounded-lg text-xs transition-colors ${
                    expCategory === c.value
                      ? "bg-green-600 text-white"
                      : "bg-neutral-800 text-neutral-400 hover:text-white"
                  }`}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Label</label>
            <input
              type="text"
              placeholder="e.g. Vercel Pro"
              value={expLabel}
              onChange={(e) => setExpLabel(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Amount ($)
            </label>
            <input
              type="number"
              placeholder="0"
              value={expAmount}
              onChange={(e) => setExpAmount(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Notes</label>
            <input
              type="text"
              placeholder="Optional"
              value={expNotes}
              onChange={(e) => setExpNotes(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-neutral-300 cursor-pointer">
            <input
              type="checkbox"
              checked={expRecurring}
              onChange={(e) => setExpRecurring(e.target.checked)}
              className="accent-green-600"
            />
            Recurring
          </label>
          <button
            onClick={addExpense}
            disabled={!expLabel.trim() || !expAmount}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
          >
            Add Expense
          </button>
        </div>

        {expenses.length > 0 && (
          <table className="w-full text-sm mt-4">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                <th className="px-4 py-2">Date</th>
                <th className="px-4 py-2">Category</th>
                <th className="px-4 py-2">Label</th>
                <th className="px-4 py-2 text-right">Amount</th>
                <th className="px-4 py-2">Type</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {expenses.map((e) => (
                <tr
                  key={e.id}
                  className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                >
                  <td className="px-4 py-2 text-neutral-300">{e.date}</td>
                  <td className="px-4 py-2">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${CATEGORY_COLORS[e.category]}`}
                    >
                      {e.category}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-white">{e.label}</td>
                  <td className="px-4 py-2 text-right text-white">
                    {fmt(e.amount)}
                  </td>
                  <td className="px-4 py-2 text-neutral-400">
                    {e.recurring ? "Recurring" : "One-time"}
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() =>
                        saveExpenses(expenses.filter((x) => x.id !== e.id))
                      }
                      className="text-neutral-600 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* AI Analysis */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <button
          onClick={analyze}
          disabled={loading || (employees.length === 0 && expenses.length === 0)}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Optimize Costs
        </button>
        <ErrorBanner error={error} />
        {loading && <LoadingSpinner />}
        {analysis && (
          <DocumentResult document={analysis} title="cost-optimization" />
        )}
      </div>
    </div>
  );
}

function InvestorsTab() {
  const { data: founders, save: saveFounders } = useFinanceData<FounderEntry[]>(
    "alum_finance_founders_v1",
    []
  );
  const { data: investors, save: saveInvestors } = useFinanceData<
    InvestorEntry[]
  >("alum_finance_investors_v1", []);
  const { loading, error, callApi } = useFinanceApi();
  const [analysis, setAnalysis] = useState<string | null>(null);

  // Founder form
  const [fName, setFName] = useState("");
  const [fRole, setFRole] = useState("");
  const [fEquity, setFEquity] = useState("");
  const [fVesting, setFVesting] = useState("48");
  const [fCliff, setFCliff] = useState("12");

  // Investor form
  const [iName, setIName] = useState("");
  const [iType, setIType] = useState<InvestorEntry["type"]>("angel");
  const [iAmount, setIAmount] = useState("");
  const [iDate, setIDate] = useState("");
  const [iEquity, setIEquity] = useState("");
  const [iRound, setIRound] = useState<InvestorEntry["round"]>("pre-seed");
  const [iTerms, setITerms] = useState("");
  const [iNotes, setINotes] = useState("");

  const founderEquity = founders.reduce((s, f) => s + f.equityPercent, 0);
  const investorEquity = investors.reduce((s, i) => s + i.equityPercent, 0);
  const totalAllocated = founderEquity + investorEquity;
  const unallocated = Math.max(0, 100 - totalAllocated);
  const totalRaised = computeTotalRaised(investors);

  const capTable = [
    ...founders.map((f) => ({
      name: f.name,
      role: f.role,
      equity: f.equityPercent,
      type: "Founder",
    })),
    ...investors.map((i) => ({
      name: i.name,
      role: i.type,
      equity: i.equityPercent,
      type: "Investor",
    })),
    { name: "Unallocated", role: "Option Pool", equity: unallocated, type: "Pool" },
  ];

  const addFounder = () => {
    if (!fName.trim() || !fEquity) return;
    saveFounders([
      {
        id: uid(),
        name: fName.trim(),
        role: fRole.trim(),
        equityPercent: parseFloat(fEquity),
        vestingMonths: parseInt(fVesting) || 48,
        cliffMonths: parseInt(fCliff) || 12,
      },
      ...founders,
    ]);
    setFName("");
    setFRole("");
    setFEquity("");
  };

  const addInvestor = () => {
    if (!iName.trim() || !iAmount) return;
    saveInvestors([
      {
        id: uid(),
        name: iName.trim(),
        type: iType,
        amount: parseFloat(iAmount),
        date: iDate,
        equityPercent: parseFloat(iEquity) || 0,
        round: iRound,
        terms: iTerms.trim(),
        notes: iNotes.trim(),
      },
      ...investors,
    ]);
    setIName("");
    setIAmount("");
    setIDate("");
    setIEquity("");
    setITerms("");
    setINotes("");
  };

  const analyze = async () => {
    const allData = gatherAllFinanceData();
    const res = await callApi({
      action: "assess_fundraising",
      investors,
      founders,
      revenue: allData.revenue,
      expenses: allData.expenses,
      employees: allData.employees,
    });
    if (res) setAnalysis(res.analysis as string);
  };

  return (
    <div className="space-y-6">
      {/* Summary Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Total Raised"
          value={fmt(totalRaised)}
          color="text-green-400"
        />
        <MetricCard
          label="Founder Equity"
          value={`${founderEquity.toFixed(1)}%`}
        />
        <MetricCard
          label="Investor Equity"
          value={`${investorEquity.toFixed(1)}%`}
        />
        <MetricCard
          label="Unallocated"
          value={`${unallocated.toFixed(1)}%`}
          color={
            unallocated > 15
              ? "text-green-400"
              : unallocated > 5
                ? "text-yellow-400"
                : "text-red-400"
          }
        />
      </div>

      {/* Cap Table Visualization */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
        <div className="text-sm text-neutral-400 mb-3">Cap Table</div>
        <div className="space-y-2">
          {capTable.map((row) => (
            <div key={row.name + row.type} className="flex items-center gap-3 text-sm">
              <span className="w-32 text-neutral-300 truncate">{row.name}</span>
              <span className="w-20 text-neutral-500 text-xs">{row.type}</span>
              <div className="flex-1 bg-neutral-800 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all ${
                    row.type === "Founder"
                      ? "bg-green-600"
                      : row.type === "Investor"
                        ? "bg-blue-600"
                        : "bg-neutral-600"
                  }`}
                  style={{ width: `${Math.max(row.equity, 1)}%` }}
                />
              </div>
              <span className="w-16 text-right text-white">
                {row.equity.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Founders Form */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Founders</div>
        <div className="grid grid-cols-5 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Name</label>
            <input
              type="text"
              placeholder="Founder name"
              value={fName}
              onChange={(e) => setFName(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Role</label>
            <input
              type="text"
              placeholder="CEO"
              value={fRole}
              onChange={(e) => setFRole(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Equity (%)
            </label>
            <input
              type="number"
              placeholder="0"
              value={fEquity}
              onChange={(e) => setFEquity(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Vesting (mo)
            </label>
            <input
              type="number"
              value={fVesting}
              onChange={(e) => setFVesting(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Cliff (mo)
            </label>
            <input
              type="number"
              value={fCliff}
              onChange={(e) => setFCliff(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <button
          onClick={addFounder}
          disabled={!fName.trim() || !fEquity}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Add Founder
        </button>

        {founders.length > 0 && (
          <table className="w-full text-sm mt-4">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                <th className="px-4 py-2">Name</th>
                <th className="px-4 py-2">Role</th>
                <th className="px-4 py-2 text-right">Equity</th>
                <th className="px-4 py-2 text-right">Vesting</th>
                <th className="px-4 py-2 text-right">Cliff</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {founders.map((f) => (
                <tr
                  key={f.id}
                  className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                >
                  <td className="px-4 py-2 text-white">{f.name}</td>
                  <td className="px-4 py-2 text-neutral-300">{f.role}</td>
                  <td className="px-4 py-2 text-right text-white">
                    {f.equityPercent}%
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {f.vestingMonths}mo
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {f.cliffMonths}mo
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() =>
                        saveFounders(founders.filter((x) => x.id !== f.id))
                      }
                      className="text-neutral-600 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Investors Form */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Investors & Funding</div>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Investor Name
            </label>
            <input
              type="text"
              placeholder="e.g. Y Combinator"
              value={iName}
              onChange={(e) => setIName(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Type</label>
            <div className="flex gap-1 flex-wrap">
              {INVESTOR_TYPES.map((t) => (
                <button
                  key={t.value}
                  onClick={() => setIType(t.value)}
                  className={`px-2 py-1 rounded-lg text-xs transition-colors ${
                    iType === t.value
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
            <label className="block text-xs text-neutral-500 mb-1">Round</label>
            <div className="flex gap-1 flex-wrap">
              {FUNDING_ROUNDS.map((r) => (
                <button
                  key={r.value}
                  onClick={() => setIRound(r.value)}
                  className={`px-2 py-1 rounded-lg text-xs transition-colors ${
                    iRound === r.value
                      ? "bg-green-600 text-white"
                      : "bg-neutral-800 text-neutral-400 hover:text-white"
                  }`}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Amount ($)
            </label>
            <input
              type="number"
              placeholder="0"
              value={iAmount}
              onChange={(e) => setIAmount(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Date</label>
            <input
              type="date"
              value={iDate}
              onChange={(e) => setIDate(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Equity (%)
            </label>
            <input
              type="number"
              placeholder="0"
              value={iEquity}
              onChange={(e) => setIEquity(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Terms
            </label>
            <input
              type="text"
              placeholder="e.g. $10M cap, 20% discount"
              value={iTerms}
              onChange={(e) => setITerms(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Notes</label>
            <input
              type="text"
              placeholder="Optional"
              value={iNotes}
              onChange={(e) => setINotes(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <button
          onClick={addInvestor}
          disabled={!iName.trim() || !iAmount}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Add Investor
        </button>

        {investors.length > 0 && (
          <table className="w-full text-sm mt-4">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                <th className="px-4 py-2">Name</th>
                <th className="px-4 py-2">Type</th>
                <th className="px-4 py-2">Round</th>
                <th className="px-4 py-2 text-right">Amount</th>
                <th className="px-4 py-2 text-right">Equity</th>
                <th className="px-4 py-2">Terms</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {investors.map((i) => (
                <tr
                  key={i.id}
                  className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                >
                  <td className="px-4 py-2 text-white">{i.name}</td>
                  <td className="px-4 py-2 text-neutral-400">{i.type}</td>
                  <td className="px-4 py-2 text-neutral-400">{i.round}</td>
                  <td className="px-4 py-2 text-right text-white">
                    {fmt(i.amount)}
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {i.equityPercent}%
                  </td>
                  <td className="px-4 py-2 text-neutral-400 text-xs truncate max-w-[200px]">
                    {i.terms || "—"}
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() =>
                        saveInvestors(investors.filter((x) => x.id !== i.id))
                      }
                      className="text-neutral-600 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* AI Analysis */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <button
          onClick={analyze}
          disabled={loading}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Assess Fundraising Readiness
        </button>
        <ErrorBanner error={error} />
        {loading && <LoadingSpinner />}
        {analysis && (
          <DocumentResult document={analysis} title="fundraising-assessment" />
        )}
      </div>

      {/* Deal Evaluator */}
      <DealEvaluator founders={founders} investors={investors} />
    </div>
  );
}

function DealEvaluator({
  founders,
  investors,
}: {
  founders: FounderEntry[];
  investors: InvestorEntry[];
}) {
  const { loading, error, callApi } = useFinanceApi();
  const [analysis, setAnalysis] = useState<string | null>(null);

  const [dealName, setDealName] = useState("");
  const [dealAmount, setDealAmount] = useState("");
  const [dealEquity, setDealEquity] = useState("");
  const [dealType, setDealType] = useState<InvestorEntry["type"]>("vc");
  const [dealRound, setDealRound] = useState<InvestorEntry["round"]>("seed");
  const [dealTerms, setDealTerms] = useState("");

  const impliedValuation =
    parseFloat(dealEquity) > 0
      ? parseFloat(dealAmount) / (parseFloat(dealEquity) / 100)
      : 0;

  const evaluate = async () => {
    if (!dealAmount || !dealEquity) return;
    const allData = gatherAllFinanceData();
    const res = await callApi({
      action: "evaluate_deal",
      investorName: dealName.trim() || "Unnamed Investor",
      investmentAmount: parseFloat(dealAmount),
      equityAsk: parseFloat(dealEquity),
      investorType: dealType,
      round: dealRound,
      additionalTerms: dealTerms.trim(),
      founders,
      investors,
      revenue: allData.revenue,
      expenses: allData.expenses,
      employees: allData.employees,
    });
    if (res) setAnalysis(res.analysis as string);
  };

  return (
    <div className="bg-neutral-900 border border-purple-800/50 rounded-lg p-6 space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-white">Deal Evaluator</h3>
        <p className="text-sm text-neutral-400 mt-1">
          Enter an investor&apos;s offer to get an AI assessment — fair valuation, realistic counter-offer, and how to deploy the capital.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <input
          placeholder="Investor name"
          value={dealName}
          onChange={(e) => setDealName(e.target.value)}
          className="bg-black border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white placeholder-neutral-600"
        />
        <select
          value={dealType}
          onChange={(e) => setDealType(e.target.value as InvestorEntry["type"])}
          className="bg-black border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="angel">Angel</option>
          <option value="vc">VC</option>
          <option value="safe">SAFE</option>
          <option value="convertible_note">Convertible Note</option>
          <option value="grant">Grant</option>
        </select>
        <select
          value={dealRound}
          onChange={(e) => setDealRound(e.target.value as InvestorEntry["round"])}
          className="bg-black border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="pre-seed">Pre-Seed</option>
          <option value="seed">Seed</option>
          <option value="series-a">Series A</option>
          <option value="series-b">Series B</option>
        </select>
        <div className="relative">
          <span className="absolute left-3 top-2.5 text-neutral-500 text-sm">$</span>
          <input
            type="number"
            placeholder="Investment amount"
            value={dealAmount}
            onChange={(e) => setDealAmount(e.target.value)}
            className="bg-black border border-neutral-700 rounded-lg pl-7 pr-3 py-2 text-sm text-white placeholder-neutral-600 w-full"
          />
        </div>
        <div className="relative">
          <input
            type="number"
            placeholder="Equity asked (%)"
            value={dealEquity}
            onChange={(e) => setDealEquity(e.target.value)}
            className="bg-black border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white placeholder-neutral-600 w-full"
          />
          <span className="absolute right-3 top-2.5 text-neutral-500 text-sm">%</span>
        </div>
        {impliedValuation > 0 && (
          <div className="flex items-center text-sm text-purple-400">
            Implied valuation: ${(impliedValuation / 1_000_000).toFixed(1)}M
          </div>
        )}
      </div>

      <textarea
        placeholder="Additional terms, conditions, or notes (optional)"
        value={dealTerms}
        onChange={(e) => setDealTerms(e.target.value)}
        rows={2}
        className="w-full bg-black border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white placeholder-neutral-600"
      />

      <button
        onClick={evaluate}
        disabled={loading || !dealAmount || !dealEquity}
        className="bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white text-sm px-5 py-2 rounded-lg transition-colors"
      >
        Evaluate This Deal
      </button>

      <ErrorBanner error={error} />
      {loading && <LoadingSpinner />}
      {analysis && (
        <DocumentResult document={analysis} title="deal-evaluation" />
      )}
    </div>
  );
}

function ValuationTab() {
  const { loading, error, callApi } = useFinanceApi();
  const [analysis, setAnalysis] = useState<string | null>(null);

  const [growthRate, setGrowthRate] = useState("50");
  const [discountRate, setDiscountRate] = useState("25");
  const [revenueMultiple, setRevenueMultiple] = useState("10");
  const [berkus, setBerkus] = useState({
    "Sound Idea": 250000,
    Prototype: 250000,
    "Quality Team": 250000,
    "Strategic Relationships": 250000,
    "Product Rollout": 250000,
  });

  const allData = useMemo(gatherAllFinanceData, []);

  const mrr = computeMRR(allData.revenue);
  const arr = computeARR(allData.revenue);
  const growth = computeMRRGrowth(allData.revenue);
  const burn = computeMonthlyBurn(allData.employees, allData.expenses);
  const totalRaised = computeTotalRaised(allData.investors);
  const grossMargin = computeGrossMargin(allData.revenue, allData.expenses);

  const revMultipleVal = arr * parseFloat(revenueMultiple || "10");
  const dcf = computeSimpleDCF(
    arr,
    parseFloat(growthRate || "50"),
    parseFloat(discountRate || "25")
  );
  const berkusTotal = Object.values(berkus).reduce((s, v) => s + v, 0);

  const ruleOf40 = (growth ?? 0) + (arr > 0 ? ((arr - burn * 12) / arr) * 100 : 0);

  const valuations = [
    { method: "Revenue Multiple", value: revMultipleVal, weight: arr > 0 ? 0.4 : 0 },
    { method: "DCF", value: dcf.npv, weight: arr > 0 ? 0.3 : 0 },
    { method: "Berkus", value: berkusTotal, weight: arr > 0 ? 0.1 : 0.5 },
  ].filter((v) => v.weight > 0);

  const totalWeight = valuations.reduce((s, v) => s + v.weight, 0);
  const compositeValuation =
    totalWeight > 0
      ? valuations.reduce((s, v) => s + v.value * (v.weight / totalWeight), 0)
      : berkusTotal;

  const dataCompleteness =
    [
      allData.revenue.length > 0,
      allData.employees.length > 0,
      allData.expenses.length > 0,
      allData.investors.length > 0,
      allData.founders.length > 0,
    ].filter(Boolean).length * 20;

  const analyze = async () => {
    const res = await callApi({
      action: "generate_valuation",
      revenue: allData.revenue,
      expenses: allData.expenses,
      employees: allData.employees,
      investors: allData.investors,
      founders: allData.founders,
      assumptions: {
        growthRate: parseFloat(growthRate),
        discountRate: parseFloat(discountRate),
        revenueMultiple: parseFloat(revenueMultiple),
        berkusScores: berkus,
      },
    });
    if (res) setAnalysis(res.analysis as string);
  };

  return (
    <div className="space-y-6">
      {/* Financial Summary */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="MRR" value={fmt(mrr)} color="text-green-400" />
        <MetricCard label="ARR" value={fmt(arr)} color="text-green-400" />
        <MetricCard
          label="Gross Margin"
          value={`${grossMargin.toFixed(1)}%`}
          color={grossMargin >= 75 ? "text-green-400" : grossMargin >= 50 ? "text-yellow-400" : "text-red-400"}
        />
      </div>

      {/* Assumptions */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Assumptions</div>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Annual Growth Rate (%)
            </label>
            <input
              type="number"
              value={growthRate}
              onChange={(e) => setGrowthRate(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Discount Rate (%)
            </label>
            <input
              type="number"
              value={discountRate}
              onChange={(e) => setDiscountRate(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Revenue Multiple (x)
            </label>
            <input
              type="number"
              value={revenueMultiple}
              onChange={(e) => setRevenueMultiple(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
      </div>

      {/* Valuation Methods */}
      <div className="grid grid-cols-2 gap-4">
        {/* Revenue Multiple */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
          <div className="text-sm text-neutral-400 mb-2">Revenue Multiple</div>
          <div className="text-2xl font-bold text-white">{fmt(revMultipleVal)}</div>
          <div className="text-xs text-neutral-500 mt-1">
            {fmt(arr)} ARR x {revenueMultiple}x
          </div>
        </div>

        {/* DCF */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
          <div className="text-sm text-neutral-400 mb-2">Simplified DCF</div>
          <div className="text-2xl font-bold text-white">{fmt(dcf.npv)}</div>
          <div className="text-xs text-neutral-500 mt-1">
            5yr NPV + terminal value at 3x
          </div>
        </div>
      </div>

      {/* DCF Projections */}
      {arr > 0 && (
        <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-4">
          <div className="text-sm text-neutral-400 mb-3">DCF Projections</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-neutral-400 text-left border-b border-neutral-800">
                <th className="px-3 py-2">Year</th>
                <th className="px-3 py-2 text-right">Revenue</th>
                <th className="px-3 py-2 text-right">Discounted</th>
              </tr>
            </thead>
            <tbody>
              {dcf.projections.map((p) => (
                <tr key={p.year} className="border-b border-neutral-800/50">
                  <td className="px-3 py-2 text-neutral-300">Year {p.year}</td>
                  <td className="px-3 py-2 text-right text-white">
                    {fmt(p.revenue)}
                  </td>
                  <td className="px-3 py-2 text-right text-neutral-300">
                    {fmt(p.discounted)}
                  </td>
                </tr>
              ))}
              <tr className="border-t border-neutral-700">
                <td className="px-3 py-2 text-neutral-300">Terminal</td>
                <td className="px-3 py-2" />
                <td className="px-3 py-2 text-right text-white">
                  {fmt(dcf.terminalValue)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Berkus Method */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm text-neutral-400">Berkus Method</div>
          <div className="text-lg font-bold text-white">{fmt(berkusTotal)}</div>
        </div>
        <div className="space-y-3">
          {Object.entries(berkus).map(([factor, value]) => (
            <div key={factor} className="flex items-center gap-4">
              <span className="w-48 text-sm text-neutral-300">{factor}</span>
              <input
                type="range"
                min={0}
                max={500000}
                step={25000}
                value={value}
                onChange={(e) =>
                  setBerkus({ ...berkus, [factor]: parseInt(e.target.value) })
                }
                className="flex-1 accent-green-600"
              />
              <span className="w-20 text-right text-sm text-white">
                {fmt(value)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* SaaS Health */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="Rule of 40"
          value={ruleOf40.toFixed(1)}
          color={ruleOf40 >= 40 ? "text-green-400" : "text-yellow-400"}
          sub={ruleOf40 >= 40 ? "Passing" : "Below threshold"}
        />
        <MetricCard
          label="Monthly Burn"
          value={fmt(burn)}
          color="text-red-400"
        />
        <MetricCard
          label="Total Raised"
          value={fmt(totalRaised)}
        />
      </div>

      {/* Composite */}
      <div className="bg-neutral-900 border border-green-500/30 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-2">Composite Valuation</div>
        <div className="text-3xl font-bold text-green-400">
          {fmt(compositeValuation)}
        </div>
        <div className="flex items-center gap-3 mt-2">
          <span className="text-xs text-neutral-400">Data Completeness:</span>
          <div className="flex-1 max-w-[200px] bg-neutral-800 rounded-full h-2">
            <div
              className="bg-green-600 h-2 rounded-full"
              style={{ width: `${dataCompleteness}%` }}
            />
          </div>
          <span className="text-xs text-neutral-400">{dataCompleteness}%</span>
        </div>
        <div className="text-xs text-neutral-500 mt-2">
          Weighted: {valuations.map((v) => `${v.method} (${(v.weight / totalWeight * 100).toFixed(0)}%)`).join(" + ")}
        </div>
      </div>

      {/* AI Report */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <button
          onClick={analyze}
          disabled={loading}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Generate Valuation Report
        </button>
        <ErrorBanner error={error} />
        {loading && <LoadingSpinner />}
        {analysis && (
          <DocumentResult document={analysis} title="valuation-report" />
        )}
      </div>
    </div>
  );
}

function PricingTab() {
  const { data: competitors, save: saveCompetitors } = useFinanceData<
    CompetitorEntry[]
  >("alum_finance_competitors_v1", []);
  const { loading, error, callApi } = useFinanceApi();
  const [analysis, setAnalysis] = useState<string | null>(null);

  const [cName, setCName] = useState("");
  const [cFree, setCFree] = useState("0");
  const [cPro, setCPro] = useState("");
  const [cEnterprise, setCEnterprise] = useState("");
  const [cNotes, setCNotes] = useState("");

  const [targetMargin, setTargetMargin] = useState("75");
  const [customerCount, setCustomerCount] = useState("");
  const [growthTarget, setGrowthTarget] = useState("100");

  const avgPro =
    competitors.length > 0
      ? competitors.reduce((s, c) => s + c.proPrice, 0) / competitors.length
      : 0;
  const avgEnterprise =
    competitors.length > 0
      ? competitors.reduce((s, c) => s + c.enterprisePrice, 0) /
        competitors.length
      : 0;

  const addCompetitor = () => {
    if (!cName.trim() || !cPro) return;
    saveCompetitors([
      {
        id: uid(),
        name: cName.trim(),
        freePrice: parseFloat(cFree) || 0,
        proPrice: parseFloat(cPro),
        enterprisePrice: parseFloat(cEnterprise) || 0,
        notes: cNotes.trim(),
      },
      ...competitors,
    ]);
    setCName("");
    setCFree("0");
    setCPro("");
    setCEnterprise("");
    setCNotes("");
  };

  const analyze = async () => {
    const allData = gatherAllFinanceData();
    const res = await callApi({
      action: "advise_pricing",
      competitors,
      expenses: allData.expenses,
      employees: allData.employees,
      revenue: allData.revenue,
      currentPricing: {
        free: 0,
        pro_monthly: 49,
        pro_yearly: 468,
        enterprise_monthly: 199,
        enterprise_yearly: 1908,
      },
      targetMargin: parseFloat(targetMargin) || undefined,
      customerCount: parseInt(customerCount) || undefined,
      growthTarget: parseFloat(growthTarget) || undefined,
    });
    if (res) setAnalysis(res.analysis as string);
  };

  return (
    <div className="space-y-6">
      {/* Current Pricing */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">
          Current AluminatAI Pricing
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-neutral-800 rounded-lg p-4 text-center">
            <div className="text-neutral-400 text-xs mb-1">Free</div>
            <div className="text-2xl font-bold text-white">$0</div>
            <div className="text-neutral-500 text-xs">per month</div>
          </div>
          <div className="bg-neutral-800 border border-green-500/30 rounded-lg p-4 text-center">
            <div className="text-green-400 text-xs mb-1">Pro</div>
            <div className="text-2xl font-bold text-white">$49</div>
            <div className="text-neutral-500 text-xs">
              per month ($39/mo yearly)
            </div>
          </div>
          <div className="bg-neutral-800 rounded-lg p-4 text-center">
            <div className="text-neutral-400 text-xs mb-1">Enterprise</div>
            <div className="text-2xl font-bold text-white">$199</div>
            <div className="text-neutral-500 text-xs">
              per month ($159/mo yearly)
            </div>
          </div>
        </div>
      </div>

      {/* Market Position */}
      {competitors.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <MetricCard
            label="Avg Competitor Pro Price"
            value={fmt(avgPro)}
            sub={
              49 > avgPro
                ? `AluminatAI is ${fmt(49 - avgPro)} above avg`
                : `AluminatAI is ${fmt(avgPro - 49)} below avg`
            }
            color={Math.abs(49 - avgPro) < 10 ? "text-green-400" : "text-yellow-400"}
          />
          <MetricCard
            label="Avg Competitor Enterprise Price"
            value={fmt(avgEnterprise)}
            sub={
              199 > avgEnterprise
                ? `AluminatAI is ${fmt(199 - avgEnterprise)} above avg`
                : `AluminatAI is ${fmt(avgEnterprise - 199)} below avg`
            }
            color={Math.abs(199 - avgEnterprise) < 30 ? "text-green-400" : "text-yellow-400"}
          />
        </div>
      )}

      {/* Competitor Entry */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Competitor Pricing</div>
        <div className="grid grid-cols-5 gap-4 mb-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Name</label>
            <input
              type="text"
              placeholder="Competitor"
              value={cName}
              onChange={(e) => setCName(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Free ($)
            </label>
            <input
              type="number"
              value={cFree}
              onChange={(e) => setCFree(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Pro ($)
            </label>
            <input
              type="number"
              placeholder="0"
              value={cPro}
              onChange={(e) => setCPro(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Enterprise ($)
            </label>
            <input
              type="number"
              placeholder="0"
              value={cEnterprise}
              onChange={(e) => setCEnterprise(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">Notes</label>
            <input
              type="text"
              placeholder="Differentiators"
              value={cNotes}
              onChange={(e) => setCNotes(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
        <button
          onClick={addCompetitor}
          disabled={!cName.trim() || !cPro}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Add Competitor
        </button>

        {competitors.length > 0 && (
          <table className="w-full text-sm mt-4">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                <th className="px-4 py-2">Name</th>
                <th className="px-4 py-2 text-right">Free</th>
                <th className="px-4 py-2 text-right">Pro</th>
                <th className="px-4 py-2 text-right">Enterprise</th>
                <th className="px-4 py-2">Notes</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody>
              {competitors.map((c) => (
                <tr
                  key={c.id}
                  className="border-b border-neutral-800/50 hover:bg-neutral-800/30"
                >
                  <td className="px-4 py-2 text-white">{c.name}</td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {fmt(c.freePrice)}
                  </td>
                  <td className="px-4 py-2 text-right text-white">
                    {fmt(c.proPrice)}
                  </td>
                  <td className="px-4 py-2 text-right text-white">
                    {fmt(c.enterprisePrice)}
                  </td>
                  <td className="px-4 py-2 text-neutral-400 text-xs truncate max-w-[200px]">
                    {c.notes || "—"}
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() =>
                        saveCompetitors(
                          competitors.filter((x) => x.id !== c.id)
                        )
                      }
                      className="text-neutral-600 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pricing Inputs */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Analysis Inputs</div>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Target Gross Margin (%)
            </label>
            <input
              type="number"
              value={targetMargin}
              onChange={(e) => setTargetMargin(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Current Customer Count
            </label>
            <input
              type="number"
              placeholder="0"
              value={customerCount}
              onChange={(e) => setCustomerCount(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
          <div>
            <label className="block text-xs text-neutral-500 mb-1">
              Annual Growth Target (%)
            </label>
            <input
              type="number"
              value={growthTarget}
              onChange={(e) => setGrowthTarget(e.target.value)}
              className={INPUT_CLS}
            />
          </div>
        </div>
      </div>

      {/* AI Analysis */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <button
          onClick={analyze}
          disabled={loading}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Get Pricing Recommendations
        </button>
        <ErrorBanner error={error} />
        {loading && <LoadingSpinner />}
        {analysis && (
          <DocumentResult document={analysis} title="pricing-recommendations" />
        )}
      </div>
    </div>
  );
}

function ReportsTab() {
  const { loading, error, callApi } = useFinanceApi();
  const [reportType, setReportType] = useState<
    "pnl" | "cashflow" | "board" | "metrics"
  >("board");
  const [document, setDocument] = useState<string | null>(null);

  const allData = useMemo(gatherAllFinanceData, []);

  const mrr = computeMRR(allData.revenue);
  const arr = computeARR(allData.revenue);
  const growth = computeMRRGrowth(allData.revenue);
  const burn = computeMonthlyBurn(allData.employees, allData.expenses);
  const totalRaised = computeTotalRaised(allData.investors);
  const grossMargin = computeGrossMargin(allData.revenue, allData.expenses);
  const totalEquity = computeTotalEquity(allData.founders, allData.investors);

  const generate = async () => {
    const res = await callApi({
      action: "generate_report",
      reportType,
      revenue: allData.revenue,
      expenses: allData.expenses,
      employees: allData.employees,
      investors: allData.investors,
      founders: allData.founders,
    });
    if (res) setDocument(res.document as string);
  };

  const REPORT_TYPES: {
    value: "pnl" | "cashflow" | "board" | "metrics";
    label: string;
  }[] = [
    { value: "pnl", label: "P&L Statement" },
    { value: "cashflow", label: "Cash Flow" },
    { value: "board", label: "Board Report" },
    { value: "metrics", label: "Metrics Dashboard" },
  ];

  return (
    <div className="space-y-6">
      {/* Quick Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="MRR" value={fmt(mrr)} color="text-green-400" />
        <MetricCard label="ARR" value={fmt(arr)} color="text-green-400" />
        <MetricCard label="MRR Growth" value={fmtPct(growth)} color={growth !== null && growth >= 0 ? "text-green-400" : "text-red-400"} />
      </div>
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Burn Rate" value={fmt(burn)} color="text-red-400" />
        <MetricCard label="Total Raised" value={fmt(totalRaised)} />
        <MetricCard
          label="Gross Margin"
          value={`${grossMargin.toFixed(1)}%`}
          color={grossMargin >= 75 ? "text-green-400" : "text-yellow-400"}
        />
        <MetricCard
          label="Equity Allocated"
          value={`${totalEquity.toFixed(1)}%`}
          sub={`${(100 - totalEquity).toFixed(1)}% unallocated`}
        />
      </div>

      {/* Report Selector */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
        <div className="text-sm text-neutral-400 mb-4">Generate Report</div>
        <div className="flex gap-2 mb-4">
          {REPORT_TYPES.map((r) => (
            <button
              key={r.value}
              onClick={() => {
                setReportType(r.value);
                setDocument(null);
              }}
              className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                reportType === r.value
                  ? "bg-green-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
        <button
          onClick={generate}
          disabled={loading}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Generate {REPORT_TYPES.find((r) => r.value === reportType)?.label}
        </button>
        <ErrorBanner error={error} />
        {loading && <LoadingSpinner />}
        {document && (
          <DocumentResult
            document={document}
            title={`${reportType}-report`}
          />
        )}
      </div>
    </div>
  );
}

// ── Main Page ──

export default function FinancialAdvisorPage() {
  const [tab, setTab] = useState<TabId>("revenue");

  return (
    <div className="bg-neutral-950 text-neutral-100 p-6 min-h-screen">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-1">
          AI Financial Advisor
        </h1>
        <p className="text-neutral-400 text-sm mb-6">
          Track finances, model valuations, and get AI-powered financial
          guidance
        </p>

        {/* Tab Bar */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                tab === t.id
                  ? "bg-green-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {tab === "revenue" && <RevenueTab />}
        {tab === "expenses" && <ExpensesTab />}
        {tab === "investors" && <InvestorsTab />}
        {tab === "valuation" && <ValuationTab />}
        {tab === "pricing" && <PricingTab />}
        {tab === "reports" && <ReportsTab />}
      </div>
    </div>
  );
}
