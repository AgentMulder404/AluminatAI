"use client";

interface DateRangePickerProps {
  from: string;
  to: string;
  onChange: (from: string, to: string) => void;
}

function daysAgo(n: number): string {
  const d = new Date(Date.now() - n * 86400000);
  return d.toISOString().slice(0, 10);
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function yearStart(): string {
  return `${new Date().getUTCFullYear()}-01-01`;
}

const PRESETS: Array<{ label: string; from: string; to: string }> = [
  { label: "7d", from: daysAgo(7), to: today() },
  { label: "30d", from: daysAgo(30), to: today() },
  { label: "90d", from: daysAgo(90), to: today() },
  { label: "YTD", from: yearStart(), to: today() },
];

export default function DateRangePicker({ from, to, onChange }: DateRangePickerProps) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {PRESETS.map((p) => (
        <button
          key={p.label}
          onClick={() => onChange(p.from, p.to)}
          className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
            from === p.from && to === p.to
              ? "bg-indigo-600 text-white"
              : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
          }`}
        >
          {p.label}
        </button>
      ))}
      <input
        type="date"
        value={from}
        onChange={(e) => onChange(e.target.value, to)}
        className="bg-neutral-800 text-neutral-300 text-xs rounded-md px-2 py-1 border border-neutral-700 focus:border-indigo-500 outline-none"
      />
      <span className="text-neutral-500 text-xs">to</span>
      <input
        type="date"
        value={to}
        onChange={(e) => onChange(from, e.target.value)}
        className="bg-neutral-800 text-neutral-300 text-xs rounded-md px-2 py-1 border border-neutral-700 focus:border-indigo-500 outline-none"
      />
    </div>
  );
}
