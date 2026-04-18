// POST /api/reports/carbon
// Cookie-auth. Generates a GHG Protocol Scope 2 carbon report in CSV, JSON, or PDF.
// Node.js runtime required for PDF generation.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "nodejs";

const KWH_RATE = 0.12;
const METHODOLOGY = "GHG Protocol Corporate Standard — Location-Based Method";
const EMISSION_FACTOR_SOURCE = "Electricity Maps — lifecycle";
const SCOPE = "Scope 2 (indirect — purchased electricity)";

interface ReportBody {
  format: "csv" | "json" | "pdf";
  start_date: string;
  end_date: string;
  cluster_tag?: string;
}

interface JobRow {
  job_id: string | null;
  model_tag: string | null;
  gpu_name: string | null;
  grid_zone: string | null;
  start_time: string;
  end_time: string;
  gpu_count: number;
  scope_2_kwh: number;
  scope_2_co2e_kg: number | null;
  avg_carbon_g_per_kwh: number | null;
}

async function buildJobRows(
  supabase: ReturnType<typeof createSupabaseServerClient>,
  userId: string,
  startDate: string,
  endDate: string,
  clusterTag?: string
): Promise<JobRow[]> {
  let query = supabase
    .from("gpu_metrics")
    .select(
      "job_id, model_tag, gpu_name, grid_zone, " +
        "energy_delta_j, gpu_fraction, carbon_g_per_kwh, gpu_uuid, time"
    )
    .eq("user_id", userId)
    .gte("time", startDate)
    .lte("time", endDate)
    .not("job_id", "is", null)
    .limit(10000);

  if (clusterTag) query = query.eq("cluster_tag", clusterTag);

  const { data: rawData, error } = await query;
  if (error) throw new Error(error.message);
  const data = (rawData ?? []) as unknown as Record<string, unknown>[];

  // Group by (job_id, model_tag, gpu_name, grid_zone)
  type Key = string;
  const map = new Map<
    Key,
    {
      job_id: string;
      model_tag: string | null;
      gpu_name: string | null;
      grid_zone: string | null;
      gpu_uuids: Set<string>;
      energy_j: number;
      co2e_g: number | null;
      carbon_samples: number;
      min_time: string;
      max_time: string;
    }
  >();

  for (const r of data) {
    if (!r.job_id) continue;
    const frac = r.gpu_fraction ?? 1;
    const ej = (r.energy_delta_j ?? 0) * frac;
    const key: Key = `${r.job_id}|||${r.model_tag}|||${r.gpu_name}|||${r.grid_zone}`;

    let agg = map.get(key);
    if (!agg) {
      agg = {
        job_id: r.job_id,
        model_tag: r.model_tag,
        gpu_name: r.gpu_name,
        grid_zone: r.grid_zone,
        gpu_uuids: new Set(),
        energy_j: 0,
        co2e_g: null,
        carbon_samples: 0,
        min_time: r.time,
        max_time: r.time,
      };
      map.set(key, agg);
    }

    if (r.gpu_uuid) agg.gpu_uuids.add(r.gpu_uuid);
    agg.energy_j += ej;

    if (r.carbon_g_per_kwh != null) {
      agg.co2e_g = (agg.co2e_g ?? 0) + (ej / 3_600_000) * r.carbon_g_per_kwh;
      agg.carbon_samples++;
    }

    if (r.time < agg.min_time) agg.min_time = r.time;
    if (r.time > agg.max_time) agg.max_time = r.time;
  }

  return [...map.values()].map((a) => ({
    job_id: a.job_id,
    model_tag: a.model_tag,
    gpu_name: a.gpu_name,
    grid_zone: a.grid_zone,
    start_time: a.min_time,
    end_time: a.max_time,
    gpu_count: a.gpu_uuids.size,
    scope_2_kwh: a.energy_j / 3_600_000,
    scope_2_co2e_kg: a.co2e_g != null ? a.co2e_g / 1000 : null,
    avg_carbon_g_per_kwh: a.carbon_samples > 0 ? (a.co2e_g ?? 0) / (a.energy_j / 3_600_000) : null,
  }));
}

function buildCsv(rows: JobRow[], startDate: string, endDate: string): string {
  const headers = [
    "job_id",
    "model_tag",
    "gpu_arch",
    "grid_zone",
    "start_time",
    "end_time",
    "gpu_count",
    "scope_2_kwh",
    "scope_2_co2e_kg",
    "avg_carbon_g_per_kwh",
    "cost_usd",
    "methodology",
    "emission_factor_source",
    "scope",
  ].join(",");

  const lines = rows.map((r) =>
    [
      r.job_id ?? "",
      r.model_tag ?? "",
      r.gpu_name ?? "",
      r.grid_zone ?? "",
      r.start_time,
      r.end_time,
      r.gpu_count,
      r.scope_2_kwh.toFixed(6),
      r.scope_2_co2e_kg?.toFixed(6) ?? "",
      r.avg_carbon_g_per_kwh?.toFixed(2) ?? "",
      (r.scope_2_kwh * KWH_RATE).toFixed(4),
      `"${METHODOLOGY}"`,
      `"${EMISSION_FACTOR_SOURCE}"`,
      `"${SCOPE}"`,
    ].join(",")
  );

  // Metadata header comment
  const meta = [
    `# AluminatiAI Carbon Report`,
    `# Reporting period: ${startDate} to ${endDate}`,
    `# Methodology: ${METHODOLOGY}`,
    `# Emission factor source: ${EMISSION_FACTOR_SOURCE}`,
    `# Scope: ${SCOPE}`,
    `# Generated at: ${new Date().toISOString()}`,
  ].join("\n");

  return `${meta}\n${headers}\n${lines.join("\n")}`;
}

function buildJson(rows: JobRow[], userId: string, startDate: string, endDate: string) {
  const totalKwh = rows.reduce((s, r) => s + r.scope_2_kwh, 0);
  const totalCo2eKg = rows.reduce((s, r) => s + (r.scope_2_co2e_kg ?? 0), 0);

  return {
    metadata: {
      org_id: userId,
      reporting_period: { start: startDate, end: endDate },
      methodology: METHODOLOGY,
      emission_factor_source: EMISSION_FACTOR_SOURCE,
      scope: SCOPE,
      generated_at: new Date().toISOString(),
      standards: ["EU AI Act energy disclosure", "California SB 253", "SEC Scope 1/2"],
    },
    summary: {
      total_scope_2_kwh: totalKwh,
      total_scope_2_co2e_kg: totalCo2eKg,
      total_cost_usd: totalKwh * KWH_RATE,
      job_count: rows.length,
    },
    jobs: rows.map((r) => ({
      ...r,
      cost_usd: r.scope_2_kwh * KWH_RATE,
      methodology: METHODOLOGY,
      emission_factor_source: EMISSION_FACTOR_SOURCE,
    })),
  };
}

async function buildPdf(rows: JobRow[], userId: string, startDate: string, endDate: string): Promise<Buffer> {
  // Dynamic import to avoid edge runtime issues
  const { default: ReactPDF, Document, Page, Text, View, StyleSheet } = await import("@react-pdf/renderer");
  const React = (await import("react")).default;

  const styles = StyleSheet.create({
    page: { padding: 40, fontSize: 9, fontFamily: "Helvetica", color: "#1a1a1a" },
    coverTitle: { fontSize: 22, fontWeight: "bold", marginBottom: 6 },
    coverSubtitle: { fontSize: 11, color: "#555", marginBottom: 24 },
    section: { marginBottom: 16 },
    sectionTitle: { fontSize: 11, fontWeight: "bold", marginBottom: 6, borderBottomWidth: 1, borderBottomColor: "#ddd", paddingBottom: 3 },
    row: { flexDirection: "row", marginBottom: 2 },
    label: { width: 180, color: "#555" },
    value: { flex: 1 },
    tableHeader: { flexDirection: "row", backgroundColor: "#f0f0f0", padding: 4, marginBottom: 2, fontWeight: "bold" },
    tableRow: { flexDirection: "row", padding: "3 4", borderBottomWidth: 0.5, borderBottomColor: "#e0e0e0" },
    col1: { width: 90 },
    col2: { width: 70 },
    col3: { width: 60 },
    col4: { width: 50 },
    col5: { width: 55 },
    col6: { width: 45 },
    footer: { position: "absolute", bottom: 24, left: 40, right: 40, fontSize: 7, color: "#999", textAlign: "center" },
    summaryBox: { backgroundColor: "#f8f8f8", border: "1 solid #ddd", padding: 10, borderRadius: 4, marginBottom: 16 },
    summaryRow: { flexDirection: "row", justifyContent: "space-between", marginBottom: 4 },
    summaryLabel: { color: "#555" },
    summaryValue: { fontWeight: "bold" },
  });

  const totalKwh = rows.reduce((s, r) => s + r.scope_2_kwh, 0);
  const totalCo2eKg = rows.reduce((s, r) => s + (r.scope_2_co2e_kg ?? 0), 0);
  const generatedAt = new Date().toISOString();

  const doc = React.createElement(
    Document,
    {},
    // Cover page
    React.createElement(
      Page,
      { size: "A4", style: styles.page },
      React.createElement(View, { style: { marginBottom: 32 } },
        React.createElement(Text, { style: styles.coverTitle }, "AluminatiAI Carbon Report"),
        React.createElement(Text, { style: styles.coverSubtitle }, `Reporting Period: ${startDate} — ${endDate}`),
        React.createElement(Text, { style: { fontSize: 9, color: "#888" } }, `Generated: ${generatedAt}`)
      ),
      // Executive summary
      React.createElement(View, { style: styles.summaryBox },
        React.createElement(Text, { style: { ...styles.sectionTitle, borderBottomWidth: 0 } }, "Executive Summary — Scope 2 Emissions"),
        React.createElement(View, { style: styles.summaryRow },
          React.createElement(Text, { style: styles.summaryLabel }, "Total Energy Consumed"),
          React.createElement(Text, { style: styles.summaryValue }, `${totalKwh.toFixed(4)} kWh`)
        ),
        React.createElement(View, { style: styles.summaryRow },
          React.createElement(Text, { style: styles.summaryLabel }, "Total CO₂e (Scope 2)"),
          React.createElement(Text, { style: styles.summaryValue }, `${totalCo2eKg.toFixed(4)} kg CO₂eq`)
        ),
        React.createElement(View, { style: styles.summaryRow },
          React.createElement(Text, { style: styles.summaryLabel }, "Estimated Electricity Cost"),
          React.createElement(Text, { style: styles.summaryValue }, `$${(totalKwh * KWH_RATE).toFixed(2)} USD`)
        ),
        React.createElement(View, { style: styles.summaryRow },
          React.createElement(Text, { style: styles.summaryLabel }, "Job Count"),
          React.createElement(Text, { style: styles.summaryValue }, String(rows.length))
        )
      ),
      // Per-job table
      React.createElement(View, { style: styles.section },
        React.createElement(Text, { style: styles.sectionTitle }, "Per-Job Detail"),
        React.createElement(View, { style: styles.tableHeader },
          React.createElement(Text, { style: styles.col1 }, "Job ID"),
          React.createElement(Text, { style: styles.col2 }, "Model"),
          React.createElement(Text, { style: styles.col3 }, "GPU"),
          React.createElement(Text, { style: styles.col4 }, "Zone"),
          React.createElement(Text, { style: styles.col5 }, "kWh"),
          React.createElement(Text, { style: styles.col6 }, "CO₂e kg")
        ),
        ...rows.slice(0, 40).map((r, i) =>
          React.createElement(View, { key: i, style: styles.tableRow },
            React.createElement(Text, { style: { ...styles.col1, fontSize: 7 } }, (r.job_id ?? "").slice(0, 16)),
            React.createElement(Text, { style: styles.col2 }, (r.model_tag ?? "—").slice(0, 14)),
            React.createElement(Text, { style: styles.col3 }, (r.gpu_name ?? "—").slice(0, 12)),
            React.createElement(Text, { style: styles.col4 }, r.grid_zone ?? "—"),
            React.createElement(Text, { style: styles.col5 }, r.scope_2_kwh.toFixed(5)),
            React.createElement(Text, { style: styles.col6 }, r.scope_2_co2e_kg?.toFixed(5) ?? "—")
          )
        ),
        rows.length > 40 ? React.createElement(Text, { style: { color: "#888", marginTop: 4 } }, `... and ${rows.length - 40} more rows (see CSV/JSON export for full data)`) : null
      ),
      // Methodology
      React.createElement(View, { style: styles.section },
        React.createElement(Text, { style: styles.sectionTitle }, "Methodology"),
        React.createElement(View, { style: styles.row },
          React.createElement(Text, { style: styles.label }, "Standard:"),
          React.createElement(Text, { style: styles.value }, METHODOLOGY)
        ),
        React.createElement(View, { style: styles.row },
          React.createElement(Text, { style: styles.label }, "Emission factor source:"),
          React.createElement(Text, { style: styles.value }, EMISSION_FACTOR_SOURCE)
        ),
        React.createElement(View, { style: styles.row },
          React.createElement(Text, { style: styles.label }, "Scope:"),
          React.createElement(Text, { style: styles.value }, SCOPE)
        ),
        React.createElement(View, { style: styles.row },
          React.createElement(Text, { style: styles.label }, "Accounting method:"),
          React.createElement(Text, { style: styles.value }, "Location-based (grid average emission factor)")
        ),
        React.createElement(View, { style: { ...styles.row, marginTop: 6 } },
          React.createElement(Text, { style: { color: "#777" } },
            "Energy measured by AluminatiAI GPU agent (NVML). Carbon intensity sourced from " +
            "Electricity Maps at time of consumption and locked into each measurement record " +
            "to ensure historical accuracy."
          )
        )
      ),
      // Footer
      React.createElement(
        Text,
        { style: styles.footer },
        "Generated by AluminatiAI · GHG Protocol Scope 2 (location-based) · EU AI Act ready · SB 253 aligned · SEC Scope 1/2"
      )
    )
  );

  const stream = await ReactPDF.renderToBuffer(doc);
  return Buffer.from(stream);
}

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: ReportBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { format, start_date, end_date, cluster_tag } = body;

  if (!format || !["csv", "json", "pdf"].includes(format)) {
    return NextResponse.json({ error: "format must be csv, json, or pdf" }, { status: 400 });
  }
  if (!start_date || !end_date) {
    return NextResponse.json({ error: "start_date and end_date are required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  let rows: JobRow[];
  try {
    rows = await buildJobRows(supabase, user.id, start_date, end_date, cluster_tag);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }

  const slug = `${start_date.slice(0, 10)}-${end_date.slice(0, 10)}`;

  if (format === "csv") {
    const csv = buildCsv(rows, start_date, end_date);
    return new NextResponse(csv, {
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": `attachment; filename="aluminatai-carbon-${slug}.csv"`,
      },
    });
  }

  if (format === "json") {
    const json = buildJson(rows, user.id, start_date, end_date);
    return new NextResponse(JSON.stringify(json, null, 2), {
      headers: {
        "Content-Type": "application/json",
        "Content-Disposition": `attachment; filename="aluminatai-carbon-${slug}.json"`,
      },
    });
  }

  // PDF
  try {
    const pdfBuffer = await buildPdf(rows, user.id, start_date, end_date);
    return new NextResponse(pdfBuffer, {
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="aluminatai-carbon-${slug}.pdf"`,
      },
    });
  } catch (err) {
    return NextResponse.json({ error: `PDF generation failed: ${String(err)}` }, { status: 500 });
  }
}
