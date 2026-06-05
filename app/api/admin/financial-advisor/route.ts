import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import Anthropic from "@anthropic-ai/sdk";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export const maxDuration = 60;

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

const FINANCIAL_ADVISOR_SYSTEM = `
You are an expert CFO and financial advisor specializing in early-stage SaaS startups. You are advising NemulAI, a GPU cost intelligence and energy monitoring SaaS platform.

Company context:
- Product: Python agent for GPU monitoring (power, temperature, utilization) + web dashboard for cost attribution, benchmarking, and chargeback
- Pricing: Free ($0), Pro ($49/mo or $468/yr), Enterprise ($199/mo or $1,908/yr)
- Stage: Early-stage, pre-Series A
- Infrastructure: Serverless (Vercel/Next.js), Supabase PostgreSQL, Stripe billing
- Market: DevOps/MLOps teams, AI startups, data centers, research labs

Your analysis should be:
1. Data-driven: reference specific numbers from the provided financial data
2. Actionable: provide specific recommendations with dollar amounts and timelines
3. Honest: flag risks and concerns directly, do not sugarcoat
4. Benchmarked: compare against SaaS industry standards (median SaaS gross margin ~75%, Rule of 40, typical CAC payback 12-18 months, seed round median $3-5M at $10-20M post-money)
5. Professional: suitable for board presentations and investor meetings

Format all responses in clean markdown with headers, bullet points, and tables where appropriate.
`.trim();

function getClient() {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY not configured");
  }
  return new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
}

async function callClaude(
  system: string,
  userMessage: string,
  maxTokens = 4096
): Promise<string> {
  const client = getClient();
  const message = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: maxTokens,
    system,
    messages: [{ role: "user", content: userMessage }],
  });
  const block = message.content[0];
  if (block.type === "text") return block.text;
  throw new Error("Unexpected response format");
}

function formatCurrency(n: number): string {
  return `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

async function handleAnalyzeRevenue(body: Record<string, unknown>) {
  const revenue = body.revenue as Array<Record<string, unknown>>;
  if (!revenue || revenue.length === 0) {
    return NextResponse.json({ error: "No revenue data provided" }, { status: 400 });
  }

  const lines = revenue.map(
    (r) => `- ${r.date}: ${r.label} (${r.source}) — ${formatCurrency(r.amount as number)}${r.recurring ? " [recurring]" : " [one-time]"}${r.notes ? ` — ${r.notes}` : ""}`
  ).join("\n");

  const userMessage = `Analyze the following revenue data for NemulAI and provide:
1. Revenue trend analysis (month-over-month growth, acceleration/deceleration)
2. Revenue composition and concentration risk
3. MRR/ARR calculation and trajectory
4. Next quarter revenue forecast with assumptions
5. Revenue diversification recommendations
6. Key risks to revenue growth

Revenue entries:
${lines}`;

  const analysis = await callClaude(FINANCIAL_ADVISOR_SYSTEM, userMessage);
  return NextResponse.json({ analysis, generatedAt: new Date().toISOString() });
}

async function handleOptimizeCosts(body: Record<string, unknown>) {
  const employees = body.employees as Array<Record<string, unknown>> ?? [];
  const expenses = body.expenses as Array<Record<string, unknown>> ?? [];
  const revenue = body.revenue as Array<Record<string, unknown>> ?? [];

  const empLines = employees.map(
    (e) => `- ${e.name} (${e.role}, ${e.type}): ${formatCurrency(e.monthlySalary as number)}/mo, equity: ${e.equityPercent}%, ${e.active ? "active" : "inactive"}, started: ${e.startDate}`
  ).join("\n");

  const expLines = expenses.map(
    (e) => `- ${e.date}: ${e.label} (${e.category}) — ${formatCurrency(e.amount as number)}/mo${e.recurring ? " [recurring]" : " [one-time]"}${e.notes ? ` — ${e.notes}` : ""}`
  ).join("\n");

  const revLines = revenue.map(
    (r) => `- ${r.date}: ${r.label} — ${formatCurrency(r.amount as number)}${r.recurring ? " [recurring]" : ""}`
  ).join("\n");

  const userMessage = `Analyze the following cost structure for NemulAI and provide:
1. Monthly burn rate breakdown (payroll vs. operating expenses)
2. Cost optimization opportunities with estimated savings
3. Compensation benchmarking against early-stage SaaS startups
4. Optimal team structure recommendations for current stage
5. Expense categorization health check
6. Runway projection and cash conservation strategies

Team & Payroll:
${empLines || "No employees entered yet."}

Operating Expenses:
${expLines || "No expenses entered yet."}

Revenue (for context):
${revLines || "No revenue entered yet."}`;

  const analysis = await callClaude(FINANCIAL_ADVISOR_SYSTEM, userMessage);
  return NextResponse.json({ analysis, generatedAt: new Date().toISOString() });
}

async function handleAssessFundraising(body: Record<string, unknown>) {
  const investors = body.investors as Array<Record<string, unknown>> ?? [];
  const founders = body.founders as Array<Record<string, unknown>> ?? [];
  const revenue = body.revenue as Array<Record<string, unknown>> ?? [];
  const expenses = body.expenses as Array<Record<string, unknown>> ?? [];
  const employees = body.employees as Array<Record<string, unknown>> ?? [];

  const founderLines = founders.map(
    (f) => `- ${f.name} (${f.role}): ${f.equityPercent}% equity, ${f.vestingMonths}mo vesting / ${f.cliffMonths}mo cliff`
  ).join("\n");

  const investorLines = investors.map(
    (i) => `- ${i.name} (${i.type}, ${i.round}): ${formatCurrency(i.amount as number)} for ${i.equityPercent}% equity${i.terms ? ` — terms: ${i.terms}` : ""}`
  ).join("\n");

  const userMessage = `Assess NemulAI's fundraising readiness and cap table health:

1. Fundraising readiness score (1-10) with justification
2. Cap table health analysis (dilution, founder control, option pool)
3. Recommended next round: size, target valuation, instrument type
4. Key metrics that need improvement before next raise
5. Suggested investor profile and outreach strategy
6. Pitch narrative strengths and weaknesses based on financials

Founders:
${founderLines || "No founders entered yet."}

Investors & Funding History:
${investorLines || "No investors entered yet."}

Revenue (${revenue.length} entries): Total MRR from recurring = ${formatCurrency(
    (revenue as Array<Record<string, unknown>>)
      .filter((r) => r.recurring)
      .reduce((sum, r) => sum + (r.amount as number), 0)
  )}

Team Size: ${employees.length} (${employees.filter((e) => e.active).length} active)
Monthly Burn: ${formatCurrency(
    employees.filter((e) => e.active).reduce((s, e) => s + (e.monthlySalary as number), 0) +
    (expenses as Array<Record<string, unknown>>).filter((e) => e.recurring).reduce((s, e) => s + (e.amount as number), 0)
  )}`;

  const analysis = await callClaude(FINANCIAL_ADVISOR_SYSTEM, userMessage, 6144);
  return NextResponse.json({ analysis, generatedAt: new Date().toISOString() });
}

async function handleEvaluateDeal(body: Record<string, unknown>) {
  const investorName = (body.investorName as string) ?? "Unnamed Investor";
  const investmentAmount = body.investmentAmount as number ?? 0;
  const equityAsk = body.equityAsk as number ?? 0;
  const investorType = (body.investorType as string) ?? "unknown";
  const round = (body.round as string) ?? "unknown";
  const additionalTerms = (body.additionalTerms as string) ?? "";

  const founders = body.founders as Array<Record<string, unknown>> ?? [];
  const investors = body.investors as Array<Record<string, unknown>> ?? [];
  const revenue = body.revenue as Array<Record<string, unknown>> ?? [];
  const expenses = body.expenses as Array<Record<string, unknown>> ?? [];
  const employees = body.employees as Array<Record<string, unknown>> ?? [];

  const founderLines = founders.map(
    (f) => `- ${f.name} (${f.role}): ${f.equityPercent}% equity`
  ).join("\n");

  const investorLines = investors.map(
    (i) => `- ${i.name} (${i.type}, ${i.round}): ${formatCurrency(i.amount as number)} for ${i.equityPercent}%`
  ).join("\n");

  const mrr = (revenue as Array<Record<string, unknown>>)
    .filter((r) => r.recurring)
    .reduce((sum, r) => sum + (r.amount as number), 0);

  const monthlyBurn =
    employees.filter((e) => e.active).reduce((s, e) => s + (e.monthlySalary as number), 0) +
    (expenses as Array<Record<string, unknown>>).filter((e) => e.recurring).reduce((s, e) => s + (e.amount as number), 0);

  const impliedValuation = equityAsk > 0 ? investmentAmount / (equityAsk / 100) : 0;

  const userMessage = `An investor has offered NemulAI a deal. Evaluate it thoroughly:

THE OFFER:
- Investor: ${investorName} (${investorType})
- Investment: ${formatCurrency(investmentAmount)}
- Equity asked: ${equityAsk}%
- Implied pre-money valuation: ${formatCurrency(impliedValuation)}
- Round: ${round}
${additionalTerms ? `- Additional terms/notes: ${additionalTerms}` : ""}

CURRENT CAP TABLE:
Founders:
${founderLines || "No founders entered yet."}

Existing Investors:
${investorLines || "No existing investors."}

CURRENT FINANCIALS:
- MRR: ${formatCurrency(mrr)}
- ARR: ${formatCurrency(mrr * 12)}
- Monthly Burn: ${formatCurrency(monthlyBurn)}
- Runway: ${monthlyBurn > 0 ? `${Math.round(mrr / monthlyBurn * 30)} days at current burn` : "N/A"}
- Team Size: ${employees.length} (${employees.filter((e) => e.active).length} active)

Please provide:
1. **Deal Assessment** — Is this a fair deal? Score the offer 1-10. Compare the implied valuation to typical benchmarks for this stage and these metrics.
2. **Realistic Counter-Offer** — What equity stake should you realistically give for this investment amount? What valuation should you target? Provide a specific counter-proposal.
3. **Dilution Impact** — Show what the cap table looks like post-deal at both the asked equity AND your recommended equity. How much founder control remains?
4. **Best Use of Funds** — Given NemulAI's current stage, metrics, and burn rate, provide a specific allocation plan for how to deploy this capital (e.g., X% engineering hires, Y% marketing, Z% runway buffer). Prioritize by impact.
5. **Negotiation Strategy** — Key leverage points, what to push back on, what terms to insist on (pro-rata rights, board seats, liquidation preference, anti-dilution).
6. **Walk-Away Analysis** — At what point does this deal become bad? What's the minimum acceptable terms?`;

  const analysis = await callClaude(FINANCIAL_ADVISOR_SYSTEM, userMessage, 8192);
  return NextResponse.json({ analysis, generatedAt: new Date().toISOString() });
}

async function handleGenerateValuation(body: Record<string, unknown>) {
  const revenue = body.revenue as Array<Record<string, unknown>> ?? [];
  const expenses = body.expenses as Array<Record<string, unknown>> ?? [];
  const employees = body.employees as Array<Record<string, unknown>> ?? [];
  const investors = body.investors as Array<Record<string, unknown>> ?? [];
  const founders = body.founders as Array<Record<string, unknown>> ?? [];
  const assumptions = body.assumptions as Record<string, unknown> ?? {};

  const recurringRevenue = revenue.filter((r) => r.recurring);
  const mrr = recurringRevenue.reduce((sum, r) => sum + (r.amount as number), 0);
  const arr = mrr * 12;
  const monthlyBurn =
    employees.filter((e) => e.active).reduce((s, e) => s + (e.monthlySalary as number), 0) +
    (expenses as Array<Record<string, unknown>>).filter((e) => e.recurring).reduce((s, e) => s + (e.amount as number), 0);
  const totalRaised = investors.reduce((s, i) => s + (i.amount as number), 0);
  const founderEquity = founders.reduce((s, f) => s + (f.equityPercent as number), 0);
  const investorEquity = investors.reduce((s, i) => s + (i.equityPercent as number), 0);

  const userMessage = `Generate a comprehensive valuation report for NemulAI using multiple methodologies:

Financial Summary:
- Current MRR: ${formatCurrency(mrr)}
- Current ARR: ${formatCurrency(arr)}
- Monthly Burn Rate: ${formatCurrency(monthlyBurn)}
- Total Raised: ${formatCurrency(totalRaised)}
- Team Size: ${employees.filter((e) => e.active).length}
- Founder Equity: ${founderEquity}%
- Investor Equity: ${investorEquity}%
- Revenue Entries: ${revenue.length} (${recurringRevenue.length} recurring)

Assumptions:
- Growth Rate: ${assumptions.growthRate ?? "not specified (estimate from data)"}%
- Discount Rate: ${assumptions.discountRate ?? 25}%
- Revenue Multiple: ${assumptions.revenueMultiple ?? 10}x

Berkus Scores (0-500000 each):
${assumptions.berkusScores ? Object.entries(assumptions.berkusScores as Record<string, number>).map(([k, v]) => `- ${k}: ${formatCurrency(v)}`).join("\n") : "Not provided — estimate based on company profile."}

Please provide:
1. **Revenue Multiple Valuation**: ARR × multiple, with justification for the multiple
2. **Simplified DCF**: 5-year projection with terminal value, show year-by-year table
3. **Berkus Method**: Score each of the 5 factors with justification
4. **SaaS Health Metrics**: Rule of 40, gross margin estimate, unit economics assessment
5. **Composite Valuation**: Weighted average with confidence level
6. **Comparable Companies**: Reference 3-5 similar SaaS companies and their valuations
7. **Key Value Drivers & Risks**: What increases/decreases the valuation`;

  const analysis = await callClaude(FINANCIAL_ADVISOR_SYSTEM, userMessage, 8192);
  return NextResponse.json({ analysis, generatedAt: new Date().toISOString() });
}

async function handleAdvisePricing(body: Record<string, unknown>) {
  const competitors = body.competitors as Array<Record<string, unknown>> ?? [];
  const expenses = body.expenses as Array<Record<string, unknown>> ?? [];
  const employees = body.employees as Array<Record<string, unknown>> ?? [];
  const revenue = body.revenue as Array<Record<string, unknown>> ?? [];
  const currentPricing = body.currentPricing as Record<string, number>;
  const targetMargin = body.targetMargin as number | undefined;
  const customerCount = body.customerCount as number | undefined;
  const growthTarget = body.growthTarget as number | undefined;

  const compLines = competitors.map(
    (c) => `- ${c.name}: Free=${formatCurrency(c.freePrice as number)}, Pro=${formatCurrency(c.proPrice as number)}, Enterprise=${formatCurrency(c.enterprisePrice as number)}${c.notes ? ` — ${c.notes}` : ""}`
  ).join("\n");

  const monthlyBurn =
    employees.filter((e) => e.active).reduce((s, e) => s + (e.monthlySalary as number), 0) +
    expenses.filter((e) => e.recurring).reduce((s, e) => s + (e.amount as number), 0);

  const mrr = revenue.filter((r) => r.recurring).reduce((s, r) => s + (r.amount as number), 0);

  const userMessage = `Analyze NemulAI's pricing strategy and provide recommendations:

Current NemulAI Pricing:
- Free: ${formatCurrency(currentPricing?.free ?? 0)}/mo
- Pro: ${formatCurrency(currentPricing?.pro_monthly ?? 49)}/mo (${formatCurrency(currentPricing?.pro_yearly ?? 468)}/yr)
- Enterprise: ${formatCurrency(currentPricing?.enterprise_monthly ?? 199)}/mo (${formatCurrency(currentPricing?.enterprise_yearly ?? 1908)}/yr)

Competitor Pricing:
${compLines || "No competitor data entered yet."}

Financial Context:
- Current MRR: ${formatCurrency(mrr)}
- Monthly Burn: ${formatCurrency(monthlyBurn)}
- Customer Count: ${customerCount ?? "not provided"}
- Target Gross Margin: ${targetMargin ?? "not specified"}%
- Growth Target: ${growthTarget ?? "not specified"}%

Please provide:
1. **Market Positioning Analysis**: Where NemulAI sits vs. competitors (premium, mid-market, budget)
2. **Price Elasticity Assessment**: Are current prices too high, too low, or right?
3. **Recommended Pricing Changes**: Specific new prices with justification
4. **Annual Discount Optimization**: Is the 20% annual discount optimal?
5. **Feature Gating Strategy**: What features should gate each tier?
6. **Upsell & Cross-sell Opportunities**: Revenue expansion strategies
7. **Pricing Experimentation Plan**: A/B tests to validate recommendations`;

  const analysis = await callClaude(FINANCIAL_ADVISOR_SYSTEM, userMessage, 6144);
  return NextResponse.json({ analysis, generatedAt: new Date().toISOString() });
}

async function handleGenerateReport(body: Record<string, unknown>) {
  const reportType = body.reportType as string;
  const revenue = body.revenue as Array<Record<string, unknown>> ?? [];
  const expenses = body.expenses as Array<Record<string, unknown>> ?? [];
  const employees = body.employees as Array<Record<string, unknown>> ?? [];
  const investors = body.investors as Array<Record<string, unknown>> ?? [];
  const founders = body.founders as Array<Record<string, unknown>> ?? [];
  const period = body.period as string | undefined;

  const mrr = revenue.filter((r) => r.recurring).reduce((s, r) => s + (r.amount as number), 0);
  const arr = mrr * 12;
  const payroll = employees.filter((e) => e.active).reduce((s, e) => s + (e.monthlySalary as number), 0);
  const opex = expenses.filter((e) => e.recurring).reduce((s, e) => s + (e.amount as number), 0);
  const totalRaised = investors.reduce((s, i) => s + (i.amount as number), 0);
  const totalRevenue = revenue.reduce((s, r) => s + (r.amount as number), 0);

  const reportPrompts: Record<string, string> = {
    pnl: `Generate a professional Profit & Loss (Income) Statement for NemulAI for period: ${period ?? "current"}.

Include:
- Revenue section broken down by source (subscriptions, consulting, grants, other)
- Cost of Revenue / COGS (infrastructure costs)
- Gross Profit and Gross Margin %
- Operating Expenses by category (payroll, tools, services, marketing, legal, other)
- EBITDA
- Net Income / Loss
- Month-over-month comparison if data allows

Format as a proper financial statement with line items and totals.`,

    cashflow: `Generate a Cash Flow Statement for NemulAI for period: ${period ?? "current"}.

Include:
- Cash Flow from Operations (revenue collected minus operating expenses)
- Cash Flow from Investing (if applicable)
- Cash Flow from Financing (funding rounds, loans)
- Net Cash Position
- Beginning and ending cash balance
- Cash runway projection

Format as a proper financial statement.`,

    board: `Generate a Board Report / Executive Financial Summary for NemulAI for period: ${period ?? "current"}.

Include:
- Executive Summary (3-5 bullet points)
- KPI Dashboard: MRR, ARR, growth rate, burn rate, runway, gross margin
- Financial Highlights and Lowlights
- Team update (headcount, key hires/departures)
- Cap table summary
- Strategic Priorities for next quarter
- Key Risks and Mitigations
- Ask / Recommendations from the board

Format professionally, suitable for board presentation.`,

    metrics: `Generate a SaaS Metrics Dashboard Report for NemulAI for period: ${period ?? "current"}.

Include:
- MRR and ARR with trend
- MRR Growth Rate (month-over-month)
- Burn Rate and Runway
- Gross Margin
- Revenue per Employee
- Burn Multiple (net burn / net new ARR)
- Rule of 40 score
- Unit Economics estimates (CAC, LTV, LTV:CAC if data available)
- Benchmarking against SaaS medians

Format with clear sections and comparative benchmarks.`,
  };

  const prompt = reportPrompts[reportType];
  if (!prompt) {
    return NextResponse.json({ error: `Unknown report type: ${reportType}` }, { status: 400 });
  }

  const financialContext = `
Financial Data Summary:
- MRR: ${formatCurrency(mrr)}
- ARR: ${formatCurrency(arr)}
- Monthly Payroll: ${formatCurrency(payroll)}
- Monthly OpEx: ${formatCurrency(opex)}
- Total Monthly Burn: ${formatCurrency(payroll + opex)}
- Total Revenue (all-time): ${formatCurrency(totalRevenue)}
- Total Raised: ${formatCurrency(totalRaised)}
- Active Employees: ${employees.filter((e) => e.active).length}
- Founders: ${founders.length}
- Investors: ${investors.length}

Revenue Breakdown:
${revenue.map((r) => `- ${r.date}: ${r.label} (${r.source}) — ${formatCurrency(r.amount as number)}${r.recurring ? " [recurring]" : ""}`).join("\n") || "No data."}

Expense Breakdown:
${expenses.map((e) => `- ${e.date}: ${e.label} (${e.category}) — ${formatCurrency(e.amount as number)}${e.recurring ? " [recurring]" : ""}`).join("\n") || "No data."}

Team:
${employees.map((e) => `- ${e.name} (${e.role}, ${e.type}): ${formatCurrency(e.monthlySalary as number)}/mo`).join("\n") || "No data."}
`;

  const document = await callClaude(
    FINANCIAL_ADVISOR_SYSTEM,
    `${prompt}\n\n${financialContext}`,
    8192
  );
  return NextResponse.json({
    document,
    reportType,
    generatedAt: new Date().toISOString(),
  });
}

export async function POST(req: NextRequest) {
  try {
    const cookieClient = await createSupabaseCookieClient();
    const {
      data: { user },
      error: authError,
    } = await cookieClient.auth.getUser();

    if (authError || !user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const rl = await rateLimit(`admin:${user.id}`, 30);
    if (!rl.success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429, headers: getRateLimitHeaders(rl) }
      );
    }

    const body = (await req.json()) as Record<string, unknown>;
    const action = body.action as string;

    switch (action) {
      case "analyze_revenue":
        return handleAnalyzeRevenue(body);
      case "optimize_costs":
        return handleOptimizeCosts(body);
      case "assess_fundraising":
        return handleAssessFundraising(body);
      case "evaluate_deal":
        return handleEvaluateDeal(body);
      case "generate_valuation":
        return handleGenerateValuation(body);
      case "advise_pricing":
        return handleAdvisePricing(body);
      case "generate_report":
        return handleGenerateReport(body);
      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (e) {
    console.error("[financial-advisor]", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Internal error" },
      { status: 500 }
    );
  }
}
