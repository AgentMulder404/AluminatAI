import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import Anthropic from "@anthropic-ai/sdk";

export const maxDuration = 60;

const ADMIN_EMAILS = (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

const ALUMINATAI_SECURITY_CONTEXT = `
AluminatAI Security Posture:
- Infrastructure: Vercel serverless (Next.js), no self-hosted servers
- Database: Supabase PostgreSQL with Row Level Security (RLS) policies on all user-facing tables
- Authentication: Cookie-based SSR auth via @supabase/ssr; service role key for server-side operations only
- API Auth: API keys with "alum_" prefix (~340 bits entropy), validated server-side via validateApiKey()
- Rate Limiting: Sliding window rate limiter (Upstash Redis primary, in-memory fallback) — 100 req/min ingest, 60 req/min dashboard
- Payments: Stripe integration with webhook signature verification
- Audit Logging: All admin actions logged to audit_log table with user_id, action, metadata, IP
- RBAC: Role-based access control (owner, admin, member, viewer) enforced at API layer
- Data Retention: Configurable retention policies, cron-based cleanup
- Encryption: API keys hashed with SHA-256 before storage; pgcrypto gen_random_bytes() for key generation
- Monitoring: Agent heartbeat system (5-min intervals), SLA dashboard tracking uptime/latency
- Vulnerability Disclosure: 90-day coordinated disclosure policy, contact@aluminatiai.com
- SOC 2: Not yet certified (in progress)
- GDPR: Data processing on EU Supabase region available; no formal DPA published yet
- Data Collected: GPU metrics (power, temperature, utilization), user email, Stripe billing info
- Third-party Integrations: Apify (prospect discovery), Resend (transactional email), Stripe (billing)
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

async function handleGenerateContract(body: Record<string, unknown>) {
  const { contractType, companyName, dealTerms, specialClauses, jurisdiction } =
    body;
  if (!contractType || !companyName) {
    return NextResponse.json(
      { error: "contractType and companyName are required" },
      { status: 400 }
    );
  }

  const templates: Record<string, string> = {
    MSA: "Master Service Agreement with: definitions, scope of services, payment terms (net-30), warranties, limitation of liability, indemnification, confidentiality, term and termination (30-day notice), governing law, and general provisions.",
    DPA: "GDPR-compliant Data Processing Addendum with: definitions (controller, processor, data subject), scope and purpose of processing, types of personal data, obligations of the processor, sub-processors, data subject rights, security measures, breach notification (72hr), data deletion/return, audit rights, and cross-border transfer mechanisms (SCCs).",
    SOW: "Statement of Work with: project overview, deliverables table (item, description, due date), timeline/milestones, acceptance criteria, resource allocation, assumptions and dependencies, change management process, and payment schedule tied to milestones.",
    NDA: "Mutual Non-Disclosure Agreement with: definition of confidential information, exclusions (public knowledge, independent development, prior possession), obligations of receiving party, permitted disclosures (legal compulsion), term (2 years), return/destruction of materials, remedies (injunctive relief), and standard carve-outs.",
  };

  const template = templates[contractType as string] || templates.MSA;

  const document = await callClaude(
    "You are a legal document drafter for AluminatAI, a GPU cost intelligence and energy monitoring SaaS platform. Generate professional, enforceable contracts in markdown format. Use clear section numbering. Include signature blocks at the end with Company Name, Authorized Signatory, Title, and Date fields for both parties.",
    `Generate a ${contractType} between AluminatAI, Inc. and ${companyName}.

Template structure: ${template}

${jurisdiction ? `Governing law: ${jurisdiction}` : "Governing law: State of Delaware, USA"}
${dealTerms ? `Deal terms and specifics:\n${dealTerms}` : ""}
${specialClauses ? `Additional clauses to include:\n${specialClauses}` : ""}

Use today's date as the effective date. AluminatAI's address is: AluminatAI, Inc., Delaware, USA.`
  );

  return NextResponse.json({
    document,
    contractType,
    generatedAt: new Date().toISOString(),
  });
}

async function handleRedlineContract(body: Record<string, unknown>) {
  const { contractText } = body;
  if (!contractText) {
    return NextResponse.json(
      { error: "contractText is required" },
      { status: 400 }
    );
  }

  const raw = await callClaude(
    `You are a legal contract reviewer for AluminatAI, a GPU cost intelligence SaaS company. Analyze contracts and identify clauses that are risky or unfavorable to AluminatAI. For each flagged item provide the exact clause text, risk level, category, explanation, and suggested counter-language.

Respond ONLY with valid JSON in this exact format:
{
  "summary": "brief overall assessment",
  "flags": [
    {
      "clause": "exact text from the contract",
      "riskLevel": "high" | "medium" | "low",
      "category": "liability" | "indemnity" | "ip" | "sla" | "termination" | "payment" | "data_privacy" | "other",
      "explanation": "why this is risky for AluminatAI",
      "suggestedLanguage": "proposed alternative clause text"
    }
  ]
}`,
    `Analyze this contract and flag risky clauses:\n\n${contractText}`
  );

  try {
    const parsed = JSON.parse(raw);
    return NextResponse.json(parsed);
  } catch {
    return NextResponse.json({ summary: raw, flags: [] });
  }
}

async function handleFillQuestionnaire(body: Record<string, unknown>) {
  const { questionnaireText } = body;
  if (!questionnaireText) {
    return NextResponse.json(
      { error: "questionnaireText is required" },
      { status: 400 }
    );
  }

  const raw = await callClaude(
    `You are a security compliance specialist for AluminatAI. Answer security questionnaires based on AluminatAI's actual security posture.

${ALUMINATAI_SECURITY_CONTEXT}

Parse the questionnaire and answer each question honestly. If something is not yet implemented, say so transparently (e.g., "In progress" or "Not yet — planned for Q3 2026").

Respond ONLY with valid JSON in this exact format:
{
  "answers": [
    {
      "question": "the original question text",
      "answer": "detailed, honest answer",
      "confidence": "high" | "medium" | "low"
    }
  ]
}`,
    `Answer each question in this security questionnaire:\n\n${questionnaireText}`
  );

  try {
    const parsed = JSON.parse(raw);
    return NextResponse.json(parsed);
  } catch {
    return NextResponse.json({ answers: [{ question: "Parse error", answer: raw, confidence: "low" }] });
  }
}

async function handleScanCompliance(body: Record<string, unknown>) {
  const { frameworks } = body;
  if (!frameworks || !Array.isArray(frameworks) || frameworks.length === 0) {
    return NextResponse.json(
      { error: "frameworks array is required (e.g., [\"GDPR\", \"CCPA\", \"SOC2\"])" },
      { status: 400 }
    );
  }

  const raw = await callClaude(
    `You are a compliance auditor evaluating AluminatAI's compliance posture.

${ALUMINATAI_SECURITY_CONTEXT}

Evaluate compliance with the requested frameworks. Be honest — mark items as "fail" or "needs_attention" when appropriate. This is an internal tool, not customer-facing.

Respond ONLY with valid JSON in this exact format:
{
  "frameworks": [
    {
      "name": "FRAMEWORK_NAME",
      "score": "X/Y requirements met",
      "items": [
        {
          "requirement": "specific requirement description",
          "status": "pass" | "fail" | "needs_attention",
          "notes": "explanation of current state"
        }
      ]
    }
  ]
}`,
    `Evaluate AluminatAI's compliance with these frameworks: ${(frameworks as string[]).join(", ")}`
  );

  try {
    const parsed = JSON.parse(raw);
    return NextResponse.json(parsed);
  } catch {
    return NextResponse.json({ frameworks: [] });
  }
}

async function handleGeneratePolicy(body: Record<string, unknown>) {
  const { policyType, companyDetails, productDescription, jurisdiction } = body;
  if (!policyType) {
    return NextResponse.json(
      { error: "policyType is required" },
      { status: 400 }
    );
  }

  const policyNames: Record<string, string> = {
    terms: "Terms of Service",
    privacy: "Privacy Policy",
    acceptable_use: "Acceptable Use Policy",
  };

  const policyName = policyNames[policyType as string] || "Terms of Service";

  const document = await callClaude(
    "You are a legal policy drafter. Generate a complete, professional legal document in markdown format. Include standard sections, effective date (today), and jurisdiction-specific language where applicable. The document should be comprehensive and ready for legal review.",
    `Generate a ${policyName} for AluminatAI.

AluminatAI is a GPU cost intelligence and energy monitoring SaaS platform. We provide a Python agent that runs on users' GPU machines to collect metrics (power draw, temperature, utilization), and a web dashboard for visualization, cost attribution, and benchmarking.

${companyDetails ? `Additional company details: ${companyDetails}` : ""}
${productDescription ? `Product specifics: ${productDescription}` : ""}
${jurisdiction ? `Primary jurisdiction: ${jurisdiction}` : "Primary jurisdiction: United States (Delaware)"}

Key data points to address:
- We collect GPU hardware metrics from user machines via an installed agent
- User accounts with email authentication
- Stripe handles all payment processing
- Data is stored in Supabase (PostgreSQL) with configurable retention
- We offer team/organization features with RBAC
- API keys are used for agent authentication`,
    8192
  );

  return NextResponse.json({
    document,
    policyType,
    generatedAt: new Date().toISOString(),
  });
}

async function handleReviewOutreach(body: Record<string, unknown>) {
  const { emailText, targetRegions } = body;
  if (!emailText) {
    return NextResponse.json(
      { error: "emailText is required" },
      { status: 400 }
    );
  }

  const regions = Array.isArray(targetRegions) && targetRegions.length > 0
    ? (targetRegions as string[])
    : ["US", "EU", "Canada"];

  const regulationMap: Record<string, string> = {
    US: "CAN-SPAM Act",
    EU: "GDPR (email marketing provisions)",
    Canada: "CASL (Canadian Anti-Spam Legislation)",
  };

  const regulations = regions
    .map((r) => regulationMap[r] || r)
    .join(", ");

  const raw = await callClaude(
    `You are an email compliance expert. Analyze outreach emails for regulatory compliance. Be specific about which parts of the email violate or risk violating regulations.

Respond ONLY with valid JSON in this exact format:
{
  "overallCompliant": true | false,
  "summary": "brief overall assessment",
  "issues": [
    {
      "regulation": "which regulation this relates to",
      "issue": "description of the compliance issue",
      "severity": "violation" | "warning" | "suggestion",
      "fix": "specific suggested fix",
      "originalText": "the problematic text from the email (if applicable)"
    }
  ]
}`,
    `Check this outreach email for compliance with: ${regulations}

Email text:
${emailText}`
  );

  try {
    const parsed = JSON.parse(raw);
    return NextResponse.json(parsed);
  } catch {
    return NextResponse.json({
      overallCompliant: false,
      summary: raw,
      issues: [],
    });
  }
}

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { action } = body;

  try {
    switch (action) {
      case "generate_contract":
        return await handleGenerateContract(body);
      case "redline_contract":
        return await handleRedlineContract(body);
      case "fill_questionnaire":
        return await handleFillQuestionnaire(body);
      case "scan_compliance":
        return await handleScanCompliance(body);
      case "generate_policy":
        return await handleGeneratePolicy(body);
      case "review_outreach":
        return await handleReviewOutreach(body);
      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : "Internal error";
    const status = message.includes("not configured") ? 500 : 502;
    return NextResponse.json({ error: message }, { status });
  }
}
