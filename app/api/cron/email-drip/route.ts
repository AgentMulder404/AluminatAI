// GET /api/cron/email-drip
// Schedule: 0 10 * * * (daily at 10 AM UTC)
// Sends onboarding drip emails to new users at day 0, 1, 3, 7, 14.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { sendEmail } from "@/lib/notifications";
import { verifyCronSecret } from "@/lib/auth-helpers";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

// Map drip index (0-4) to minimum days since signup
const DRIP_DAY_MAP: Record<number, number> = {
  0: 0,  // Welcome — same day
  1: 1,  // Install agent
  2: 3,  // First dashboard
  3: 7,  // Set a budget
  4: 14, // Invite your team
};

interface DripEmail {
  subject: string;
  body: string;
}

function getDripEmail(index: number, userEmail: string): DripEmail {
  switch (index) {
    case 0:
      return {
        subject: "You just saved your first dollar on GPU compute",
        body: `Hey there!

Welcome to NemulAI. You've taken the first step toward cutting your GPU bill.

Here's what to do next:
1. Install the agent: pip install nemulai
2. Set your key: export ALUMINATAI_API_KEY=YOUR_KEY
3. Start monitoring: nemulai

Your dashboard will show you exactly where your money is going: https://www.nemulai.com/dashboard

Takes under 2 minutes. No performance impact — the agent reads hardware counters only.

— The NemulAI Team`,
      };
    case 1:
      return {
        subject: "Your first GPU cost report is ready",
        body: `Hi!

Head to your dashboard to see what your GPUs are actually costing you:

- Real-time cost per GPU and per job
- Waste detection — idle GPUs flagged automatically
- Cloud cost comparison vs AWS/GCP on-demand pricing

Dashboard: https://www.nemulai.com/dashboard

Not seeing data? Make sure the agent is running: nemulai service status

— The NemulAI Team`,
      };
    case 2:
      return {
        subject: "Here's what you're wasting on idle GPUs",
        body: `Hi!

NemulAI has been watching your GPUs. Here's the thing: idle GPUs still draw 60-80% of peak power. That's cash burning while nothing useful happens.

Check your waste alerts to see which machines are costing you money for nothing: https://www.nemulai.com/dashboard

The agent is already learning your workload patterns to give you smarter optimization recommendations over time.

— The NemulAI Team`,
      };
    case 3:
      return {
        subject: "Your GPU fleet is learning your patterns",
        body: `Hi!

NemulAI's self-learning agent has been studying your workload patterns for a week now. It's getting smarter about when your GPUs are productive vs idle.

Check your optimization recommendations — they get more accurate the longer the agent runs: https://www.nemulai.com/dashboard

Set a budget alert to stay ahead of cost surprises: Dashboard → Settings → Budgets.

— The NemulAI Team`,
      };
    case 4:
      return {
        subject: "Your team should see these GPU cost savings",
        body: `Hi!

You've been running NemulAI for two weeks. Time to bring the team in.

With team chargeback, every team sees their own GPU spend:
- Per-team cost attribution with one env var
- Individual budget alerts
- Shared dashboard access

Invite your team: https://www.nemulai.com/dashboard/settings

Running 100+ GPUs? Our Enterprise plan includes fleet-wide optimization, SSO, and dedicated support at $15/GPU/mo: https://www.nemulai.com/enterprise

— The NemulAI Team`,
      };
    default:
      return { subject: "", body: "" };
  }
}

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Fetch users who haven't completed the drip sequence
  const { data: users, error } = await supabase
    .from("users")
    .select("id, email, created_at, onboarding_drip_sent")
    .lt("onboarding_drip_sent", 5)
    .limit(500);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  let usersProcessed = 0;
  let emailsSent = 0;

  const now = Date.now();

  for (const user of users ?? []) {
    const createdAt = new Date(user.created_at).getTime();
    const daysSinceSignup = Math.floor((now - createdAt) / 86_400_000);
    const dripIndex = user.onboarding_drip_sent as number;
    const requiredDay = DRIP_DAY_MAP[dripIndex];

    if (requiredDay === undefined || daysSinceSignup < requiredDay) continue;

    usersProcessed++;

    const email = getDripEmail(dripIndex, user.email);
    if (!email.subject) continue;

    const sent = await sendEmail(user.email, email.subject, email.body);

    if (sent) {
      // Increment drip counter — optimistic lock via the current value
      const { error: updateErr } = await supabase
        .from("users")
        .update({ onboarding_drip_sent: dripIndex + 1 })
        .eq("id", user.id)
        .eq("onboarding_drip_sent", dripIndex); // Optimistic lock

      if (!updateErr) emailsSent++;
    }
  }

  return NextResponse.json({ users_processed: usersProcessed, emails_sent: emailsSent });
}
