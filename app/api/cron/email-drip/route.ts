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
        subject: "Welcome to NemulAI — GPU cost intelligence starts here",
        body: `Hey there!

Welcome to NemulAI. You've taken the first step toward understanding your GPU energy costs.

Here's what to do next:
1. Install the agent on your GPU machine: pip install nemulai-agent
2. Run it: nemulai-agent run --api-key YOUR_KEY
3. Watch your dashboard light up: https://www.nemulai.com/dashboard

If you need help, reply to this email or check the docs: https://www.nemulai.com/docs/agent

— The NemulAI Team`,
      };
    case 1:
      return {
        subject: "Step 1: Install the NemulAI agent on your GPU",
        body: `Hi again!

Haven't installed the agent yet? It takes under 2 minutes:

  pip install nemulai-agent
  nemulai-agent run --api-key YOUR_KEY

The agent runs as a lightweight daemon, sampling GPU power, utilization, and temperature every 5 seconds. No performance impact — it reads NVML counters only.

Supports: bare metal, Docker, Kubernetes DaemonSet, SLURM prologue/epilogue.

Docs: https://www.nemulai.com/docs/agent

— The NemulAI Team`,
      };
    case 2:
      return {
        subject: "Your first GPU cost data is in — check your dashboard",
        body: `Hi!

By now your agent should be reporting metrics. Head to your dashboard to see:

- Real-time cost per GPU
- Energy breakdown (kWh, CO2e)
- Cloud cost comparison — see how much you're saving vs AWS/GCP on-demand

Dashboard: https://www.nemulai.com/dashboard

Not seeing data? Make sure the agent is running: nemulai-agent service status

— The NemulAI Team`,
      };
    case 3:
      return {
        subject: "Pro tip: Set a GPU budget to avoid cost surprises",
        body: `Hi!

Now that you have cost data flowing, set a budget alert to stay on top of spending:

1. Go to Dashboard → Settings
2. Create a budget (daily, weekly, or monthly)
3. Choose alert channels: email, Slack, PagerDuty, or OpsGenie

You'll get warned when spend hits your threshold — before it becomes a problem.

Dashboard: https://www.nemulai.com/dashboard/settings

— The NemulAI Team`,
      };
    case 4:
      return {
        subject: "Invite your team — GPU cost visibility for everyone",
        body: `Hi!

NemulAI works best when the whole team can see GPU costs. Invite teammates to:

- View cost breakdowns by team, model, or cluster
- Get their own budget alerts
- Track their workloads' energy efficiency

Invite your team: https://www.nemulai.com/dashboard/settings

Running 100+ GPUs? Check out our Enterprise plan with SSO, SLA dashboard, and dedicated support: https://www.nemulai.com/enterprise

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
