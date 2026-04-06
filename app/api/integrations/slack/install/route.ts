// GET  /api/integrations/slack/install       → redirect to Slack OAuth
// GET  /api/integrations/slack/install?code=  → OAuth callback, store tokens
// DELETE /api/integrations/slack/install      → revoke installation

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requirePlan } from "@/lib/plans";

export const runtime = "edge";

const SLACK_CLIENT_ID = process.env.SLACK_CLIENT_ID ?? "";
const SLACK_CLIENT_SECRET = process.env.SLACK_CLIENT_SECRET ?? "";
const SLACK_REDIRECT_URI = process.env.SLACK_REDIRECT_URI ?? "";
const SLACK_SCOPES = "chat:write,commands,channels:read";

// CSRF state token helpers — HMAC-SHA256 signed, 10-minute expiry
const STATE_SECRET = process.env.SLACK_CLIENT_SECRET ?? "fallback-oauth-state-key";

async function signState(userId: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(STATE_SECRET), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const nonce = crypto.getRandomValues(new Uint8Array(16));
  const nonceHex = Array.from(nonce).map(b => b.toString(16).padStart(2, "0")).join("");
  const exp = Date.now() + 10 * 60 * 1000; // 10 min
  const payload = `${userId}.${nonceHex}.${exp}`;
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", key, enc.encode(payload)));
  const sigHex = Array.from(sig).map(b => b.toString(16).padStart(2, "0")).join("");
  return `${payload}.${sigHex}`;
}

async function verifyState(state: string): Promise<string | null> {
  const parts = state.split(".");
  if (parts.length !== 4) return null;
  const [userId, , expStr, sigHex] = parts;
  const exp = Number(expStr);
  if (isNaN(exp) || Date.now() > exp) return null; // expired

  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(STATE_SECRET), { name: "HMAC", hash: "SHA-256" }, false, ["verify"]
  );
  const payload = parts.slice(0, 3).join(".");
  const sigBytes = new Uint8Array(sigHex.match(/.{2}/g)!.map(b => parseInt(b, 16)));
  const valid = await crypto.subtle.verify("HMAC", key, sigBytes, enc.encode(payload));
  return valid ? userId : null;
}

async function authenticate() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) return null;
  return user;
}

export async function GET(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Plan gate: Slack requires Pro+
  const planCheck = await requirePlan(user.id, "slack_integration");
  if (!planCheck.allowed) {
    return NextResponse.json(
      { error: planCheck.reason, upgrade_to: planCheck.upgrade_to },
      { status: 403 }
    );
  }

  const code = req.nextUrl.searchParams.get("code");

  // Step 1: If no code, redirect to Slack OAuth authorization page
  if (!code) {
    if (!SLACK_CLIENT_ID) {
      return NextResponse.json(
        { error: "Slack integration not configured (missing SLACK_CLIENT_ID)" },
        { status: 503 }
      );
    }

    const authorizeUrl = new URL("https://slack.com/oauth/v2/authorize");
    authorizeUrl.searchParams.set("client_id", SLACK_CLIENT_ID);
    authorizeUrl.searchParams.set("scope", SLACK_SCOPES);
    authorizeUrl.searchParams.set("redirect_uri", SLACK_REDIRECT_URI);
    authorizeUrl.searchParams.set("state", await signState(user.id));

    return NextResponse.redirect(authorizeUrl.toString());
  }

  // Verify CSRF state token
  const state = req.nextUrl.searchParams.get("state");
  if (!state) {
    return NextResponse.json({ error: "Missing OAuth state" }, { status: 400 });
  }
  const stateUserId = await verifyState(state);
  if (!stateUserId || stateUserId !== user.id) {
    return NextResponse.json({ error: "Invalid or expired OAuth state" }, { status: 403 });
  }

  // Step 2: Exchange code for tokens
  if (!SLACK_CLIENT_ID || !SLACK_CLIENT_SECRET) {
    return NextResponse.json(
      { error: "Slack integration not configured" },
      { status: 503 }
    );
  }

  const tokenRes = await fetch("https://slack.com/api/oauth.v2.access", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: SLACK_CLIENT_ID,
      client_secret: SLACK_CLIENT_SECRET,
      code,
      redirect_uri: SLACK_REDIRECT_URI,
    }),
  });

  const tokenData = await tokenRes.json();

  if (!tokenData.ok) {
    return NextResponse.json(
      { error: `Slack OAuth failed: ${tokenData.error}` },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  const { error: upsertErr } = await supabase
    .from("slack_installations")
    .upsert(
      {
        user_id: user.id,
        slack_team_id: tokenData.team?.id ?? "",
        slack_team_name: tokenData.team?.name ?? "",
        bot_token: tokenData.access_token,
        bot_user_id: tokenData.bot_user_id ?? "",
        scope: tokenData.scope ?? "",
        installed_at: new Date().toISOString(),
        is_active: true,
      },
      { onConflict: "user_id,slack_team_id" }
    );

  if (upsertErr) {
    return NextResponse.json({ error: upsertErr.message }, { status: 500 });
  }

  // Redirect back to dashboard settings
  return NextResponse.redirect(
    new URL("/dashboard/settings?slack=connected", req.url)
  );
}

// DELETE — revoke a Slack installation
export async function DELETE(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { slack_team_id } = await req.json();

  if (!slack_team_id) {
    return NextResponse.json(
      { error: "slack_team_id is required" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  // Revoke the token with Slack
  const { data: install } = await supabase
    .from("slack_installations")
    .select("bot_token")
    .eq("user_id", user.id)
    .eq("slack_team_id", slack_team_id)
    .maybeSingle();

  if (install?.bot_token) {
    await fetch("https://slack.com/api/auth.revoke", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${install.bot_token}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
    }).catch(() => {}); // best-effort revocation
  }

  const { error } = await supabase
    .from("slack_installations")
    .delete()
    .eq("user_id", user.id)
    .eq("slack_team_id", slack_team_id);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ revoked: true });
}
