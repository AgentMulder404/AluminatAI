import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(_req: NextRequest) {
  const res = NextResponse.next();

  res.headers.set("X-Content-Type-Options", "nosniff");
  res.headers.set("X-Frame-Options", "DENY");
  res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  res.headers.set(
    "Strict-Transport-Security",
    "max-age=31536000; includeSubDomains"
  );
  res.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=()"
  );
  res.headers.set(
    "Content-Security-Policy-Report-Only",
    [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://app.cal.com",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: https:",
      "font-src 'self'",
      "connect-src 'self' https://*.supabase.co wss://*.supabase.co",
      "frame-src https://app.cal.com",
    ].join("; ")
  );

  return res;
}

export const config = {
  matcher: [
    // Apply to all routes except static assets
    "/((?!_next/static|_next/image|favicon\\.ico).*)",
  ],
};
