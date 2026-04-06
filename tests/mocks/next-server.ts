/**
 * Mock for next/server — provides NextRequest and NextResponse
 * that work in a standard Node/edge-runtime test environment.
 */

export class NextRequest extends Request {
  nextUrl: URL;

  constructor(input: RequestInfo | URL, init?: RequestInit) {
    super(input, init);
    this.nextUrl = new URL(
      typeof input === "string" ? input : input instanceof URL ? input.href : input.url
    );
  }
}

export class NextResponse extends Response {
  static json(body: unknown, init?: ResponseInit): NextResponse {
    const headers = new Headers(init?.headers);
    headers.set("content-type", "application/json");
    return new NextResponse(JSON.stringify(body), {
      ...init,
      headers,
    }) as NextResponse;
  }

  static redirect(url: string | URL, status = 307): NextResponse {
    return new NextResponse(null, {
      status,
      headers: { Location: typeof url === "string" ? url : url.href },
    }) as NextResponse;
  }

  static next(): NextResponse {
    return new NextResponse(null, { status: 200 }) as NextResponse;
  }
}
