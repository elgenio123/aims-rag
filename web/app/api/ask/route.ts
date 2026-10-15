import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL ?? "http://localhost:8000";
const LIMIT = Number(process.env.RATE_LIMIT_PER_MINUTE ?? "10");
const WINDOW_MS = 60_000;

// In-memory, per-container limiter: enough to stop casual abuse draining OpenRouter credits.
const hits = new Map<string, number[]>();

function clientIp(req: NextRequest): string {
  // The load balancer appends the real client IP; earlier entries can be forged by the client.
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",").pop()!.trim();
  return req.headers.get("x-real-ip") ?? "unknown";
}

function rateLimited(ip: string): boolean {
  const now = Date.now();
  const recent = (hits.get(ip) ?? []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= LIMIT) {
    hits.set(ip, recent);
    return true;
  }
  recent.push(now);
  hits.set(ip, recent);
  if (hits.size > 10_000) {
    for (const [key, times] of hits) {
      if (times.every((t) => now - t >= WINDOW_MS)) hits.delete(key);
    }
  }
  return false;
}

export async function POST(req: NextRequest) {
  if (rateLimited(clientIp(req))) {
    return NextResponse.json(
      { detail: "Too many questions. Please wait a minute and try again." },
      { status: 429 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }
  const question =
    typeof (body as { question?: unknown })?.question === "string"
      ? (body as { question: string }).question.trim()
      : "";
  if (!question || question.length > 500) {
    return NextResponse.json(
      { detail: "Question must be between 1 and 500 characters." },
      { status: 400 },
    );
  }

  try {
    const res = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(90_000),
    });
    const data = await res.json().catch(() => ({ detail: "Unexpected response from the API." }));
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: "The assistant is unreachable right now. Please try again shortly." },
      { status: 503 },
    );
  }
}
