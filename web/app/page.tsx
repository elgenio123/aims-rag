"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

type TraceItem = { doc_id: string | null; source_url: string | null; score: number | null };

type Exchange = {
  id: number;
  question: string;
  status: "loading" | "done" | "error";
  answer?: string;
  trace?: TraceItem[];
  error?: string;
};

const NOT_AVAILABLE = "the information is not available";
const MAX_LEN = 500;
const EXAMPLES = [
  { title: "Admissions", text: "What are the admission requirements for the AIMS master's program?" },
  { title: "Scholarships", text: "Which scholarships are available for AIMS Cameroon students?" },
  { title: "Academics", text: "What courses are taught in the structured master's program?" },
  { title: "Campus", text: "Where is AIMS Cameroon located?" },
];

function isSafeUrl(url: string | null): url is string {
  return !!url && /^https?:\/\//i.test(url);
}

function uniqueSources(trace: TraceItem[]): TraceItem[] {
  const seen = new Set<string>();
  return trace.filter((t) => {
    let key = t.doc_id ?? "";
    if (isSafeUrl(t.source_url)) {
      const { host, path } = prettyUrl(t.source_url);
      key = `${host}${path}`;
    }
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function prettyUrl(url: string): { host: string; path: string } {
  try {
    const u = new URL(url);
    const path = decodeURIComponent(u.pathname).replace(/\/$/, "") || "/";
    return { host: u.hostname.replace(/^www\./, ""), path };
  } catch {
    return { host: url, path: "" };
  }
}

const markdownComponents: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="md-table">
      <table>{children}</table>
    </div>
  ),
};

function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

function Sources({ trace }: { trace: TraceItem[] }) {
  const [open, setOpen] = useState(false);
  const items = uniqueSources(trace);
  return (
    <div className={`sources${open ? " open" : ""}`}>
      <button type="button" className="sources-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M9 6l6 6-6 6" />
        </svg>
        {items.length} source{items.length === 1 ? "" : "s"}
      </button>
      <div className="sources-body">
        <ul>
          {items.map((t, i) => {
            const pct = typeof t.score === "number" ? Math.max(0, Math.min(1, t.score)) * 100 : 0;
            const label = isSafeUrl(t.source_url) ? prettyUrl(t.source_url) : { host: t.doc_id ?? "document", path: "" };
            const inner = (
              <>
                <span className="src-index">{i + 1}</span>
                <span className="src-text">
                  <span className="src-host">{label.host}</span>
                  <span className="src-path">{label.path}</span>
                </span>
                {typeof t.score === "number" && (
                  <span className="src-score" title={`Similarity ${t.score.toFixed(3)}`}>
                    <span className="bar">
                      <span style={{ width: open ? `${pct}%` : 0 }} />
                    </span>
                    {t.score.toFixed(2)}
                  </span>
                )}
              </>
            );
            return (
              <li key={i} style={{ ["--i" as string]: i }}>
                {isSafeUrl(t.source_url) ? (
                  <a href={t.source_url} target="_blank" rel="noopener noreferrer">
                    {inner}
                  </a>
                ) : (
                  <div className="src-static">{inner}</div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

export default function Home() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState("");
  const nextId = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const busy = exchanges.some((e) => e.status === "loading");
  const started = exchanges.length > 0;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [exchanges]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [question]);

  async function ask(q: string) {
    const text = q.trim();
    if (!text || busy) return;
    const id = nextId.current++;
    setExchanges((prev) => [...prev, { id, question: text, status: "loading" }]);
    setQuestion("");

    const update = (patch: Partial<Exchange>) =>
      setExchanges((prev) => prev.map((e) => (e.id === id ? { ...e, ...patch } : e)));

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await res.json();
      if (!res.ok) {
        update({ status: "error", error: typeof data?.detail === "string" ? data.detail : "Something went wrong." });
        return;
      }
      update({ status: "done", answer: data.answer, trace: data.trace ?? [] });
    } catch {
      update({ status: "error", error: "Network error. Check your connection and try again." });
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    ask(question);
  }

  function reset() {
    if (busy) return;
    setExchanges([]);
    inputRef.current?.focus();
  }

  return (
    <div className={`app${started ? " started" : ""}`}>
      <div className="backdrop" aria-hidden="true">
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
        <span className="grid" />
      </div>

      <header className="topbar">
        <button type="button" className="brand" onClick={reset} aria-label="Start a new conversation">
          <span className="logo" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 3l9 5-9 5-9-5 9-5z" />
              <path d="M7 10.5v4.5c0 1.5 2.2 3 5 3s5-1.5 5-3v-4.5" />
            </svg>
          </span>
          <span className="brand-name">AIMS Cameroon Assistant</span>
        </button>
        {started && (
          <button type="button" className="ghost" onClick={reset} disabled={busy}>
            New chat
          </button>
        )}
      </header>

      <main className="main">
        {!started ? (
          <section className="hero">
            <span className="pill">
              <span className="dot" /> Grounded in official AIMS documents
            </span>
            <h1>
              Ask anything about <span className="gradient-text">AIMS Cameroon</span>
            </h1>
            <p className="lede">
              Admissions, programs, scholarships and campus life — every answer cites the pages it came from.
            </p>
            <div className="examples">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={ex.text}
                  type="button"
                  className="example"
                  style={{ ["--i" as string]: i }}
                  onClick={() => ask(ex.text)}
                >
                  <span className="example-title">{ex.title}</span>
                  <span className="example-text">{ex.text}</span>
                  <svg className="example-arrow" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M5 12h14M13 6l6 6-6 6" />
                  </svg>
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="thread" aria-live="polite">
            {exchanges.map((ex) => {
              const unavailable = ex.answer?.trim().toLowerCase().startsWith(NOT_AVAILABLE);
              return (
                <article key={ex.id} className="exchange">
                  <div className="bubble question">{ex.question}</div>
                  {ex.status === "loading" && (
                    <div className="bubble answer loading" aria-label="Searching documents">
                      <span className="typing">
                        <i />
                        <i />
                        <i />
                      </span>
                      <span className="loading-label">Searching documents</span>
                      <span className="skeleton" />
                      <span className="skeleton short" />
                    </div>
                  )}
                  {ex.status === "error" && (
                    <div className="bubble answer error" role="alert">
                      <strong>Couldn&apos;t get an answer.</strong> {ex.error}
                      <button type="button" className="retry" onClick={() => ask(ex.question)} disabled={busy}>
                        Try again
                      </button>
                    </div>
                  )}
                  {ex.status === "done" && (
                    <div className={`bubble answer${unavailable ? " unavailable" : ""}`}>
                      {unavailable && <span className="badge">Not in the documents</span>}
                      <Markdown text={ex.answer ?? ""} />
                      {!unavailable && ex.trace && ex.trace.length > 0 && <Sources trace={ex.trace} />}
                    </div>
                  )}
                </article>
              );
            })}
            <div ref={bottomRef} className="anchor" />
          </section>
        )}
      </main>

      <footer className="dock">
        <form className={`composer${busy ? " busy" : ""}`} onSubmit={onSubmit}>
          <textarea
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value.slice(0, MAX_LEN))}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(question);
              }
            }}
            placeholder="Ask about admissions, programs, scholarships…"
            rows={1}
            aria-label="Your question"
          />
          <span className={`count${question.length > MAX_LEN * 0.9 ? " warn" : ""}`}>
            {question.length}/{MAX_LEN}
          </span>
          <button type="submit" className="send" disabled={busy || !question.trim()} aria-label="Send question">
            {busy ? (
              <span className="spinner" />
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            )}
          </button>
        </form>
        <p className="hint">Answers are generated from scraped AIMS Cameroon pages and may be incomplete.</p>
      </footer>
    </div>
  );
}
