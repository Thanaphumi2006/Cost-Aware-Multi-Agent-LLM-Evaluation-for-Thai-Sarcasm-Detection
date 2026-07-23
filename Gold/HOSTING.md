# Hosting the demo publicly (safely)

The developer app (`app.py`, served at `/`) must **never** face the internet: it has a key-input box,
a corrections writer, and batch endpoints, all sharing one API key a stranger could spend or wipe.
For public traffic use **`serve_public.py`** instead. It mounts only:

| Route | Purpose |
|---|---|
| `GET /app` (and `/`) | the demo page (`app.html`) |
| `POST /api/fetch_comments` | fetch Thai comments from a YouTube/Pantip/Reddit link (host-allowlisted) |
| `POST /api/escalate` | the paid tier of the cascade (WangchanBERTa → gpt-4.1-mini) |
| `GET /healthz` | liveness probe |

Everything else returns 404. The API key is read from the environment **only** — there is no route
that sets it from a request.

## What it already does for you

- **Key never leaves the server.** No `/api/key`; the browser can't submit or read it.
- **SSRF-proof fetching.** `fetch_social.platform_of` parses the real hostname and allowlists
  `youtube.com` / `youtu.be` / `pantip.com` / `reddit.com` / `redd.it` (exact or subdomain). Anything
  else, and any URL resolving to a private/loopback/link-local IP, is refused.
- **Abuse limits.** Per-IP hourly + global daily caps on the paid tier, a per-request text cap, and a
  64 KB body cap. Tune via env (below).
- **Hardening headers** on every response: `Content-Security-Policy` (self + Google Fonts only),
  `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
- **Safe by default.** Binds `127.0.0.1` unless you say otherwise; warns if you bind `0.0.0.0`
  without `TRUST_PROXY`.

## Recommended deployment

Run it under a real WSGI server, on loopback, behind an HTTPS reverse proxy.

```bash
pip install -r requirements.txt gunicorn          # or: waitress (Windows)
export OPENAI_API_KEY=sk-...                       # the ONLY place the key lives
export TRUST_PROXY=1                               # you are behind nginx/Caddy (see note)
python Gold/train_final_wcb.py                     # once, if you want the WangchanBERTa tier

gunicorn -w 2 -b 127.0.0.1:8000 --chdir Gold serve_public:app
# Windows:  waitress-serve --listen=127.0.0.1:8000 --call serve_public:app   (run from Gold/)
```

Then terminate TLS and forward in your proxy. nginx example:

```nginx
server {
    listen 443 ssl;
    server_name sarcasm.example.com;
    # ssl_certificate ... ;  (use certbot / your CA)

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;   # append, don't overwrite
        proxy_set_header Host $host;
    }
}
```

**Why `$proxy_add_x_forwarded_for` + `TRUST_PROXY=1`:** that directive *appends* the real client IP
that nginx saw, so the rightmost `X-Forwarded-For` entry is trustworthy and a client cannot spoof its
way past the per-IP limit. `serve_public.py` reads exactly that entry. If you have two proxies, adjust.
Without `TRUST_PROXY`, every request looks like it came from the proxy and shares one bucket.

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | (none) | the server's key. Missing → escalate stays cue/WCB-only, never errors. |
| `TRUST_PROXY` | off | trust the rightmost `X-Forwarded-For` for the client IP (set only behind your own proxy). |
| `PUBLIC_IP_HOURLY_LIMIT` | 60 | paid-tier calls per IP per hour. |
| `PUBLIC_DAILY_LIMIT` | 2000 | paid-tier calls per day, everyone combined (your cost ceiling). |
| `PUBLIC_MAX_TEXT` | 2000 | max characters per escalate request. |
| `PUBLIC_FETCH_LIMIT` | 80 | max comments fetched per link. |
| `HOST` / `PORT` | 127.0.0.1 / 8000 | bind address for a direct run (prefer gunicorn). |

## Cost and safety notes

- **The daily cap is your spend ceiling.** At `PUBLIC_DAILY_LIMIT=2000`, the paid tier fires at most
  2000× gpt-4.1-mini/day. Set it to whatever you're willing to pay; the cue + WangchanBERTa tiers are
  free and answer most traffic, so real escalation volume is a fraction of page views.
- **Rate-limit state is per process and in-memory.** With `-w 2` each worker keeps its own counters, so
  effective caps are ~N× the numbers above and reset on restart. For hard, shared caps put a limiter in
  the proxy (nginx `limit_req`) or a shared store; for a hobby deployment the in-process caps are enough.
- **The model can be wrong** (real-world F1 ≈ 0.46, finding 20). Present results as a hint, not a verdict.
- **HTTPS is required** in practice — without it the comments people paste and the results travel in
  clear text. Terminate TLS at the proxy.
- Keep `app.py` (the dev page) bound to `127.0.0.1`, or don't run it on the public box at all.
