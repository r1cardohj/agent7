---
name: websearch
description: How to query the search tool well — operators, result triage, and when to read the page instead.
---

# Web search

The `search` tool calls DeepSeek's Search API and returns ranked pages with a
long `summary`. Read this once per task, before the first search, and search
according to it.

## When to search

- Facts that can have changed since training: releases, versions, APIs, prices,
  changelogs, CVEs, "is X still maintained", "which library should I use in 2026".
- Error messages and stack traces you cannot explain from the local code.
- Anything the user frames as current ("latest", "now", "this week", a date).

Do **not** search for things the repo already answers: file contents, configs,
dependency lists, git history. `read` / `bash` are cheaper and authoritative.
A wrong local answer beats a plausible web answer — check the workspace first.

## Writing the query

The tool passes search syntax straight through. `query` must be 1-128 chars.

| Operator | Example | Effect |
|---|---|---|
| `site:` | `rfc site:github.com OR site:datatracker.ietf.org` | restrict to sites |
| `-site:` | `grpc -site:medium.com` | exclude sites |
| `"..."` | `"context deadline exceeded"` | exact phrase |
| `-` | `python asyncio -tutorial` | exclude a word |
| `-"..."` | `nginx -"release notes"` | exclude a phrase |
| `lang:` | `lang:zh` / `lang:zh,en` | restrict language |
| `after:` / `before:` | `after:2025-01-01` | restrict publish date |

Rules of thumb:

- Put the exact error string or API name in quotes; drop filler words.
- Add `after:<year>` when recency matters, and `site:` when you know the
  authority (official docs, GitHub, an RFC, a vendor changelog).
- Search once per question, not once per sentence. Two or three queries with
  different angles beat eight near-duplicates.
- `count` defaults to 8; it does not change the price, so do not shrink it.

## Reading the results

Each result line carries `score` (relevance, 0-3), `authority` (0-4, `-1`
unknown), a date, and the site name. The body is a summary, not the page.

- Trust `score >= 1.5`; below that the tool already hides results, and a
  "showing the top N" note means nothing matched well — rephrase instead of
  trusting the leftovers.
- Prefer `authority >= 3` for factual claims; a blog post and an official doc
  that disagree resolve in the doc's favour.
- Check dates against the question. An undated page about a fast-moving API is
  a liability; an `after:` operator usually fixes the query.
- When two results contradict each other, say so in the answer instead of
  silently picking one.

## From results to the page

`search` returns summaries. Call `fetch` on a URL when you need the exact
wording of an API signature, a config key, a table, or a code sample — that is
cheaper and more accurate than searching again.

Fetch one or two URLs, not all of them: 5-8 pages of full text will crowd out
the code you are actually editing. If `fetch` reports no readable text (JS-only
pages, paywalls, 403), rely on the summary or find another source.

## Failures

- `no results` — retry once with fewer operators and shorter phrasing; if the
  second try is empty, answer from what you have and say the search was empty.
- `402` — the account has no balance. Stop searching and tell the user; do not
  retry.
- `429` or `5xx` — the tool already retried. Wait, do the local work first, then
  search again. Search is limited to 100 requests/s per account and `fetch` to
  25/s, and per-domain fetch limits are tighter still.

Search and fetch are billed per successful request (only when results come
back), so a precise query is the cheap one.
