# Runbook — generating an article

How to change the topic and run the pipeline end to end.

> The pipeline is **five scripts run in order**, each picking up the previous one's
> state from Postgres. There is no API yet, so this is the way to run it.

---

## 1. Where to change the topic

**One place: [`test_m2_full.py`](test_m2_full.py), line 37.**

```python
request = ResearchRequest(
    topic="Eligibility criteria under Section 29A of the Insolvency and Bankruptcy Code, 2016",
    practice_area="insolvency",
    article_config=ArticleConfig(article_type="explainer", target_words=2000),
    budget_inr=Decimal("50.00"),
)
```

Change `topic` — that's the only required edit. Update `practice_area` too so the
planner has the right context.

### Writing a good topic

The topic drives everything downstream: sub-queries, source ranking, and which
passages of each judgment get read. Be specific.

| | Example |
|---|---|
| ✅ Good | `"Doctrine of legitimate expectation in Indian administrative law"` |
| ✅ Good | `"Section 138 Negotiable Instruments Act: liability of company directors"` |
| ❌ Too broad | `"Company law"` |
| ❌ Not a legal topic | `"How to start a business"` |

**Constraint: Indian law only.** The planner prompt, source-quality table, citation
parser, and the case-law provider are all India-specific. A US or UK topic will run
without erroring but produce a poor article — foreign sources rank as low-authority
and their citations won't parse.

### Optional settings (same block)

| Field | Options | Notes |
|---|---|---|
| `article_type` | `blog`, `explainer`, `client_alert`, `white_paper`, `journal_note` | |
| `target_words` | 400–12000 | ~2000 is a good default |
| `audience` | `practitioner`, `student`, `general`, `policy` | defaults to `practitioner` |
| `budget_inr` | Decimal | run ceiling; ₹50 suits a normal run |
| `max_research_loops` | 0–3 | how many repair loops are allowed |

---

## 2. Before you start

Start the database (Postgres on port **5433**):

```bash
docker compose up -d
```

Check that `.env` has `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`,
`SERPAPI_API_KEY`, and `INDIANKANOON_API_TOKEN` set.

---

## 3. Run the pipeline

Run these **in order**. Each takes minutes and makes real, paid API calls.

### Step 1 — Plan, search, evaluate

```bash
PYTHONPATH=. python test_m2_full.py
```

Turns your topic into legal issues and sub-queries, searches four providers, and
selects the best sources.

**Prints a run ID — copy it.** Later steps default to the most recent run, so you
only need it if you want to target a specific one.

### Step 2 — Fetch and extract evidence

```bash
PYTHONPATH=. python test_m3_full.py
```

Downloads each source and pulls out evidence with verbatim quotes, then verifies
every quote against the original text.

Some fetch failures are normal (a few sites block bots or have broken certificates).
Judgments now come through the IndianKanoon API, so those should succeed.

### Step 3 — Check for gaps

```bash
PYTHONPATH=. python test_gap_check_full.py
```

Reports coverage per legal issue and a verdict:

- **`complete`** → skip to Step 5
- **`needs_more_research`** → run Step 4 first
- **`exhausted`** → gaps remain but the loop budget is spent; continue to Step 5

### Step 4 — Repair loop *(only if Step 3 said `needs_more_research`)*

```bash
PYTHONPATH=. python test_repair_loop_full.py
```

Re-plans and re-searches to fill the gaps, then re-checks. You can run this at most
`max_research_loops` times.

### Step 5 — Generate the article

```bash
PYTHONPATH=. python test_generation_full.py
```

Thesis → outline → per-section drafting → verification → voice pass → assembly.

Writes **`article_draft_<first-8-chars-of-run-id>.md`** to the project root.

### Targeting a specific run

Every step after Step 1 defaults to the most recent run. To use an older one, pass
its ID:

```bash
PYTHONPATH=. python test_generation_full.py a88abe54-a6dd-475f-b914-6a4dfbf83328
```

---

## 4. Reading the result

The final step prints a verification verdict.

| Verdict | Meaning |
|---|---|
| **`passed`** | Every citation resolves, every quote matches its source, all arithmetic checks out |
| **`failed`** | **The safety net working, not necessarily a bug.** Read the reported issues — a hallucinated citation or altered quote is exactly what it exists to catch |

Other things you'll see in the output:

- **`unresolved_citation_ratio`** — share of citations that print as
  `[citation unresolved]`. Fails the run above 25%.
- **`voice_pass_rejected_unsafe_edit`** — the polish step tried to change a citation,
  so its rewrite was discarded and the original kept. Harmless.
- **`calc_count=0`** — no derived figures. Normal for doctrinal topics.

---

## 5. Costs and timing

Roughly **$0.90–1.20 per article**, about 10–15 minutes across all steps.

| Step | Cost |
|---|---|
| 1 — plan/search/evaluate | ~$0.05 |
| 2 — fetch/extract | ~$0.43 (measured) |
| 3 — gap check | ~$0.02 |
| 4 — repair loop *(if needed)* | ~$0.35 |
| 5 — generation | ~$0.55 |

Research runs on GPT-4o; article generation runs on Claude Sonnet 5. Controlled by
`LLM_PROVIDER` and `GENERATION_LLM_PROVIDER` in `.env`.

Judgment-heavy runs are slower by design — court sites are limited to 1 request per
second to avoid the rate-limiting that used to drop judgments entirely.

---

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| `Connect call failed ... 5433` | Postgres isn't running — `docker compose up -d` |
| Step 2 fetches almost nothing | Check network; a few 403s are normal, near-total failure isn't |
| Step 3 reports blocking gaps | Run Step 4, or accept and continue — an `exhausted` verdict still generates |
| Generation fails verification | Re-run Step 5; drafting is non-deterministic and often passes on retry |
| Want to re-generate without re-researching | Just re-run Step 5 — it reuses stored evidence |

**Note:** the output filename is keyed to the run ID, so re-running Step 5 **overwrites**
the previous article. Copy it first if you want to compare.

### Checking state

```bash
PYTHONPATH=. python check_sources.py
```

Lists runs and their source counts.

---

## 7. Tests

After any code change:

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

```bash
PYTHONPATH=. python test_generation_synthetic.py
```

68 and 45 checks respectively, all mocked — no API calls, no cost.

The second one is authoritative: its helper counts failures rather than asserting,
so pytest has previously reported "passed" while checks were genuinely failing.
