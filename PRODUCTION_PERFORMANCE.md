# Production page latency

Slow providers and lineup generation previously held HTML requests open. In
addition, visiting `/model` triggered settlement, training, and live slate loading.
The old `_bounded` helper timed out its future but then waited for the worker
when leaving the executor context, defeating its timeout.

## Behavior

- Expensive public GET pages share two background rendering workers and a bounded
  queue. Repeated visits to the same validated parameter combination share work.
  A cold page renders the navigation immediately and checks for data every five
  seconds. Cached HTML stays fresh for two minutes. Older pages are explicitly
  labeled with their saved UTC time and a warning about outdated prices.
- The last successful page is available for up to one hour during refresh errors.
  After that, the loading/retry page is shown. A provider outage can still delay
  data availability; this does not fabricate prices, results, or lineups.
- Up to 24 disposable HTML responses are persisted under
  `$SCOREEDGE_DATA_DIR/.data_cache/page_responses` (12 MiB memory budget and
  2 MiB per page). Only files in that cache namespace are pruned.
- The homepage, NFL, NBA, DFS, and hunt pages are queued after startup and every
  four minutes. Existing provider caches and background learning continue.
- `/model` reads saved reports and cached slate predictions. Training and
  settlement continue through the existing background learning tick.
- `/calibration`, `/backtest`, and `/staking` render directly from saved
  outcomes, schedules, player stats, and prices. They do not wait for provider
  rendering workers, initialize feeds, start learning, or auto-settle unsaved
  picks. Missing scores remain pending. Cached-only reads do not insert empty
  entries into the learning feed caches or prevent subsequent live refreshes.
- Gzip reduces transfers of large HTML pages. Existing API routes, form POSTs,
  provider restrictions, and health checks keep their request behavior.

## Validation and deployment

Run offline checks from the repository root with the app dependencies plus
`httpx` (test client only) installed:

```powershell
python -B -m unittest discover -s tests -v
```

Tests cover slow-provider isolation, deduplication, real timeouts, bounded work,
saved-page recovery after restart, stale labels, safe cache eviction, gzip, and
the actual Model page without network, settlement, or training calls.

Before merging, confirm Render deploys `master`. No change in instance size,
provider credentials, or data-source settings is required. After deployment:

1. Check `/healthz` and `/api/runtime-status`. The latter exposes the deployed
   commit prefix, cache counts, pending renders, and refresh failure count only.
2. Open DFS, Touchdown Hunt, Home Run Hunt, and Model. A cold data page should
   show the navigation immediately and eventually show the completed report.
3. Revisit after refresh and verify `X-Page-Cache: hit` or `stale`. Check Render
   logs for `Rendered page` durations and refresh failures, and memory metrics
   under normal traffic. A 501 was not reproduced; retain a timestamp and the
   matching Render log entry if the reported error recurs.

For rollback, redeploy the preceding commit. Setting
`SCOREEDGE_BACKGROUND_PAGES=0` disables the page renderer while retaining the
timeout correction, gzip, and read-only Model page. `SCOREEDGE_PREWARM=0` disables
all background warming (including the existing background learning tick).

Private local training code, saved learning data, credentials, and Google Drive
backups are not included in this change.
