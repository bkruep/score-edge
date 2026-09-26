"""Auction draft valuation.

Converts the 2026 redraft board (ADP + market value) into realistic auction
salary-cap dollar prices for a league configuration. The numbers are built to
look like a professional $200, 12-team PPR auction:

  - The overall #1 pick anchors near the top of the market (e.g. ~$60).
  - Everyone else scales down from that anchor by how close their market value
    (0-100) sits to the top, using a power curve so the studs take a fat share
    and the drop-off is steep but smooth.
  - Deep/bench players fall to the $1 minimum.

The Draft Assistant treats these as open-market prices: when you draft a
player in auction mode it debits that price from your team budget.
"""

DEFAULT_BUDGET = 200
TOP_ANCHOR = 60      # approximate price for the overall #1 pick in $200
POWER = 2.0          # steepness of the value->price curve
MIN_BID = 1


def auction_values(board, budget: int = DEFAULT_BUDGET, teams: int = 12, roster=None) -> dict:
    """Return player_id -> dollar price.

    ``board`` is the 2026 board (DataFrame or list of records). Uses the
    ``value_score`` column (0-100) as the market signal. All rows are priced so
    the top of the board shows realistic open-market values; bench-quality
    players drop to the $1 minimum. ``budget``/``teams`` only shape the anchor
    so the market is scaled to the league you're in (larger budget -> higher
    prices), keeping the same relative curve.
    """
    if board is None:
        return {}
    if hasattr(board, "copy") and hasattr(board, "iterrows"):
        df = board.copy()
    else:
        df = __import__("pandas").DataFrame(board)
    if df.empty or "value_score" not in df.columns:
        return {str(p): MIN_BID for p in df.get("player_id", []) if p}

    teams = max(1, int(teams or 1))
    budget = max(1, int(budget or DEFAULT_BUDGET))
    # Scale the anchor to the league budget so $100 leagues differ sensibly
    # from $200 leagues while keeping the same shape.
    anchor = TOP_ANCHOR * (budget / DEFAULT_BUDGET)

    rows = []
    for _, r in df.iterrows():
        pid = str(r["player_id"])
        vs = float(r.get("value_score", 0.0) or 0.0)
        pos = r.get("position", "")
        rows.append({"pid": pid, "vs": vs, "pos": pos})

    if not rows:
        return {}

    top_vs = max((x["vs"] for x in rows), default=1.0) or 1.0
    out: dict[str, int] = {}
    for x in rows:
        if x["vs"] <= 0:
            out[x["pid"]] = MIN_BID
            continue
        ratio = x["vs"] / top_vs
        price = anchor * (ratio ** POWER)
        out[x["pid"]] = max(MIN_BID, round(price))

    return out
