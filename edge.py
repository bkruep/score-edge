"""Sport-agnostic deep analytics over multi-book price quotes.

Every function here is pure math on price data — no sport, league, or provider
knowledge. That keeps it reusable: NFL/MLB/NBA/NHL/MLS all feed the same
engine, and college boards get it for free when they're next touched.

Five analytical families:
  1. Market structure  — devig / vig / no-vig fair prices (Shin's method)
  2. Sharp detection    — per-book price residuals vs the market consensus
  3. Movement forensics — a quote's path through the cached price history
  4. Cross-market logic — complement pairs, arbitrage, internal consistency
  5. Portfolio math     — correlation-aware sizing and true win-prob edges
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ---------------------------------------------------------------- conversions


def to_decimal(american: int | float) -> float:
    """American price -> decimal multiplier."""
    try:
        am = float(american)
    except (TypeError, ValueError):
        return 0.0
    if am == 0:
        return 0.0
    return 1.0 + am / 100.0 if am > 0 else 1.0 + 100.0 / (-am)


def to_american(decimal: float) -> int:
    """Decimal multiplier -> nearest whole american price."""
    if decimal <= 1.0:
        return -100
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100.0 / (decimal - 1.0)))


def implied_probability(american: int | float) -> float:
    """Raw implied probability including vig, as a fraction (0..1)."""
    dec = to_decimal(american)
    return 0.0 if dec <= 0 else 1.0 / dec


def expected_value(american: int | float, true_prob: float) -> float:
    """EV per unit staked given a de-vigged probability estimate."""
    return true_prob * to_decimal(american) - 1.0


def edge_points(american: int | float, consensus_american: int | float) -> float:
    """Longshot-invariant edge in implied-probability points (x100).

    Comparing raw implied probabilities penalizes longshots for the vig
    embedded in their price. This measures the best price against the
    consensus price on the probability scale instead, so a +150 with edge is
    ranked the same as a -150 with equal edge.
    """
    best = implied_probability(american)
    cons = implied_probability(consensus_american)
    if best <= 0 or cons <= 0:
        return 0.0
    return (best - cons) * 100.0


def kelly_fraction(american: int | float, true_prob: float,
                   cap: float = 0.05, scale: float = 0.5) -> float:
    """Half-Kelly stake as a bankroll fraction, capped.

    Full Kelly on a noisy probability estimate is ruinous; half-Kelly with a
    hard cap is the sane default for a price edge this small.
    """
    b = to_decimal(american) - 1.0
    if b <= 0:
        return 0.0
    p = min(max(true_prob, 0.0), 1.0)
    f = (b * p - (1.0 - p)) / b
    if f <= 0:
        return 0.0
    return min(f * scale, cap)


# ------------------------------------------------------- 1. market structure


def proportional_devig(prices: Sequence[int | float]) -> list[float]:
    """Remove the vig proportionally, preserving each price's share of the book.

    Simple, unbiased for balanced books, slightly overstates longshots.
    """
    probs = [implied_probability(p) for p in prices]
    total = sum(probs)
    if total <= 0:
        return [0.0] * len(prices)
    return [p / total for p in probs]


def shin_devig(prices: Sequence[int | float]) -> list[float]:
    """Shin's method: correct for the favourite-longshot bias.

    Longshots are systematically overbet, so their raw implied probabilities
    carry more vig than favourites'. Shin solves for the insider-trading
    parameter z and backs it out of each price, which recovers materially more
    accurate fair probabilities on longshot lines than proportional removal.
    """
    probs = [implied_probability(p) for p in prices]
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return probs
    total = sum(probs)
    if total <= 0:
        return [0.0] * n

    def excess(z: float) -> float:
        """Sum of Shin-adjusted probabilities; solve excess(z) == 1.

        Decreasing in z: at z=0 it sums to sqrt(total) > 1, and it falls as z
        rises. So a root exists only when excess(0.4) is still <= 1.
        """
        acc = 0.0
        for p in probs:
            root = math.sqrt(max(z * z + 4.0 * (1.0 - z) * p * p / total, 0.0))
            acc += (root - z) / (2.0 * (1.0 - z))
        return acc

    lo, hi = 0.0, 0.4
    if excess(lo) < 1.0 or excess(hi) > 1.0:
        # No root in range: the book carries no exploitable longshot bias.
        return proportional_devig(prices)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if excess(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    out: list[float] = []
    for p in probs:
        root = math.sqrt(max(z * z + 4.0 * (1.0 - z) * p * p / total, 0.0))
        out.append((root - z) / (2.0 * (1.0 - z)))
    s = sum(out)
    return [v / s for v in out] if s > 0 else out


def vig_percent(prices: Sequence[int | float]) -> float:
    """Book overround as a percentage of the fair total (100 = no vig)."""
    total = sum(implied_probability(p) for p in prices)
    return (total - 1.0) * 100.0 if total > 0 else 0.0


def hold_pct(prices: Sequence[int | float]) -> float:
    """Expected hold per unit staked, i.e. the vig as a fraction of handle."""
    total = sum(implied_probability(p) for p in prices)
    return (total - 1.0) / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class MarketStructure:
    """Full picture of one market's pricing: raw, devigged, and what the vig is."""
    n_outcomes: int
    implied: list[float]
    fair: list[float]
    vig_pct: float
    hold_pct: float
    method: str
    sharp_ratio: float | None = None

    def fair_for(self, idx: int) -> float:
        return self.fair[idx] if 0 <= idx < len(self.fair) else 0.0


def analyze_market(prices: Sequence[int | float], method: str = "shin") -> MarketStructure:
    """Devig a complete market's prices into fair probabilities.

    `method='shin'` corrects favourite-longshot bias (better for moneyline and
    longshot props); `method='proportional'` is the conservative fallback.
    """
    prices = [p for p in prices if p]
    if not prices:
        return MarketStructure(0, [], [], 0.0, 0.0, method)
    imp = [implied_probability(p) for p in prices]
    used = method
    if method == "shin":
        fair = shin_devig(prices)
        # shin_devig silently falls back to proportional when the market has no
        # exploitable longshot bias (even money, or an underround). Report that
        # honestly instead of crediting the result to a method that never ran.
        if abs(fair[0] - proportional_devig(prices)[0]) < 1e-9:
            used = "proportional"
    else:
        fair = proportional_devig(prices)
        used = "proportional"
    return MarketStructure(
        n_outcomes=len(prices),
        implied=imp,
        fair=fair,
        vig_pct=vig_percent(prices),
        hold_pct=hold_pct(prices),
        method=used,
    )


# ------------------------------------------------------- 2. sharp detection

# Books that historically price closest to the closing line. Used as a prior;
# measured residuals always take precedence over this list.
SHARP_BOOKS = frozenset({
    "pinnacle", "betfair_ex_eu", "betfair", "smarkets", "matchbook",
    "betcris", "betonlineag", "lowvig", "1xbet", "marathonbet",
    "williamhill", "unibet_us", "bovada",
})
SOFT_BOOKS = frozenset({
    "betrivers", "bovada_lv", "fanduel", "draftkings", "pointsbetus",
    "ballys", "espn_bet", "cbssports", "fanatics", "mybookieag",
    "betmgm", "caesars", "bet365",
})


def book_tier(book: str) -> str:
    """Classify a book as 'sharp', 'soft', or 'unknown'."""
    key = (book or "").strip().casefold().replace(" ", "").replace("-", "").replace("_", "")
    for sharp in SHARP_BOOKS:
        if key == sharp.replace(" ", "").replace("-", "").replace("_", ""):
            return "sharp"
    for soft in SOFT_BOOKS:
        if key == soft.replace(" ", "").replace("-", "").replace("_", ""):
            return "soft"
    return "unknown"


@dataclass(frozen=True, slots=True)
class BookResidual:
    """How far one book's price sits from where the rest of the market is."""
    book: str
    tier: str
    price: int
    implied: float
    residual: float
    n_quotes: int

    @property
    def is_outlier_high(self) -> bool:
        """Offering a materially better price than the market — often the sharpest."""
        return self.residual >= 0.015

    @property
    def is_outlier_low(self) -> bool:
        return self.residual <= -0.015


def book_residuals(book_odds: dict[str, int]) -> list[BookResidual]:
    """Per-book deviation from the market's mean implied probability.

    A book quoting 3 points above the field on the same selection is telling
    you either its model disagrees with the market or it's shading a hot line.
    Both are tradeable information; the tier label says how much to trust it.
    """
    if not book_odds or len(book_odds) < 2:
        return []
    items = [(b, int(p)) for b, p in book_odds.items() if p]
    if len(items) < 2:
        return []
    mean_imp = sum(implied_probability(p) for _, p in items) / len(items)
    out = []
    for book, price in items:
        imp = implied_probability(price)
        out.append(BookResidual(
            book=book,
            tier=book_tier(book),
            price=price,
            implied=imp,
            residual=imp - mean_imp,
            n_quotes=len(items),
        ))
    return sorted(out, key=lambda r: r.residual, reverse=True)


@dataclass(frozen=True, slots=True)
class SharpSignal:
    """Aggregate read on whether sharp money is present in a market."""
    n_quotes: int
    n_sharp: int
    sharp_mean_residual: float
    soft_mean_residual: float
    leader: BookResidual | None
    n_outliers: int

    @property
    def split(self) -> float:
        """Sharp minus soft mean residual — positive means sharp books are longer."""
        return self.sharp_mean_residual - self.soft_mean_residual

    @property
    def confidence(self) -> float:
        """0..1 confidence that the sharp/soft split is meaningful, gated on coverage."""
        if self.n_quotes < 4:
            return 0.0
        coverage = min(self.n_sharp / 3.0, 1.0)
        return round(coverage * min(abs(self.split) / 0.03, 1.0), 3)

    @property
    def label(self) -> str:
        if self.n_quotes < 4 or self.n_sharp == 0:
            return "unclear"
        if self.split > 0.012:
            return "sharp-long"
        if self.split < -0.012:
            return "sharp-short"
        return "balanced"


def sharp_signal(book_odds: dict[str, int]) -> SharpSignal:
    """Summarize sharp-vs-soft positioning across a market's books."""
    residuals = book_residuals(book_odds)
    if not residuals:
        return SharpSignal(0, 0, 0.0, 0.0, None, 0)
    sharps = [r.residual for r in residuals if r.tier == "sharp"]
    softs = [r.residual for r in residuals if r.tier == "soft"]
    n_quotes = max(r.n_quotes for r in residuals)
    return SharpSignal(
        n_quotes=n_quotes,
        n_sharp=len(sharps),
        sharp_mean_residual=(sum(sharps) / len(sharps)) if sharps else 0.0,
        soft_mean_residual=(sum(softs) / len(softs)) if softs else 0.0,
        leader=residuals[0],
        n_outliers=sum(1 for r in residuals if r.is_outlier_high or r.is_outlier_low),
    )


# -------------------------------------------------- 3. movement forensics


@dataclass(frozen=True, slots=True)
class PricePoint:
    """One timestamped observation of a quote's best price."""
    ts: float
    price: int


@dataclass(frozen=True, slots=True)
class Movement:
    """How a price travelled from first sighting to now."""
    open_price: int | None
    current_price: int
    change: int
    change_pct: float
    n_points: int
    span_seconds: float
    path: list[int]
    volatility: float
    direction: str

    @property
    def is_steamer(self) -> bool:
        """Moved steadily shorter — money pressing one way."""
        return self.direction == "shorter" and self.volatility < 0.02

    @property
    def is_sharp_move(self) -> bool:
        """A real move, not noise: >=2 price steps inside a tight path."""
        return abs(self.change) >= 20 and self.volatility < 0.03


def movement(history: Sequence[PricePoint], current: int | None = None) -> Movement | None:
    """Reduce a price history to a readable move.

    Volatility is the standard deviation of consecutive steps, normalized by the
    average price: low volatility with a real net change means a clean,
    one-directional move rather than a book bouncing the number.
    """
    pts = [p for p in history if p and p.price]
    if not pts:
        return None
    prices = [p.price for p in pts]
    cur = current if current else prices[-1]
    opening = prices[0]
    change = cur - opening
    denom = abs(opening) if opening else 1.0
    change_pct = change / denom

    steps = [b - a for a, b in zip(prices, prices[1:])]
    if len(steps) > 1:
        mean = sum(steps) / len(steps)
        var = sum((s - mean) ** 2 for s in steps) / len(steps)
        vol = math.sqrt(var) / denom
    else:
        vol = 0.0

    span = pts[-1].ts - pts[0].ts
    if change > 0:
        direction = "longer"
    elif change < 0:
        direction = "shorter"
    else:
        direction = "flat"
    return Movement(
        open_price=opening,
        current_price=cur,
        change=change,
        change_pct=change_pct,
        n_points=len(pts),
        span_seconds=span,
        path=prices[-12:],
        volatility=round(vol, 4),
        direction=direction,
    )


@dataclass
class PriceHistory:
    """Bounded time series of prices keyed by quote identity.

    Persisted so movement reads survive a restart. Keeps a rolling window and
    caps per-key points so a chatty market can't grow the file without bound.
    """
    series: dict[str, list[PricePoint]] = field(default_factory=dict)
    max_points: int = 60
    max_keys: int = 4000

    def record(self, key: str, price: int, ts: float) -> None:
        if not key or not price:
            return
        pts = self.series.setdefault(key, [])
        if pts and abs(pts[-1].ts - ts) < 30:
            pts[-1] = PricePoint(ts, price)
        else:
            pts.append(PricePoint(ts, price))
        if len(pts) > self.max_points:
            del pts[: len(pts) - self.max_points]
        if len(self.series) > self.max_keys:
            oldest = min(self.series, key=lambda k: self.series[k][-1].ts)
            self.series.pop(oldest, None)

    def get(self, key: str) -> list[PricePoint]:
        return self.series.get(key, [])

    def movement(self, key: str, current: int | None = None) -> Movement | None:
        return movement(self.get(key), current)

    def movers(self, min_change: int = 15, limit: int = 20) -> list[tuple[str, Movement]]:
        """Biggest movers across the tracked series, most extreme first."""
        out: list[tuple[str, Movement]] = []
        for key, pts in self.series.items():
            mv = movement(pts)
            if mv and abs(mv.change) >= min_change:
                out.append((key, mv))
        out.sort(key=lambda kv: abs(kv[1].change), reverse=True)
        return out[:limit]

    def trim(self) -> None:
        """Drop series with no recent observation."""
        import time as _time
        cutoff = _time.time() - 7 * 86400
        for key in [k for k, v in self.series.items() if v and v[-1].ts < cutoff]:
            self.series.pop(key, None)


def quote_key(event_id: str, market: str, selection: str, line: float | None,
              player: str | None = None) -> str:
    """Stable identity for a quote across refreshes."""
    side = (selection or "").strip().casefold()
    who = (player or "").strip().casefold()
    ln = "" if line is None else f"{float(line):g}"
    return "|".join([event_id or "", market or "", who, side, ln])


_SIDE_TOKENS = frozenset({"over", "under", "home", "away", "draw"})

# Above this, a two-sided "arb" is a data defect, not an opportunity. Genuine
# arbitrage in a liquid market lands in the low single digits.
MAX_REAL_ARB_PCT = 8.0

# Bounds for a coherent book on a real market. Composite best prices from
# different books can sit a point or two under fair, but a large negative
# overround means the outcomes were never one market.
MIN_PLAUSIBLE_VIG_PCT = -4.0
MAX_PLAUSIBLE_VIG_PCT = 8.0

# Soccer 1X2 legitimately runs a fatter margin than a two-way market: the draw
# leg is expensive to carry, and even sharp books sit 8-12% on a balanced MLS
# game. Judging it by the two-way band would flag healthy books all day.
MAX_PLAUSIBLE_VIG_3WAY_PCT = 12.0


def max_plausible_vig(n_outcomes: int) -> float:
    """Upper bound on overround for a market of this shape."""
    return MAX_PLAUSIBLE_VIG_3WAY_PCT if n_outcomes >= 3 else MAX_PLAUSIBLE_VIG_PCT


def side_key(quote) -> str:
    """Identity of the outcome a quote represents, ignoring label variants.

    The feed posts one side under several labels in the same market ("Chicago
    White Sox" and "CHI White Sox"). Every market-level computation — devig,
    arbitrage, complement pairing, consistency — must count *outcomes*, not
    rows, or a perfectly normal two-way market looks like three or four
    outcomes and every derived number is wrong.

    Totals/three-way sides use their own token; team sides collapse onto the
    trailing word, which is the mascot ("White Sox" from "CHI White Sox").
    """
    sel = (getattr(quote, "selection", "") or "").strip()
    low = sel.casefold()
    if low in _SIDE_TOKENS:
        return low
    words = [w for w in sel.replace(".", " ").split() if w]
    return (words[-1] if words else sel).casefold()


def collapse_sides(quotes: Iterable) -> dict:
    """Best-price quote per distinct outcome within one market.

    Ties prefer the row carrying more books, since that yields a better
    consensus and sharper residuals downstream.
    """
    sides: dict[str, object] = {}
    for q in quotes or []:
        key = side_key(q)
        cur = sides.get(key)
        if cur is None:
            sides[key] = q
            continue
        a, b = int(getattr(q, "best_odds", 0) or 0), int(getattr(cur, "best_odds", 0) or 0)
        if a > b or (a == b and len(getattr(q, "book_odds", None) or {}) >
                     len(getattr(cur, "book_odds", None) or {})):
            sides[key] = q
    return sides


def group_markets(quotes: Iterable) -> dict:
    """Group quotes by (event, market, line) — the real unit of a market.

    Keying on `line` matters: one event posts several spread lines, and summing
    their prices would manufacture a huge fake overround.

    Spreads are keyed on the *magnitude* of the line, because a spread's two
    outcomes are mirrors — Home at -2.5 and Away at +2.5 are the same bet. Keying
    those separately would leave every spread market as two lone rows, hiding
    both its overround and any arbitrage.
    """
    out: dict[tuple, list] = defaultdict(list)
    for q in quotes or []:
        line = getattr(q, "line", None)
        market = getattr(q, "market", "")
        key = None if line is None else round(abs(float(line)), 2)
        if market == "point_spread" and key is not None:
            line = key
        # Player is part of the key: ten hitters each offering "over 1.5 hits"
        # at the same event are ten unrelated markets. Devigging them together
        # would average unrelated players into one nonsense price.
        player = (getattr(q, "player_name", "") or "").strip().casefold() or None
        out[(getattr(q, "event_id", ""), market, player, line)].append(q)
    return out


# -------------------------------------------------- 4. cross-market logic


@dataclass(frozen=True, slots=True)
class ArbOpportunity:
    """A guaranteed-profit pairing of two mutually exclusive outcomes."""
    key_a: str
    key_b: str
    label_a: str
    label_b: str
    price_a: int
    price_b: int
    combined_decimal: float
    profit_pct: float
    stake_a: float
    stake_b: float
    payout: float

    @property
    def is_guaranteed(self) -> bool:
        return self.profit_pct > 0

    # A "profit" this large is not an opportunity, it is a broken feed. Books do
    # not leave 40% on the table in liquid markets; that magnitude means the two
    # prices are not the same market (lost spread sign, stale/mismatched line).
    suspicious: bool = False
    note: str = ""

    @property
    def is_actionable(self) -> bool:
        """Reportable and stakeable — not a data defect wearing a profit."""
        return self.profit_pct > 0 and not self.suspicious


def find_arbitrage(pairs: Iterable[tuple[str, int, str, int]],
                   min_profit: float = 0.5) -> list[ArbOpportunity]:
    """Detect guaranteed-profit pairs across mutually exclusive outcomes.

    Each pair is (key_a, price_a, label_a, key_b, price_b, label_b). Combined
    decimal below 1.0 means both sides can be staked for a locked profit —
    rare in liquid markets and always worth surfacing.
    """
    out: list[ArbOpportunity] = []
    for key_a, price_a, label_a, key_b, price_b, label_b in pairs:
        da, db = to_decimal(price_a), to_decimal(price_b)
        if da <= 0 or db <= 0:
            continue
        # An arb needs the *inverse* decimals to sum under 1.0 — i.e. the two
        # implied probabilities total less than certainty. Summing the decimals
        # themselves is meaningless here: +130/+130 has a combined decimal of
        # 4.6 yet is a 15% locked profit.
        combined = 1.0 / da + 1.0 / db
        if combined >= 1.0:
            continue
        profit_pct = (1.0 / combined - 1.0) * 100.0
        if profit_pct < min_profit:
            continue
        # Equalize payout: stake each leg at the inverse of its own decimal.
        # Round the stakes first, then derive payout from the rounded values so
        # the printed numbers actually reconcile.
        stake_a = round(1.0 / da, 4)
        stake_b = round(1.0 / db, 4)
        payout = round(min(stake_a * da, stake_b * db), 4)
        total_staked = stake_a + stake_b
        profit_pct = (payout / total_staked - 1.0) * 100.0 if total_staked else 0.0
        if profit_pct < min_profit:
            continue
        # Real arbitrage lives in the low single digits. Beyond MAX_REAL_ARB the
        # pair is not actually a two-outcome market, so label it as a feed defect
        # instead of printing a 40% "guaranteed profit" nobody can stake.
        suspicious = profit_pct > MAX_REAL_ARB_PCT
        out.append(ArbOpportunity(
            key_a=key_a, key_b=key_b,
            label_a=label_a, label_b=label_b,
            price_a=price_a, price_b=price_b,
            combined_decimal=round(combined, 4),
            profit_pct=round(profit_pct, 2),
            stake_a=stake_a,
            stake_b=stake_b,
            payout=payout,
            suspicious=suspicious,
            note=("prices imply probabilities summing to "
                  f"{combined:.2f}, not a real two-sided market — likely a "
                  "lost spread sign or stale line" if suspicious else ""),
        ))
    return sorted(out, key=lambda a: a.profit_pct, reverse=True)


def complement_pairs(quotes: Iterable) -> list[tuple[str, int, str, int, str, int]]:
    """Pair mutually exclusive outcomes of the same market/line.

    Works off any object exposing `.market`, `.line`, `.selection`, `.best_odds`
    — i.e. BetQuote — so it covers moneyline, spread, totals, and props.

    Crucially this pairs *distinct outcomes*, not distinct rows: the feed posts
    the same side twice under different labels, and pairing those would report a
    fake guaranteed profit on a bet that can only lose money.
    """
    pairs: list[tuple[str, int, str, int, str, int]] = []
    for (_, market, _player, line), group in group_markets(quotes).items():
        sides = collapse_sides(group)
        if len(sides) < 2:
            continue
        keys = list(sides.keys())
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                a, b = sides[ka], sides[kb]
                # Label from each quote's own line, not the group key: spreads
                # share a magnitude but sit on opposite signs.
                pairs.append((
                    quote_key(a.event_id, market, a.selection, a.line, a.player_name),
                    int(a.best_odds), f"{a.selection}" + _line_suffix(getattr(a, "line", None)),
                    quote_key(b.event_id, market, b.selection, b.line, b.player_name),
                    int(b.best_odds), f"{b.selection}" + _line_suffix(getattr(b, "line", None)),
                ))
    return pairs


def _line_suffix(line) -> str:
    return "" if line is None else f" {float(line):+g}"


@dataclass(frozen=True, slots=True)
class ConsistencyIssue:
    """A market whose own prices contradict each other."""
    event: str
    market: str
    issue: str
    detail: str
    severity: str
    line: float | None = None


def internal_consistency(quotes: Iterable) -> list[ConsistencyIssue]:
    """Find books' internal contradictions — the highest-quality signal available.

    Two independent books disagreeing by more than the spread can justify is a
    genuine pricing error rather than a model difference. Over/under pairs that
    imply an impossible total, and moneyline/spread pairs whose lines disagree,
    are the same class of defect.
    """
    issues: list[ConsistencyIssue] = []
    # Group by event/market/player/line and count distinct outcomes: one spread
    # line is one market, and its two teams are its only two prices.
    for (_, market, _player, line), group in group_markets(quotes).items():
        sides = collapse_sides(group)
        prices = [int(q.best_odds) for q in sides.values() if getattr(q, "best_odds", None)]
        if len(prices) < 2:
            continue
        # A spread's two outcomes must sit on opposite sides of the line: Home
        # -1.5 and Away +1.5. When the feed hands us the same sign for both, the
        # line sign was lost upstream and the pair is not a two-sided market —
        # devigging it yields a negative vig, which is nonsense. Flag it instead.
        if market == "point_spread" and len(sides) == 2:
            signs = set()
            for q in sides.values():
                try:
                    val = float(getattr(q, "line", 0.0) or 0.0)
                except (TypeError, ValueError):
                    val = 0.0
                signs.add(1 if val > 0 else (-1 if val < 0 else 0))
            if len(signs) == 1 and 0 not in signs:
                issues.append(ConsistencyIssue(
                    event=group[0].game, market=market,
                    line=None if line is None else line,
                    issue="line_sign_mismatch",
                    detail=("both outcomes carry the same spread sign; "
                            f"cannot devig a one-sided market at {line:g}"),
                    severity="high",
                ))
                continue

        structure = analyze_market(prices)
        # A best-price composite can dip a point or two under fair value when it
        # mixes prices from different books, but a large negative overround
        # cannot happen: it means the outcomes are not one market. Devigging it
        # would hand back confident-looking fair probabilities built on a
        # contradiction, which is worse than reporting nothing.
        if structure.vig_pct < MIN_PLAUSIBLE_VIG_PCT:
            where = f" at {line:g}" if line is not None else ""
            issues.append(ConsistencyIssue(
                event=group[0].game, market=market, line=line,
                issue="implausible_underround",
                detail=(f"{structure.vig_pct:.1f}% underround across "
                        f"{len(prices)} outcome(s){where}; prices cannot "
                        "be one two-sided market"),
                severity="high",
            ))
            continue
        # A coherent book stays within the plausible band for this market's shape.
        # Past that the rows are not describing one market, which is the defect.
        if structure.vig_pct > max_plausible_vig(structure.n_outcomes):
            where = f" at {line:g}" if line is not None else ""
            issues.append(ConsistencyIssue(
                event=group[0].game, market=market,
                line=line,
                issue="excess_vig",
                detail=(f"{structure.vig_pct:.1f}% overround across "
                        f"{len(prices)} outcome(s){where}"),
                severity="high",
            ))

    # Over/under totals should roughly straddle 50/50.
    for (event_id, market, _player, line), group in group_markets(quotes).items():
        if market != "total_points" or line is None:
            continue
        sides = collapse_sides(group)
        over = next((q for k, q in sides.items() if k == "over"), None)
        under = next((q for k, q in sides.items() if k == "under"), None)
        if over is None or under is None:
            continue
        fair = proportional_devig([int(over.best_odds), int(under.best_odds)])
        skew = abs(fair[0] - 0.5)
        if skew > 0.12:
            issues.append(ConsistencyIssue(
                event=group[0].game, market="total_points", line=line,
                issue="total_skew",
                detail=f"devig says {fair[0]*100:.0f}/{fair[1]*100:.0f} at {line:g}",
                severity="medium",
            ))
    return issues


# ------------------------------------------------------ 5. portfolio math


@dataclass(frozen=True, slots=True)
class Position:
    """A priced candidate with everything needed to size it."""
    key: str
    label: str
    price: int
    fair_prob: float
    group: str = ""
    edge: float = 0.0
    book: str = ""


@dataclass(frozen=True, slots=True)
class PortfolioLeg:
    """One sized leg of a correlated portfolio."""
    key: str
    label: str
    price: int
    fair_prob: float
    stake: float
    ev: float
    edge: float
    kelly: float

    @property
    def to_win(self) -> float:
        return self.stake * to_decimal(self.price)


def position_from_quote(quote, fair_prob: float | None = None,
                        book_odds: dict | None = None) -> Position | None:
    """Build a Position from a BetQuote, devigging its sibling sides for fair_prob."""
    if not quote.best_odds:
        return None
    group = quote.game or quote.event_id
    if quote.player_name:
        group = f"{group}|{quote.player_name}"
    fair = fair_prob
    if fair is None:
        prices = [int(p) for p in (book_odds or {}).values() if p]
        if len(prices) >= 2:
            fair = shin_devig(prices)[0]
        else:
            fair = implied_probability(quote.best_odds)
    return Position(
        key=quote_key(quote.event_id, quote.market, quote.selection, quote.line, quote.player_name),
        label=(f"{quote.player_name} {quote.selection}" if quote.player_name else quote.selection)
              + (f" {quote.line:g}" if quote.line is not None else ""),
        price=int(quote.best_odds),
        fair_prob=fair,
        group=group,
        edge=expected_value(quote.best_odds, fair) * 100.0,
        book=quote.best_book,
    )


def build_portfolio(positions: Sequence[Position], bankroll: float = 100.0,
                    max_stake_pct: float = 0.02, max_legs: int = 8,
                    group_cap: float = 0.06) -> dict:
    """Size a portfolio with correlation-aware de-duplication.

    Same-game and same-player legs share outcome risk, so they compete for a
    group budget instead of each getting an independent stake. That prevents
    the classic parlay mistake of treating eight correlated props as eight
    independent edges.
    """
    ranked = sorted(positions, key=lambda p: p.edge, reverse=True)
    chosen: list[PortfolioLeg] = []
    group_used: dict[str, float] = defaultdict(float)
    total_staked = 0.0
    expected_profit = 0.0

    for p in ranked:
        if len(chosen) >= max_legs:
            break
        ev = expected_value(p.price, p.fair_prob)
        if ev <= 0:
            continue
        f = kelly_fraction(p.price, p.fair_prob, cap=max_stake_pct)
        if f <= 0:
            continue
        stake = f * bankroll
        room = group_cap * bankroll - group_used[p.group]
        if room <= 0:
            continue
        stake = min(stake, room, bankroll - total_staked)
        if stake < 0.5:
            continue
        group_used[p.group] += stake
        total_staked += stake
        expected_profit += stake * ev
        chosen.append(PortfolioLeg(
            key=p.key, label=p.label, price=p.price, fair_prob=p.fair_prob,
            stake=round(stake, 2), ev=round(ev * 100, 2),
            edge=round(p.edge, 2), kelly=round(f * 100, 2),
        ))

    if not chosen:
        return {"legs": [], "staked": 0.0, "to_win": 0.0, "expected_profit": 0.0,
                "roi": 0.0, "groups": 0, "note": "no positive-EV candidates"}
    to_win = sum(leg.to_win for leg in chosen)
    return {
        "legs": chosen,
        "staked": round(total_staked, 2),
        "to_win": round(to_win, 2),
        "expected_profit": round(expected_profit, 2),
        "roi": round(expected_profit / total_staked * 100, 2) if total_staked else 0.0,
        "groups": len(group_used),
        "note": "",
    }


def portfolio_summary(portfolio: dict) -> str:
    """One-line human read of a portfolio for KPI tiles."""
    legs = portfolio.get("legs") or []
    if not legs:
        return "no +EV candidates"
    return (f"{len(legs)} legs · {portfolio['groups']} games · "
            f"${portfolio['staked']:.0f} staked → ${portfolio['to_win']:.0f} "
            f"({portfolio['roi']:+.1f}% expected)")