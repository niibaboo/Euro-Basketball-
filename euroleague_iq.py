#!/usr/bin/env python3
"""
EuroLeague IQ — EuroLeague Basketball Team Points + Player Props Predictor
--------------------------------------------------------------
Same architecture/sibling as Blitz IQ (NFL), Orange Line (NBA), Match IQ,
etc: recency-weighted recent form, small-sample shrinkage toward a league
average, Normal distribution for team point totals, a Safest Bet Builder,
and a results tracker.

DATA SOURCE: EuroLeague's own public v2 REST API --
  https://api-live.euroleague.net/v2/competitions/E/seasons/{seasonCode}/...
This is NOT officially documented by EuroLeague, but it's a genuinely
public, keyless, unauthenticated API confirmed via its own Swagger page
(api-live.euroleague.net/swagger) and used in production by several
open-source clients (e.g. github.com/aimon7/euroleague-api, whose TypeScript
source + response fixtures this file's field names/URLs were confirmed
against). Season code format is "E<year>" where <year> is the season's
START year, e.g. "E2026" for the 2026-27 season -- EuroCup uses "U"
instead of "E".

IMPORTANT CAVEAT: this session's sandbox can't reach external hosts other
than GitHub/package registries (confirmed -- both ESPN and this API were
blocked by the sandbox's own egress proxy when tested directly), so the
endpoints/field names below are confirmed from the open-source client's
*documented, tested* source and fixtures, not a live response fetched in
this session. If something comes back empty or [!]/[DIAG] prints show up
in the Actions log, that's the first place to look -- same defensive style
as every other tool in this suite that works against an undocumented API.

Team form: walks each team's own last RECENT_GAMES *played* games from the
season schedule, pulling each game's final score from the box score's
local/road 'total.points' (NOT ESPN -- this is EuroLeague's own confirmed
field name). Normal-distribution team totals, same as Blitz IQ.

Player props: EuroLeague's API has no simple "starters" endpoint like
ESPN's NFL depth chart, so instead of a separate roster fetch, a team's
featured players are derived straight from the SAME recent box scores
already being pulled for team form -- whichever players actually appeared
in at least PLAYER_MIN_GAMES of the last RECENT_GAMES games, ranked by
average points and capped to PLAYER_POOL_SIZE. This is arguably more
robust than a static depth chart (it reflects actual recent rotation/
minutes, not a preseason roster slot). Props offered: Points (normal),
Rebounds (poisson), Assists (poisson) -- same distribution split Blitz IQ
uses for yardage-like vs. count-like stats.

Output:
    docs/index.html                       -- predictions page
    docs/euroleague_iq_predictions.csv    -- team totals
    docs/euroleague_iq_player_props.csv   -- player props
    docs/euroleague_iq.json               -- raw predictions

This repo is standalone (one tool, not a multi-tool suite like the main
sportsiq repo), so output goes straight to docs/ root rather than nested
under a docs/<tool-name>/ subfolder -- GitHub Pages serving from
main branch /docs then just works with no separate landing page needed.

NOT included in this first version (can be added later, same as every
other tool's incremental history): Hot Form / Real Streak panels. Kept out
to keep this first cut's scope sane -- team totals + player props +
Safest Bet Builder + results tracker is already "the works" from Blitz
IQ's own feature set.
"""

import os
import csv
import json
import math
import time
import requests
from datetime import datetime, timezone, timedelta

BASE = "https://api-live.euroleague.net/v2/competitions/E"
REQUEST_DELAY = 1.0          # paced, same reasoning as every other tool here --
                               # be polite to an undocumented API, especially
                               # one with no published rate limit.
RECENT_GAMES = 5
PRIOR_STRENGTH = 3
DEFAULT_TEAM_STD = 9.0        # points -- EuroLeague games run lower-scoring
                               # and lower-variance than NBA, this is a
                               # starting estimate, not a fitted value.
UPCOMING_WINDOW_DAYS = 8       # how far ahead to scan the schedule for games
                               # to predict -- EuroLeague doesn't expose a
                               # simple "this week" endpoint the way ESPN's
                               # NFL scoreboard does, so this filters the
                               # full season schedule by date instead.
PLAYER_MIN_GAMES = 2           # a player needs to have appeared in at least
                               # this many of the last RECENT_GAMES games to
                               # be considered a "featured" player worth a prop
PLAYER_POOL_SIZE = 5           # cap per team, same rough scale as Blitz IQ's
                               # one-starter-per-position (QB/RB/WR/TE = 4)


def current_season_year():
    """EuroLeague's own JS client computes this the same way: a season
    "E2026" runs Oct 2026 - May/June 2027, so from July onward we're
    already in next season's year; before July we're still finishing the
    season that started the PREVIOUS year."""
    today = datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


def season_code(year=None):
    return f"E{year if year is not None else current_season_year()}"


def _get(url, params=None):
    time.sleep(REQUEST_DELAY)
    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        print(f"  [!] request failed: {url} ({e})")
        return None
    if r.status_code != 200:
        print(f"  [!] {r.status_code} on {url}")
        return None
    return r


def norm_cdf(x, mean, std):
    if std is None or std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))


schedule_cache = {}


def get_schedule(season):
    """Full season schedule -- every game, played and upcoming. One call
    covers the whole season, cached per season so team-form/player-prop
    lookups across every game on the slate share this single fetch
    instead of each team re-requesting it."""
    if season in schedule_cache:
        return schedule_cache[season]
    r = _get(f"{BASE}/seasons/{season}/games")
    games = []
    if r is not None:
        try:
            games = r.json().get('data', [])
        except Exception as e:
            print(f"  [!] couldn't parse schedule for season {season}: {e}")
    schedule_cache[season] = games
    return games


game_stats_cache = {}


def get_game_stats(season, game_code):
    """Per-game box score -- the rich v2 'stats' endpoint (points,
    totalRebounds, assistances, etc. per player, confirmed field names),
    not the legacy live-feed boxscore which only carries raw points.
    Cached per (season, game_code) since a team's own last-N-games lookup
    and its opponent's both end up requesting some of the same games."""
    key = (season, game_code)
    if key in game_stats_cache:
        return game_stats_cache[key]
    r = _get(f"{BASE}/seasons/{season}/games/{game_code}/stats")
    stats = None
    if r is not None:
        try:
            stats = r.json()
        except Exception as e:
            print(f"    [!] couldn't parse game stats for game {game_code}: {e}")
    game_stats_cache[key] = stats
    return stats


def _team_games(schedule, tv_code):
    """This team's PLAYED games, oldest-first by date -- same ordering
    convention recency_weighted() relies on everywhere else in this
    suite (higher weight for a later index)."""
    games = []
    for g in schedule:
        if not g.get('played'):
            continue
        local = (g.get('local') or {}).get('club') or {}
        road = (g.get('road') or {}).get('club') or {}
        if local.get('tvCode') != tv_code and road.get('tvCode') != tv_code:
            continue
        games.append(g)
    games.sort(key=lambda g: g.get('date') or '')
    return games


def recency_weighted(values):
    n = len(values)
    if n == 0:
        return None
    wts = [1.3 ** i for i in range(n)]
    return sum(w * v for w, v in zip(wts, values)) / sum(wts)


def shrink(value, n, league_avg, prior=PRIOR_STRENGTH):
    return (n * value + prior * league_avg) / (n + prior)


team_form_cache = {}


def get_team_form(tv_code, schedule, season):
    """Last RECENT_GAMES played games' scored/allowed points, pulled from
    each game's box score 'total.points' on whichever side (local/road)
    this team was in that specific game."""
    if (season, tv_code) in team_form_cache:
        return team_form_cache[(season, tv_code)]

    recent = _team_games(schedule, tv_code)[-RECENT_GAMES:]
    scored, allowed = [], []
    for g in recent:
        gc = g.get('gameCode')
        if gc is None:
            continue
        stats = get_game_stats(season, gc)
        if not stats:
            continue
        local_code = ((g.get('local') or {}).get('club') or {}).get('tvCode')
        side, opp_side = ('local', 'road') if local_code == tv_code else ('road', 'local')
        pts = (stats.get(side) or {}).get('total', {}).get('points')
        opp_pts = (stats.get(opp_side) or {}).get('total', {}).get('points')
        if pts is None or opp_pts is None:
            print(f"    [!] game {gc}: couldn't find total.points for one or both sides "
                  f"(side={side}) -- box score shape may not match what was confirmed "
                  f"against the open-source client's fixtures.")
            continue
        scored.append(float(pts))
        allowed.append(float(opp_pts))

    if not scored:
        team_form_cache[(season, tv_code)] = None
        return None

    n = len(scored)
    form = {
        'avg_scored': round(sum(scored) / n, 1),
        'avg_allowed': round(sum(allowed) / n, 1),
        'n_games': n,
        'scored_list': scored,
        'allowed_list': allowed,
    }
    team_form_cache[(season, tv_code)] = form
    return form


def league_averages(all_forms):
    scored = [f['avg_scored'] for f in all_forms if f]
    allowed = [f['avg_allowed'] for f in all_forms if f]
    lg_scored = sum(scored) / len(scored) if scored else 80.0
    lg_allowed = sum(allowed) / len(allowed) if allowed else 80.0
    return lg_scored, lg_allowed


def predict(h_form, a_form, lg_scored, lg_allowed):
    h_recent_scored = recency_weighted(h_form['scored_list'])
    h_recent_allowed = recency_weighted(h_form['allowed_list'])
    a_recent_scored = recency_weighted(a_form['scored_list'])
    a_recent_allowed = recency_weighted(a_form['allowed_list'])

    h_scored = shrink(h_recent_scored, h_form['n_games'], lg_scored)
    h_allowed = shrink(h_recent_allowed, h_form['n_games'], lg_allowed)
    a_scored = shrink(a_recent_scored, a_form['n_games'], lg_scored)
    a_allowed = shrink(a_recent_allowed, a_form['n_games'], lg_allowed)

    exp_home = h_scored * (a_allowed / lg_allowed)
    exp_away = a_scored * (h_allowed / lg_allowed)
    exp_total = round(exp_home + exp_away, 1)

    total_std = math.sqrt(DEFAULT_TEAM_STD ** 2 + DEFAULT_TEAM_STD ** 2)

    return {
        'exp_home': round(exp_home, 1), 'exp_away': round(exp_away, 1),
        'exp_total': exp_total, 'total_std': round(total_std, 1),
    }


# stat_key (box score field) -> prop config. Points uses Normal (continuous,
# higher-variance, same treatment as Blitz IQ's QB Passing Yards); Rebounds/
# Assists use Poisson (lower counts, same treatment as Blitz IQ's WR/TE
# Receptions).
PLAYER_PROP_CONFIG = {
    'points':      {'label': 'Points',   'dist': 'normal',  'std': 6.0, 'prior': 11.0},
    'totalRebounds': {'label': 'Rebounds', 'dist': 'poisson', 'std': None, 'prior': 4.0},
    'assistances': {'label': 'Assists',  'dist': 'poisson', 'std': None, 'prior': 2.5},
}


def get_team_player_pool(tv_code, games, season):
    """Builds each player's recent per-stat gamelog straight from the same
    box scores team form already fetched (shared via game_stats_cache --
    no extra API calls). Returns the top PLAYER_POOL_SIZE players by
    average points among those who appeared in at least PLAYER_MIN_GAMES
    of the last RECENT_GAMES games -- a usage-based stand-in for "starters"
    since this API has no separate depth-chart endpoint."""
    per_player = {}
    for g in games:
        gc = g.get('gameCode')
        if gc is None:
            continue
        stats = get_game_stats(season, gc)
        if not stats:
            continue
        local_code = ((g.get('local') or {}).get('club') or {}).get('tvCode')
        side = 'local' if local_code == tv_code else 'road'
        team_block = stats.get(side) or {}
        for p in team_block.get('players', []):
            person = ((p.get('player') or {}).get('person')) or {}
            code = person.get('code')
            name = person.get('name')
            if not code:
                continue
            st = p.get('stats') or {}
            entry = per_player.setdefault(code, {'name': name, 'points': [], 'totalRebounds': [], 'assistances': []})
            for stat_key in PLAYER_PROP_CONFIG:
                v = st.get(stat_key)
                if v is not None:
                    try:
                        entry[stat_key].append(float(v))
                    except (TypeError, ValueError):
                        continue

    pool = [(code, e) for code, e in per_player.items() if len(e['points']) >= PLAYER_MIN_GAMES]
    pool.sort(key=lambda kv: -(sum(kv[1]['points']) / len(kv[1]['points'])))
    return pool[:PLAYER_POOL_SIZE]


def project_player_prop(stat_key, values):
    cfg = PLAYER_PROP_CONFIG[stat_key]
    if not values:
        return None
    n = len(values)
    recent = recency_weighted(values)
    shrunk = shrink(recent, n, cfg['prior'])
    return {
        'label': cfg['label'], 'dist': cfg['dist'], 'std': cfg['std'],
        'projected': round(shrunk, 1), 'n_games': n, 'recent_values': values,
        'stat_key': stat_key,
    }


def get_team_player_props(tv_code, games, season):
    pool = get_team_player_pool(tv_code, games, season)
    props = []
    for code, entry in pool:
        for stat_key in PLAYER_PROP_CONFIG:
            proj = project_player_prop(stat_key, entry.get(stat_key) or [])
            if proj:
                props.append({**proj, 'name': entry['name'], 'player_code': code})
    return props


def player_over_under_prob(proj, line):
    if proj['dist'] == 'poisson':
        p_under = poisson_cdf(math.floor(line), proj['projected'])
    else:
        p_under = norm_cdf(line, proj['projected'], proj['std'])
    return round((1 - p_under) * 100), round(p_under * 100)


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    return sum(poisson_pmf(i, lam) for i in range(int(k) + 1))


def normal_prop(mean, std, factor=0.86, round_to=0.5):
    # factor is higher than Blitz IQ's 0.72 -- EuroLeague team totals run
    # much lower-variance relative to the mean than NFL yardage stats, so
    # a line that far below the projection would almost always hit and
    # tell the user nothing; 0.86 keeps a safety margin without being
    # trivial. Same reasoning applies per-stat below.
    if mean is None or std is None:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line / round_to) * round_to
    if line < round_to:
        line = round_to
    prob_over = 1 - norm_cdf(line, mean, std)
    return {'line': line, 'prob': round(prob_over * 100), 'avg': round(mean, 1)}


def poisson_prop(mean, factor=0.75, round_to=0.5):
    if mean is None or mean <= 0:
        return None
    raw_line = mean * factor
    line = math.floor(raw_line / round_to) * round_to
    if line < round_to:
        line = round_to
    prob_over = 1 - poisson_cdf(math.floor(line), mean)
    return {'line': line, 'prob': round(prob_over * 100), 'avg': round(mean, 1)}


def hit_rate(values, line):
    if not values:
        return None
    hits = sum(1 for v in values if v > line)
    return {"hits": hits, "total": len(values)}


def format_history(lst):
    if not lst:
        return None
    return "/".join(str(v) for v in lst)


def get_upcoming_games(schedule, days_ahead=UPCOMING_WINDOW_DAYS):
    today = datetime.now(timezone.utc).date()
    upcoming = []
    for g in schedule:
        if g.get('played'):
            continue
        date_str = g.get('date')
        if not date_str:
            continue
        try:
            gd = datetime.fromisoformat(date_str[:10]).date()
        except ValueError:
            continue
        if today <= gd <= today + timedelta(days=days_ahead):
            upcoming.append(g)
    upcoming.sort(key=lambda g: g.get('date') or '')
    return upcoming


def build_predictions():
    season = current_season_year()
    code = season_code(season)
    schedule = get_schedule(code)
    if not schedule:
        print(f"  [!] empty schedule for season {code} -- nothing to predict")
        return []

    upcoming = get_upcoming_games(schedule)
    print(f"  {len(upcoming)} upcoming game(s) in the next {UPCOMING_WINDOW_DAYS} days")

    predictions = []
    all_forms = []
    team_forms_by_game = []
    for g in upcoming:
        local = (g.get('local') or {}).get('club') or {}
        road = (g.get('road') or {}).get('club') or {}
        h_code, a_code = local.get('tvCode'), road.get('tvCode')
        if not h_code or not a_code:
            continue
        h_form = get_team_form(h_code, schedule, code)
        a_form = get_team_form(a_code, schedule, code)
        if not h_form or not a_form:
            print(f"  skipping {road.get('name')} @ {local.get('name')}: missing form data "
                  f"(no played games yet this season for one side -- normal early in a new season)")
            continue
        all_forms.extend([h_form, a_form])
        team_forms_by_game.append((g, local, road, h_code, a_code, h_form, a_form))

    lg_scored, lg_allowed = league_averages(all_forms)

    for g, local, road, h_code, a_code, h_form, a_form in team_forms_by_game:
        proj = predict(h_form, a_form, lg_scored, lg_allowed)
        print(f"  Player props: {road.get('name')} @ {local.get('name')}")
        home_games = _team_games(schedule, h_code)[-RECENT_GAMES:]
        away_games = _team_games(schedule, a_code)[-RECENT_GAMES:]
        home_props = get_team_player_props(h_code, home_games, code)
        away_props = get_team_player_props(a_code, away_games, code)
        predictions.append({
            'date': g.get('date', ''),
            'season': code, 'game_code': g.get('gameCode'),
            'match': f"{road.get('name')} @ {local.get('name')}",
            'home_team': local.get('name'), 'away_team': road.get('name'),
            'home_tv_code': h_code, 'away_tv_code': a_code,
            'home_form': h_form, 'away_form': a_form,
            'home_props': home_props, 'away_props': away_props,
            **proj,
        })

    predictions.sort(key=lambda x: x['exp_total'], reverse=True)
    return predictions


def build_legs(predictions):
    legs = []
    for p in predictions:
        match_label = p["match"]
        hf, af = p["home_form"], p["away_form"]
        game_code, season = p.get("game_code"), p.get("season")
        game_date = (p.get("date") or "")[:10]

        home_total = normal_prop(p["exp_home"], DEFAULT_TEAM_STD)
        if home_total:
            legs.append({
                "match": match_label,
                "market": f"{p['home_team']} Over {home_total['line']} Points",
                "prob": home_total["prob"], "category": "Team Total",
                "hit_rate": hit_rate(hf.get("scored_list"), home_total["line"]),
                "detail": f"proj {home_total['avg']} pts ({hf['n_games']}gm)",
                "history": format_history(hf.get("scored_list")),
                "game_code": game_code, "season": season, "game_date": game_date,
                "is_home": True, "line": home_total["line"],
            })
        away_total = normal_prop(p["exp_away"], DEFAULT_TEAM_STD)
        if away_total:
            legs.append({
                "match": match_label,
                "market": f"{p['away_team']} Over {away_total['line']} Points",
                "prob": away_total["prob"], "category": "Team Total",
                "hit_rate": hit_rate(af.get("scored_list"), away_total["line"]),
                "detail": f"proj {away_total['avg']} pts ({af['n_games']}gm)",
                "history": format_history(af.get("scored_list")),
                "game_code": game_code, "season": season, "game_date": game_date,
                "is_home": False, "line": away_total["line"],
            })

        game_total = normal_prop(p["exp_total"], p["total_std"])
        if game_total:
            legs.append({
                "match": match_label,
                "market": f"Game Over {game_total['line']} Total Points",
                "prob": game_total["prob"], "category": "Game Total",
                "hit_rate": None,
                "detail": f"proj {game_total['avg']} pts ({hf['n_games']}v{af['n_games']}gm)",
                "history": None,
                "game_code": game_code, "season": season, "game_date": game_date,
                "line": game_total["line"],
            })

        for team_name, props in [(p["home_team"], p.get("home_props") or []),
                                   (p["away_team"], p.get("away_props") or [])]:
            for prop in props:
                if prop["dist"] == "poisson":
                    result = poisson_prop(prop["projected"])
                else:
                    result = normal_prop(prop["projected"], prop["std"])
                if not result:
                    continue
                legs.append({
                    "match": match_label,
                    "market": f"{prop['name']} Over {result['line']} {prop['label']}",
                    "prob": result["prob"], "category": prop["label"],
                    "hit_rate": hit_rate(prop.get("recent_values"), result["line"]),
                    "detail": f"proj {result['avg']} ({prop['n_games']}gm)",
                    "history": format_history(prop.get("recent_values")),
                    "game_code": game_code, "season": season, "game_date": game_date,
                    "line": result["line"], "player_code": prop.get("player_code"),
                    "stat_key": prop.get("stat_key"),
                    "is_home": team_name == p["home_team"],
                })
    return legs


BUILDER_TEMPLATE = """
<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:14px;font-weight:bold;margin-bottom:10px">🎯 Safest Bet Builder</div>
  <div id="categoryToggles" style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
    <label style="font-size:12px;color:#aaa">Target odds:</label>
    <input id="targetOdds" type="number" step="0.1" min="1.1" value="5.0"
      style="width:70px;background:#0f1318;border:1px solid #333;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <label style="font-size:12px;color:#aaa">Max legs:</label>
    <input id="maxLegs" type="number" step="1" min="2" value="8"
      style="width:55px;background:#0f1318;border:1px solid #333;color:white;border-radius:6px;padding:6px 8px;font-size:13px">
    <button onclick="buildSafest()"
      style="background:#3a7d7a;border:none;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      Build
    </button>
    <button onclick="buildSafest()"
      style="background:#2a3038;border:1px solid #444;color:white;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer">
      🔀 Shuffle
    </button>
  </div>
  <div id="builderResult" style="font-size:12px;color:#888">
    Untick any market type you don't want considered, set a target odds and
    leg cap, then tap Build. It rotates through whichever categories are
    ticked, groups near-tied legs and shuffles within each group so it draws
    from more of the week's games rather than always the exact same few, and
    caps at 2 legs per game to avoid stacking correlated legs from one
    matchup. Tap Shuffle for a fresh pick among equally-safe options without
    changing your settings.
  </div>
</div>
<script>
const LEGS = {legs_json};

function initCategoryToggles() {{
  const container = document.getElementById('categoryToggles');
  const cats = [...new Set(LEGS.map(l => l.category))];
  container.innerHTML = cats.map(c => `
    <label style="display:flex;align-items:center;gap:4px;color:#ccc;cursor:pointer">
      <input type="checkbox" class="catToggle" value="${{c}}" checked>
      ${{c}}
    </label>
  `).join('');
}}
initCategoryToggles();

function shuffle(arr) {{
  for (let i = arr.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }}
  return arr;
}}

function tieredShuffle(legs, bandSize) {{
  const bands = {{}};
  legs.forEach(l => {{
    const band = Math.floor(l.prob / bandSize);
    (bands[band] = bands[band] || []).push(l);
  }});
  const bandKeys = Object.keys(bands).map(Number).sort((a, b) => b - a);
  let result = [];
  bandKeys.forEach(b => {{ result = result.concat(shuffle(bands[b])); }});
  return result;
}}

function buildSafest() {{
  const target = parseFloat(document.getElementById('targetOdds').value) || 5.0;
  const maxLegs = parseInt(document.getElementById('maxLegs').value) || 8;
  const activeCats = [...document.querySelectorAll('.catToggle:checked')].map(el => el.value);

  const byCategory = {{}};
  LEGS.filter(l => l.prob > 0 && activeCats.includes(l.category)).forEach(l => {{
    (byCategory[l.category] = byCategory[l.category] || []).push(l);
  }});
  const categories = Object.keys(byCategory);
  categories.forEach(c => {{ byCategory[c] = tieredShuffle(byCategory[c], 5); }});
  const cursor = {{}};
  categories.forEach(c => cursor[c] = 0);

  const chosen = [];
  const matchCount = {{}};
  let combinedOdds = 1;
  let addedThisPass = true;

  while (addedThisPass && combinedOdds < target && chosen.length < maxLegs) {{
    addedThisPass = false;
    for (const cat of categories) {{
      if (combinedOdds >= target || chosen.length >= maxLegs) break;
      const arr = byCategory[cat];
      while (cursor[cat] < arr.length) {{
        const leg = arr[cursor[cat]];
        cursor[cat]++;
        const count = matchCount[leg.match] || 0;
        if (count >= 2) continue;
        chosen.push(leg);
        combinedOdds *= 100 / leg.prob;
        matchCount[leg.match] = count + 1;
        addedThisPass = true;
        break;
      }}
    }}
  }}

  const el = document.getElementById('builderResult');
  if (!chosen.length) {{
    el.innerHTML = 'No legs available to build from.';
    return;
  }}

  const rows = chosen.map(l =>
    `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #2a3038">
       <span>${{l.match}}<br><span style="color:#7ec8ff">${{l.market}}</span> <span style="color:#555">· ${{l.category}}</span>
       ${{l.detail ? `<br><span style="color:#666;font-size:10px">${{l.detail}}</span>` : ''}}
       ${{l.history ? `<br><span style="color:#555;font-size:10px">last games: ${{l.history}}</span>` : ''}}</span>
       <span style="text-align:right"><span style="color:#ffeb3b;font-weight:bold">${{l.prob}}%</span>${{l.hit_rate ? `<br><span style="color:#888;font-size:11px">${{l.hit_rate.hits}}/${{l.hit_rate.total}}</span>` : ''}}</span>
     </div>`
  ).join('');

  const capNote = chosen.length >= maxLegs && combinedOdds < target
    ? ' (hit the leg cap before reaching target — raise Max legs or lower Target odds)'
    : (combinedOdds < target ? ' (ran out of legs before reaching target)' : '');

  el.innerHTML = `
    <div style="color:white;font-size:13px;margin-bottom:6px">
      ${{chosen.length}} legs · est. combined odds ~<b>${{combinedOdds.toFixed(2)}}</b>${{capNote}}
    </div>
    ${{rows}}
    <div style="color:#666;font-size:10px;margin-top:8px;line-height:1.4">
      Estimate multiplies each leg's fair odds (100/probability) — real
      sportsbook odds include their margin and legs within the same game
      aren't fully independent, so treat this as a ranking tool, not a firm
      price. Team/game totals and Points use a Normal-distribution
      projection; Rebounds/Assists use Poisson. All lines are set
      automatically below the model's projection for a safety margin.
    </div>
  `;
}}
</script>"""


HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EuroLeague IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<h2 style="text-align:center">🏀 EUROLEAGUE IQ</h2>
<p style="text-align:center;color:#888;font-size:11px">Recency-weighted scoring/allowed rates, Normal-distribution projected · {generated}</p>
<p style="text-align:center;margin-bottom:16px"><a href="euroleague_iq_predictions.csv" download style="background:#222;border:1px solid #444;color:white;padding:8px 14px;border-radius:8px;text-decoration:none;font-size:13px">⬇ Download CSV</a></p>
<p style="text-align:center;margin-bottom:16px">
  <a href="results/index.html" style="color:#ffeb3b;text-decoration:none;font-size:12px">📊 Results Tracker</a>
</p>
{builder}
{cards}
<p style="text-align:center;color:#666;font-size:10px;margin-top:20px">Enter your book's Over/Under line and odds to compute an edge the same way as the other tools in this suite — this page shows the model's own projection only. Player props are built from whichever players actually featured in each team's last {recent_games} games (no fixed "starters" list), so a new or returning player may take a run or two to show up.</p>
</body></html>"""

CARD_TEMPLATE = """<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #2a3038">
  <div style="font-size:11px;color:#999;margin-bottom:4px">{date}</div>
  <div style="font-size:17px;font-weight:bold;margin-bottom:10px">{match}</div>
  <div style="display:flex;justify-content:space-between;text-align:center">
    <div><div style="color:#aaa;font-size:11px">{away_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_away}</div></div>
    <div><div style="color:#aaa;font-size:11px">TOTAL</div><div style="color:#7ec8ff;font-size:22px;font-weight:bold">{exp_total}</div></div>
    <div><div style="color:#aaa;font-size:11px">{home_team}</div><div style="color:#ffeb3b;font-size:20px;font-weight:bold">{exp_home}</div></div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:10px;display:flex;justify-content:space-between;font-size:11px">
    <div>{away_team}: {away_scored} scored/gm • {away_allowed} allowed/gm ({away_n}gm)</div>
  </div>
  <div style="background:#0f1318;border-radius:8px;padding:8px;margin-top:6px;font-size:11px">
    {home_team}: {home_scored} scored/gm • {home_allowed} allowed/gm ({home_n}gm)
  </div>
  {player_props_html}
</div>"""

PLAYER_PROP_ROW = """<div style="display:flex;justify-content:space-between;font-size:11px;padding:5px 0;border-top:1px solid #232a33">
  <div>{name} — {label}</div>
  <div style="color:#c792ea;font-weight:bold">{projected} <span style="color:#666;font-weight:normal">({n_games}gm)</span></div>
</div>"""


def player_props_section(team_label, props):
    if not props:
        return ""
    rows = "".join(PLAYER_PROP_ROW.format(**p) for p in props)
    return f'<div style="margin-top:8px"><div style="color:#888;font-size:10px;text-transform:uppercase;margin-bottom:2px">{team_label} Player Props</div>{rows}</div>'


def make_html(predictions):
    cards = "".join(CARD_TEMPLATE.format(
        date=p['date'][:16].replace('T', ' '), match=p['match'],
        away_team=p['away_team'], home_team=p['home_team'],
        exp_away=p['exp_away'], exp_home=p['exp_home'], exp_total=p['exp_total'],
        away_scored=p['away_form']['avg_scored'], away_allowed=p['away_form']['avg_allowed'],
        away_n=p['away_form']['n_games'],
        home_scored=p['home_form']['avg_scored'], home_allowed=p['home_form']['avg_allowed'],
        home_n=p['home_form']['n_games'],
        player_props_html=(
            player_props_section(p['away_team'], p.get('away_props', []))
            + player_props_section(p['home_team'], p.get('home_props', []))
        ),
    ) for p in predictions)
    if not cards:
        cards = '<p style="text-align:center;color:#666">No upcoming games with enough form data to project right now.</p>'

    legs = build_legs(predictions)
    builder = BUILDER_TEMPLATE.format(legs_json=json.dumps(legs)) if legs else ""

    return HTML_TEMPLATE.format(
        generated=datetime.now().strftime("%d %b %H:%M"),
        builder=builder, cards=cards, recent_games=RECENT_GAMES,
    )


def write_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Match', 'HomeTeam', 'AwayTeam',
            'HomeScoredAvg', 'HomeAllowedAvg', 'HomeSampleSize',
            'AwayScoredAvg', 'AwayAllowedAvg', 'AwaySampleSize',
            'ExpHome', 'ExpAway', 'ExpTotal', 'TotalStd',
            'Line', 'OverOdds', 'UnderOdds',
            'ActualHomeScore', 'ActualAwayScore', 'HitOrMiss',
        ])
        for p in predictions:
            writer.writerow([
                p['date'], p['match'], p['home_team'], p['away_team'],
                p['home_form']['avg_scored'], p['home_form']['avg_allowed'], p['home_form']['n_games'],
                p['away_form']['avg_scored'], p['away_form']['avg_allowed'], p['away_form']['n_games'],
                p['exp_home'], p['exp_away'], p['exp_total'], p['total_std'],
                '', '', '',
                '', '', '',
            ])


def write_player_props_csv(predictions, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Match', 'Team', 'Player', 'PropType',
            'Projected', 'SampleSize', 'RecentValues',
            'Line', 'OverOdds', 'UnderOdds', 'ActualValue', 'HitOrMiss',
        ])
        for p in predictions:
            for team_label, props in [(p['away_team'], p.get('away_props', [])),
                                       (p['home_team'], p.get('home_props', []))]:
                for prop in props:
                    writer.writerow([
                        p['date'], p['match'], team_label, prop['name'], prop['label'],
                        prop['projected'], prop['n_games'], '; '.join(str(v) for v in prop['recent_values']),
                        '', '', '', '', '',
                    ])


if __name__ == "__main__":
    predictions = build_predictions()
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html', 'w') as f:
        f.write(make_html(predictions))
    write_csv(predictions, 'docs/euroleague_iq_predictions.csv')
    write_player_props_csv(predictions, 'docs/euroleague_iq_player_props.csv')
    with open('docs/euroleague_iq.json', 'w') as f:
        json.dump(predictions, f, indent=2, default=str)

    try:
        import euroleague_iq_results_tracker
        euroleague_iq_results_tracker.run_results_tracker(build_legs(predictions))
    except Exception as e:
        print(f"[!] Results tracker failed, but the rest of this run succeeded: {e}")
