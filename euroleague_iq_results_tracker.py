#!/usr/bin/env python3
"""
EuroLeague IQ Results Tracker
--------------------------------------------------------------
Same architecture as the other trackers in this suite (Blitz IQ's
blitz_iq_results_tracker.py especially), adapted for EuroLeague's own v2
API. Simpler than most of its siblings on one front: EuroLeague's game
stats endpoint uses real, confirmed field names (points/totalRebounds/
assistances, player.person.code) rather than ESPN's label-matched
gamelog/boxscore arrays, so there's no [DIAG]-guarded label-guessing
needed for the stat extraction itself -- just the usual "is this game
actually finished yet" guard.

A game is identified by (season, game_code) rather than one single global
ID -- every leg carries both, same way Blitz IQ's legs carry game_id.

Designed to be imported and called from euroleague_iq.py's main().

Output:
    docs/results/log.json    -- the full log
    docs/results/index.html  -- dashboard: overall + per-category
                                 win rate, recent history
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timezone

BASE = "https://api-live.euroleague.net/v2/competitions/E"
LOG_PATH = "docs/results/log.json"
DASHBOARD_PATH = "docs/results/index.html"

STATS_CACHE = {}  # (season, game_code) -> box score, several legs share a game

CATEGORY_SCANNER = {
    "Team Total": "team_total", "Game Total": "game_total",
    "Points": "points", "Rebounds": "rebounds", "Assists": "assists",
}


def _get_game_stats(season, game_code):
    key = (season, game_code)
    if key in STATS_CACHE:
        return STATS_CACHE[key]
    try:
        r = requests.get(f"{BASE}/seasons/{season}/games/{game_code}/stats", timeout=20)
    except Exception as e:
        print(f"    [!] verification request failed: season={season} game={game_code} ({e})")
        STATS_CACHE[key] = None
        return None
    if r.status_code != 200:
        print(f"    [!] {r.status_code} on game stats for season={season} game={game_code}")
        STATS_CACHE[key] = None
        return None
    try:
        data = r.json()
    except Exception as e:
        print(f"    [!] couldn't parse game stats for season={season} game={game_code}: {e}")
        data = None
    STATS_CACHE[key] = data
    return data


def _is_final(stats):
    """No separate 'is this game finished' flag on this endpoint (unlike
    ESPN's status.type.completed) -- a game that hasn't been played yet
    simply has no total.points on either side, so that doubles as the
    finished check."""
    if not stats:
        return False
    local_pts = ((stats.get('local') or {}).get('total') or {}).get('points')
    road_pts = ((stats.get('road') or {}).get('total') or {}).get('points')
    return local_pts is not None and road_pts is not None


def _entry_id(scanner, market, date_key, match):
    raw = f"{scanner}|{market}|{date_key}|{match}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [!] Couldn't read existing results log ({e}) -- starting fresh.")
        return []


def save_log(entries):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_todays_signals(legs, streak_entries, real_streak_entries, log):
    """Legs already carry season, game_code, is_home, line, and (for
    player props) player_code/stat_key -- added specifically for this
    tracker when the legs were built. Hot Form / Real Streak entries carry
    season, game_code, is_home, and the league-average threshold captured
    at flag time ('threshold') -- verified by checking whether the team's
    ACTUAL points in that same predicted match beat it, same "did the form
    continue" check Euro Ice's own Hot Form/Real Streak panels use."""
    existing_ids = {e["id"] for e in log}
    added = 0

    def add(scanner, match, market, value, detail, season, game_code, date_key,
             is_home=None, line=None, player_code=None, stat_key=None, threshold=None):
        nonlocal added
        eid = _entry_id(scanner, market, date_key, match)
        if eid in existing_ids:
            return
        log.append({
            "id": eid, "scanner": scanner, "match": match, "market": market,
            "value": value, "detail": detail, "line": line, "threshold": threshold,
            "season": season, "game_code": game_code, "game_date": date_key,
            "date_key": date_key, "is_home": is_home,
            "player_code": player_code, "stat_key": stat_key,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "result": None, "actual": None,
        })
        existing_ids.add(eid)
        added += 1

    for leg in legs:
        scanner = CATEGORY_SCANNER.get(leg.get("category"))
        if not scanner or leg.get("game_code") is None or not leg.get("season"):
            continue
        add(scanner, leg["match"], leg["market"], leg["prob"], leg.get("detail"),
            leg["season"], leg["game_code"], leg["game_date"],
            is_home=leg.get("is_home"), line=leg.get("line"),
            player_code=leg.get("player_code"), stat_key=leg.get("stat_key"))

    for e in streak_entries:
        if e.get("game_code") is None or not e.get("season"):
            continue
        match = f"{e['team']} vs {e['opponent']}"
        add("hot_form", match, f"Hot Form vs {e['opponent']}", e["last5_avg"],
            f"L5 avg {e['last5_avg']} (league avg {e['lg_scored']})",
            e["season"], e["game_code"], (e.get("date") or "")[:10],
            is_home=e["is_home"], threshold=e["threshold"])

    for e in real_streak_entries:
        if e.get("game_code") is None or not e.get("season"):
            continue
        match = f"{e['team']} vs {e['opponent']}"
        add("real_streak", match, f"Real Streak vs {e['opponent']}", e["streak_len"],
            f"{e['streak_len']} straight above league avg ({e['threshold']}+)",
            e["season"], e["game_code"], (e.get("date") or "")[:10],
            is_home=e["is_home"], threshold=e["threshold"])

    print(f"  Results log: {added} new pick(s) logged, {len(log)} total in log")
    return log


def _verify_team_leg(entry):
    stats = _get_game_stats(entry["season"], entry["game_code"])
    if not _is_final(stats):
        return None
    local_pts = ((stats.get('local') or {}).get('total') or {}).get('points')
    road_pts = ((stats.get('road') or {}).get('total') or {}).get('points')
    if entry["scanner"] == "game_total":
        actual = float(local_pts) + float(road_pts)
    else:
        actual = float(local_pts) if entry.get("is_home") else float(road_pts)
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _find_player_stat(stats, player_code, stat_key):
    for side in ("local", "road"):
        team_block = stats.get(side) or {}
        for p in team_block.get("players", []):
            person = ((p.get("player") or {}).get("person")) or {}
            if str(person.get("code")) != str(player_code):
                continue
            val = (p.get("stats") or {}).get(stat_key)
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    print(f"    [DIAG] couldn't find player {player_code}'s '{stat_key}' stat in either "
          f"team's box score -- either they didn't feature this game, or the shape here "
          f"doesn't match what was confirmed against the open-source client's fixtures.")
    return None


def _verify_player_leg(entry):
    if not entry.get("player_code") or not entry.get("stat_key"):
        return None
    stats = _get_game_stats(entry["season"], entry["game_code"])
    if not _is_final(stats):
        return None
    actual = _find_player_stat(stats, entry["player_code"], entry["stat_key"])
    if actual is None:
        return None
    return {"actual": actual, "result": "hit" if actual > entry["line"] else "miss"}


def _verify_form_streak_entry(entry):
    """Hot Form / Real Streak are raw-form screens, not probabilistic
    predictions -- 'hit' means the flagged team's actual points in THIS
    SAME predicted match beat the league-average threshold captured when
    the entry was flagged, i.e. the form continued."""
    stats = _get_game_stats(entry["season"], entry["game_code"])
    if not _is_final(stats):
        return None
    local_pts = ((stats.get('local') or {}).get('total') or {}).get('points')
    road_pts = ((stats.get('road') or {}).get('total') or {}).get('points')
    if local_pts is None or road_pts is None:
        return None
    actual = float(local_pts) if entry.get("is_home") else float(road_pts)
    return {"actual": actual, "result": "hit" if actual > entry["threshold"] else "miss"}


def verify_pending_results(log, max_checks=200):
    today = datetime.now(timezone.utc).date().isoformat()
    checked = 0
    updated = 0

    for entry in log:
        if entry["status"] != "pending":
            continue
        if entry["date_key"] >= today:
            continue
        if checked >= max_checks:
            break
        checked += 1

        result = None
        try:
            if entry["scanner"] in ("team_total", "game_total"):
                result = _verify_team_leg(entry)
            elif entry["scanner"] in ("points", "rebounds", "assists"):
                result = _verify_player_leg(entry)
            elif entry["scanner"] in ("hot_form", "real_streak"):
                result = _verify_form_streak_entry(entry)
        except Exception as e:
            print(f"    [!] verification error for entry {entry['id']} ({entry['scanner']}): {e}")
            result = None

        if result:
            entry["status"] = "verified"
            entry["result"] = result["result"]
            entry["actual"] = result["actual"]
            entry["verified_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1

    print(f"  Results verification: checked {checked} pending entries, {updated} newly verified "
          f"({len(STATS_CACHE)} distinct game(s) looked up)")
    return log


def build_results_dashboard(log):
    verified = [e for e in log if e["status"] == "verified"]
    pending = [e for e in log if e["status"] == "pending"]

    by_scanner = {}
    for e in verified:
        d = by_scanner.setdefault(e["scanner"], {"hit": 0, "miss": 0})
        d[e["result"]] += 1

    SCANNER_LABELS = {
        "team_total": "Team Total", "game_total": "Game Total",
        "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
        "hot_form": "Hot Form", "real_streak": "Real Streak",
    }

    total_hit = sum(d["hit"] for d in by_scanner.values())
    total_miss = sum(d["miss"] for d in by_scanner.values())
    total = total_hit + total_miss
    overall_pct = round(100 * total_hit / total) if total else None

    rows = ""
    for scanner, label in SCANNER_LABELS.items():
        d = by_scanner.get(scanner, {"hit": 0, "miss": 0})
        n = d["hit"] + d["miss"]
        pct = round(100 * d["hit"] / n) if n else None
        pct_str = f"{pct}%" if pct is not None else "—"
        rows += f"""<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #2a3038">
  <span>{label}</span><span style="color:#ffeb3b;font-weight:bold">{pct_str}</span>
  <span style="color:#888;font-size:12px">{d['hit']}/{n}</span>
</div>"""

    recent = sorted(verified, key=lambda e: e.get("verified_at", ""), reverse=True)[:30]
    recent_rows = ""
    for e in recent:
        color = "#22c55e" if e["result"] == "hit" else "#ef4444"
        recent_rows += f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #2a3038;font-size:12px">
  <span>{e['match']} — {e['market']}</span><span style="color:{color};font-weight:bold">{e['result'].upper()}</span>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Results — EuroLeague IQ</title></head>
<body style="background:#0b0f14;color:white;font-family:Arial;padding:12px;max-width:600px;margin:auto">
<p style="text-align:center;margin-bottom:6px"><a href="../index.html" style="color:#7ec8ff;text-decoration:none;font-size:12px">← EuroLeague IQ</a></p>
<h2 style="text-align:center;margin-bottom:2px">📊 Results Tracker</h2>
<p style="text-align:center;color:#888;font-size:11px;margin-top:0">{datetime.now().strftime("%d %b %H:%M")} · every pick, auto-verified against real results</p>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038;text-align:center">
  <div style="font-size:11px;color:#888">OVERALL</div>
  <div style="font-size:32px;font-weight:bold;color:#ffeb3b">{overall_pct if overall_pct is not None else "—"}{"%" if overall_pct is not None else ""}</div>
  <div style="font-size:12px;color:#888">{total_hit}/{total} verified picks · {len(pending)} pending (game not final yet)</div>
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">By Category</div>
  {rows}
</div>

<div style="background:#1a1f26;border-radius:12px;padding:16px;margin:14px 0;border:1px solid #2a3038">
  <div style="font-weight:bold;margin-bottom:8px">Recent Results</div>
  {recent_rows or '<p style="color:#888;font-size:12px">Nothing verified yet — check back after a few days of picks have had time to play out.</p>'}
</div>

<div style="font-size:11px;color:#888;text-align:center;margin-top:20px;line-height:1.6">
  All categories use EuroLeague's own confirmed game-stats field names
  (points/totalRebounds/assistances) rather than a guessed label match --
  if a category stays empty for more than a few days after games finish,
  check the Actions log for [!]/[DIAG] lines. Sample sizes are small
  early on — treat percentages with real caution until there's a few
  weeks of data.
</div>
</body></html>"""

    os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)
    print(f"  Results dashboard: {total} verified, {overall_pct}% overall" if total else "  Results dashboard: no verified picks yet")


def run_results_tracker(legs, streak_entries=None, real_streak_entries=None, lg_scored=None):
    """Single entry point called from euroleague_iq.py's main()."""
    print("\nRunning results tracker...")
    log = load_log()
    log = log_todays_signals(legs, streak_entries or [], real_streak_entries or [], log)
    log = verify_pending_results(log)
    save_log(log)
    build_results_dashboard(log)
