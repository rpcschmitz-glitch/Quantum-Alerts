#!/usr/bin/env python3
"""
Quantum Trader V3.02 (Wysetrade) -> pushmeldingen op je iPhone via ntfy.

Port van de Pine-indicator (alleen LONG, geen orders, alleen meldingen).
Data: publieke Kraken-API (geen account of API-sleutel nodig).
Timeframes zoals in het script: 4H = context, 1H = setup, 15m = signaal.

Gebruik:
    pip install pandas numpy requests
    export NTFY_TOPIC="jouw-geheime-topicnaam"   # zelfde naam als in de ntfy-app
    python quantum_alerts.py --test              # stuurt een testmelding
    python quantum_alerts.py --history           # toont signalen van de afgelopen dagen (geen push)
    python quantum_alerts.py                     # draait continu, check na elke 15m-close
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------
# INSTELLINGEN
# ----------------------------------------------------------------------------
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "quantum-trader-VERANDER-MIJ")
QUOTE = "EUR"  # "EUR" past bij Revolut X; "USD" heeft meestal meer volume op Kraken
COINS = [  # Kraken-symbolen (BTC heet daar XBT). Ontbrekende paren worden overgeslagen.
    "XBT", "ETH", "XRP", "SOL", "ADA", "DOGE", "DOT", "LINK", "LTC", "AVAX",
    "BCH", "XLM", "ATOM", "UNI", "NEAR", "AAVE", "POL", "ETC", "ALGO", "TRX",
    "SUI", "HBAR",
]
SEND_EARLY = False        # EARLY-signalen ook pushen (alleen 'let op', geen entry)
SEND_TP_UPDATES = True    # melding als TP1 / TP2 geraakt wordt
REQUEST_GAP = 0.7         # seconden tussen Kraken-requests (rate limit)
STATE_FILE = Path(__file__).with_name("sent_alerts.json")

# Zelfde defaults als de inputs in de Pine-code.
P = dict(
    earlyThreshold=2, buyThreshold=3, strongThreshold=4,
    swingBars=3, swingMaxAge=120, locationAtr=0.75, overbought=75.0, volumeMin=0.9,
    enableBreakout=True, breakoutLookback=12, breakoutBodyAtr=0.75,
    breakoutVolumeMin=1.0, breakoutClosePct=0.70, breakoutRsiMax=88.0,
    breakoutMaxAtr=3.0, breakoutCooldownBars=8,
    minStopAtr=1.5, stopBuffer=0.25, tp1R=2.0, tp2R=3.0, exitBars=2,
)

API = "https://api.kraken.com/0/public"
NAN = float("nan")
isn = math.isnan


# ----------------------------------------------------------------------------
# TA-hulpfuncties (zelfde rekenregels als Pine)
# ----------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    a = s.to_numpy(float)
    out = np.full(len(a), np.nan)
    valid = np.flatnonzero(~np.isnan(a))
    if len(valid) == 0:
        return pd.Series(out, index=s.index)
    k = valid[0]
    if k + n > len(a):
        return pd.Series(out, index=s.index)
    out[k + n - 1] = a[k:k + n].mean()
    for i in range(k + n, len(a)):
        out[i] = (out[i - 1] * (n - 1) + a[i]) / n
    return pd.Series(out, index=s.index)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def rsi(close, n=14):
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    r = 100 - 100 / (1 + up / dn)
    r[up == 0] = 0.0
    r[dn == 0] = 100.0
    return r


def pivots(high, low, left):
    """Bevestigd op bar i, prijs van bar i-left (zoals ta.pivothigh/pivotlow)."""
    h, l = high.to_numpy(float), low.to_numpy(float)
    n = len(h)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(2 * left, n):
        c = i - left
        if h[c] >= h[c - left:i + 1].max():
            ph[i] = h[c]
        if l[c] <= l[c - left:i + 1].min():
            pl[i] = l[c]
    return ph, pl


# ----------------------------------------------------------------------------
# HTF-berekeningen (f_context op 4H, f_setup op 1H). Alles met [1]-verschuiving.
# ----------------------------------------------------------------------------
def htf_context(df, P):
    c = df["close"]
    e50, e200 = ema(c, 50), ema(c, 200)
    ph, pl = pivots(df["high"], df["low"], P["swingBars"])
    ph_s, pl_s = pd.Series(ph, index=df.index), pd.Series(pl, index=df.index)

    def last_two(s):
        nn = s.dropna()
        last = s.ffill()
        prev = nn.shift(1).reindex(s.index).ffill()
        return last, prev

    lastHigh, prevHigh = last_two(ph_s)
    lastLow, prevLow = last_two(pl_s)
    structure = (lastHigh > prevHigh) & (lastLow > prevLow) & (c > lastLow)
    clear = ((e50 > e200) & (c > e50)) | structure
    acceptable = clear | ((c > e50) & (e50 > e50.shift(1))) | ((c > e200) & (e200 >= e200.shift(1)))
    return pd.DataFrame({
        "ctxOk": acceptable.shift(1, fill_value=False).astype(float),
        "trendPt": clear.shift(1, fill_value=False).astype(float),
        "ctxE200": e200.shift(1),
    })


def htf_setup(df, P):
    c, h, l = df["close"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float)
    e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
    a, r = atr(df, 14).to_numpy(float), rsi(df["close"], 14)
    L, maxAge = P["swingBars"], P["swingMaxAge"]
    ph, pl = pivots(df["high"], df["low"], L)
    n = len(df)

    lastLow, lowBar = NAN, None
    impLow, impHigh, impBar, broken = NAN, NAN, None, True
    valid = np.zeros(n, bool)
    fib50, fib618 = np.full(n, np.nan), np.full(n, np.nan)
    support = np.full(n, np.nan)
    impL, impH = np.full(n, np.nan), np.full(n, np.nan)

    for i in range(n):
        if not isn(pl[i]):
            lastLow, lowBar = pl[i], i - L
        if not isn(ph[i]) and lowBar is not None:
            highBar = i - L
            if highBar > lowBar and highBar - lowBar <= maxAge and ph[i] > lastLow:
                impLow, impHigh, impBar, broken = lastLow, ph[i], highBar, False
        if not isn(impLow) and c[i] < impLow:
            broken = True
        v = (not broken) and impBar is not None and (i - impBar) <= maxAge \
            and c[i] >= impLow and c[i] <= impHigh + a[i]
        valid[i] = v
        if v:
            fib50[i] = impHigh - (impHigh - impLow) * 0.5
            fib618[i] = impHigh - (impHigh - impLow) * 0.618
        if lowBar is not None and (i - lowBar) <= maxAge:
            support[i] = lastLow
        impL[i], impH[i] = impLow, impHigh

    vprev = pd.Series(valid).shift(1, fill_value=False).to_numpy()
    s = lambda x: pd.Series(x, index=df.index).shift(1)
    return pd.DataFrame({
        "c1": s(c), "l1": s(l), "h1": s(h), "ema50": s(e50.to_numpy()), "ema200": s(e200.to_numpy()),
        "atr1": s(a), "rsi1": s(r.to_numpy()), "prsi1": pd.Series(r.to_numpy(), index=df.index).shift(2),
        "support1": s(support), "fib50": s(fib50), "fib618": s(fib618),
        "swingLow": np.where(vprev, s(impL), np.nan), "swingHigh": np.where(vprev, s(impH), np.nan),
    })


def htf_map(t15, htf_time, frame):
    """Elke 15m-bar krijgt de waarden van de HTF-bar waarin hij valt (lookahead_on)."""
    idx = np.searchsorted(htf_time, t15, side="right") - 1
    ok = idx >= 0
    out = {}
    for col in frame.columns:
        v = frame[col].to_numpy(float)
        out[col] = np.where(ok, v[np.clip(idx, 0, None)], np.nan)
    return out


# ----------------------------------------------------------------------------
# 15m-engine: score, breakout-override, risk en trade-status (bar voor bar)
# ----------------------------------------------------------------------------
def run_engine(df15, M, tick, P):
    t = df15["time"].to_numpy()
    o, h, l, c = (df15[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = df15["volume"].to_numpy(float)
    n = len(df15)
    rsi15 = rsi(df15["close"], 14).to_numpy()
    ema50_15 = ema(df15["close"], 50).to_numpy()
    atr15 = atr(df15, 14).to_numpy()
    volAvg = df15["volume"].rolling(20).mean().shift(1).to_numpy()
    recentLow = df15["low"].rolling(8).min().to_numpy()
    priorHigh = df15["high"].rolling(P["breakoutLookback"]).max().shift(1).to_numpy()

    ctxOk, trendPt, ctxE200 = M["ctxOk"], M["trendPt"], M["ctxE200"]
    c1, l1, h1, e50, e200 = M["c1"], M["l1"], M["h1"], M["ema50"], M["ema200"]
    atr1, rsi1, prsi1, sup1, fib50, fib618 = M["atr1"], M["rsi1"], M["prsi1"], M["support1"], M["fib50"], M["fib618"]

    def near(lo, hi, cc, level, tol):
        return (not isn(level)) and lo <= level + tol and hi >= level - tol and cc >= level - tol and cc <= level + tol * 2

    def rnd(x):
        return round(x / tick) * tick

    active = tp1Seen = False
    entryBar = endedBar = lastBO = None
    highestTier, weakBars = 0, 0
    entry = stop = t1 = t2 = entryStruct = entryAtr = NAN
    events = []

    for i in range(1, n):
        warm = (not isn(ctxE200[i]) and not isn(e200[i]) and not isn(atr1[i]) and atr1[i] > 0
                and not isn(atr15[i]) and atr15[i] > 0 and not isn(rsi15[i]))
        tol = atr1[i] * P["locationAtr"]

        nearFib = (not isn(fib50[i]) and l1[i] <= fib50[i] + tol and h1[i] >= fib618[i] - tol
                   and c1[i] >= fib618[i] - tol and c1[i] <= fib50[i] + tol)
        nearSup = near(l1[i], h1[i], c1[i], sup1[i], tol)
        nearEma = near(l1[i], h1[i], c1[i], e50[i], tol) or near(l1[i], h1[i], c1[i], e200[i], tol)
        locationOk = nearFib or nearSup or nearEma
        exactFib = (not isn(fib50[i]) and l1[i] <= fib50[i] and h1[i] >= fib618[i]
                    and c1[i] >= fib618[i] - atr1[i] * 0.25 and c1[i] <= fib50[i] + atr1[i] * 0.5)
        tight = atr1[i] * 0.35
        locationPt = (exactFib or near(l1[i], h1[i], c1[i], sup1[i], tight)
                      or near(l1[i], h1[i], c1[i], e50[i], tight) or near(l1[i], h1[i], c1[i], e200[i], tight))

        momentumOk = (not (rsi1[i] < 40 and rsi1[i] < prsi1[i])) and rsi15[i] >= 35 and rsi15[i] < P["overbought"]
        momentumPt = rsi15[i] > rsi15[i - 1] and (rsi1[i] >= 45 or rsi1[i] > prsi1[i])
        rv = v[i] / volAvg[i] if (not isn(volAvg[i]) and volAvg[i] > 0) else NAN
        volumePt = (not isn(rv)) and (rv >= P["volumeMin"] or (rv >= 0.7 and v[i] > v[i - 1]))
        trendPt_b = trendPt[i] == 1.0
        score = int(trendPt_b) + int(locationPt) + int(momentumPt) + int(volumePt)

        turn15 = c[i] > o[i] and c[i] > c[i - 1] and (c[i - 1] <= o[i - 1] or c[i] > h[i - 1])
        notChasing = c[i] <= c1[i] + atr1[i] and c[i] >= c1[i] - tol
        setupOk = warm and ctxOk[i] == 1.0 and locationOk
        candidate = setupOk and momentumOk and notChasing and turn15
        tier = 3 if score >= P["strongThreshold"] else 2 if score >= P["buyThreshold"] else 1 if score >= P["earlyThreshold"] else 0

        # Strong-breakout override
        rng = h[i] - l[i]
        body = max(c[i] - o[i], 0.0)
        closePos = (c[i] - l[i]) / rng if rng > 0 else 0.0
        boPrice = (not isn(priorHigh[i])) and c[i] > priorHigh[i] and c[i] > h[i - 1]
        boBody = body >= atr15[i] * P["breakoutBodyAtr"] and closePos >= P["breakoutClosePct"]
        boVol = (not isn(rv)) and rv >= P["breakoutVolumeMin"]
        boRsi = rsi15[i] >= 50 and rsi15[i] < P["breakoutRsiMax"]
        boDist = (not isn(c1[i])) and c[i] > c1[i] and c[i] <= c1[i] + atr1[i] * P["breakoutMaxAtr"]
        breakoutSetup = (P["enableBreakout"] and warm and ctxOk[i] == 1.0 and trendPt_b
                         and boPrice and boBody and boVol and boRsi and boDist)

        # Risk
        structureFloor = recentLow[i]
        if not isn(sup1[i]) and sup1[i] < c[i] and c[i] - sup1[i] <= atr1[i] * 3:
            structureFloor = min(structureFloor, sup1[i]) if not isn(structureFloor) else sup1[i]
        proposedStop = NAN
        if not isn(structureFloor) and not isn(atr1[i]):
            raw = min(structureFloor - atr1[i] * P["stopBuffer"], c[i] - atr1[i] * P["minStopAtr"])
            proposedStop = math.floor(raw / tick + 1e-9) * tick
        validRisk = (not isn(proposedStop)) and proposedStop > 0 and proposedStop <= c[i] - tick * 2

        ended = False
        exitSig = False
        stopTouched = brokenStructure = False

        cooldownOk = lastBO is None or i - lastBO >= P["breakoutCooldownBars"]
        breakoutCand = breakoutSetup and cooldownOk
        freshPullback = endedBar is not None and i > endedBar and c[i] < o[i] and c[i] < c[i - 1]
        if (not active) and ((not setupOk) or (not momentumOk) or (not notChasing) or (highestTier == 3 and freshPullback)):
            highestTier = 0

        if active and i > entryBar:
            weak = rsi15[i] < 42 and c[i] < ema50_15[i]
            weakBars = weakBars + 1 if weak else 0
            brokenStructure = c[i] < entryStruct - entryAtr * P["stopBuffer"] and rsi15[i] < 45
            stopTouched = l[i] <= stop
            exitSig = stopTouched or brokenStructure or weakBars >= P["exitBars"]
            if h[i] >= t1 and not tp1Seen:
                events.append(dict(i=i, time=int(t[i]), kind="TP1", entry=entry, stop=stop, tp1=t1, tp2=t2, price=c[i]))
            tp1Seen = tp1Seen or h[i] >= t1
            if exitSig or h[i] >= t2:
                active, ended, endedBar = False, True, i
                highestTier = 3 if setupOk else 0
                if exitSig:
                    reason = "SL geraakt (indicatief)" if stopTouched else (
                        "structuur gebroken" if brokenStructure else "momentum blijft zwak")
                    events.append(dict(i=i, time=int(t[i]), kind="EXIT", entry=entry, stop=stop, tp1=t1, tp2=t2,
                                       price=c[i], reason=reason))
                else:
                    events.append(dict(i=i, time=int(t[i]), kind="TP2", entry=entry, stop=stop, tp1=t1, tp2=t2, price=c[i]))

        signalTier = max(tier, 2) if breakoutCand else tier
        signalCand = candidate or breakoutCand
        overrideUsed = breakoutCand and tier < 2
        if (not ended) and signalCand and signalTier > highestTier and (signalTier == 1 or active or validRisk):
            early = signalTier == 1 and not active
            buy = signalTier == 2
            strong = signalTier == 3
            highestTier = signalTier
            if breakoutCand and (buy or strong):
                lastBO = i
            if early:
                events.append(dict(i=i, time=int(t[i]), kind="EARLY", price=c[i], score=score))
            if buy or strong:
                upgrade = active
                if not active:
                    active, tp1Seen, entryBar = True, False, i
                    entry = rnd(c[i])
                    stop = proposedStop
                    risk = entry - stop
                    t1 = rnd(entry + risk * P["tp1R"])
                    t2 = max(t1 + tick, rnd(entry + risk * P["tp2R"]))
                    entryStruct, entryAtr, weakBars = structureFloor, atr1[i], 0
                kind = "BREAKOUT BUY" if (overrideUsed and buy) else ("STRONG BUY" if strong else "BUY")
                events.append(dict(i=i, time=int(t[i]), kind=kind, entry=entry, stop=stop, tp1=t1, tp2=t2,
                                   price=c[i], score=score, upgrade=upgrade))
    return events


# ----------------------------------------------------------------------------
# Kraken-data
# ----------------------------------------------------------------------------
def kraken(path, **params):
    err = None
    for attempt in range(3):
        try:
            j = requests.get(f"{API}/{path}", params=params, timeout=20).json()
            if not j.get("error"):
                return j["result"]
            err = j["error"]
        except Exception as e:  # netwerkfout, ongeldige JSON, ...
            err = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Kraken {path} mislukt: {err}")


def load_pairs():
    res = kraken("AssetPairs")
    by_ws = {v.get("wsname"): v for v in res.values()}
    pairs = []
    for coin in COINS:
        v = by_ws.get(f"{coin}/{QUOTE}")
        if not v:
            print(f"  (overgeslagen: {coin}/{QUOTE} niet op Kraken)")
            continue
        dec = int(v["pair_decimals"])
        pairs.append(dict(name=f"{'BTC' if coin == 'XBT' else coin}/{QUOTE}", alt=v["altname"],
                          dec=dec, tick=float(v.get("tick_size", 10 ** -dec))))
    return pairs


def ohlc(alt, interval):
    res = kraken("OHLC", pair=alt, interval=interval)
    key = [k for k in res if k != "last"][0]
    df = pd.DataFrame(res[key], columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    return df


def analyze(pair, fetch=ohlc):
    d15 = fetch(pair["alt"], 15)
    time.sleep(REQUEST_GAP)
    d60 = fetch(pair["alt"], 60)
    time.sleep(REQUEST_GAP)
    d240 = fetch(pair["alt"], 240)
    time.sleep(REQUEST_GAP)
    if time.time() < d15["time"].iloc[-1] + 900:  # laatste 15m-candle is nog niet gesloten
        d15 = d15.iloc[:-1].reset_index(drop=True)
    ctx = htf_context(d240, P)
    st = htf_setup(d60, P)
    t15 = d15["time"].to_numpy()
    M = {}
    M.update(htf_map(t15, d240["time"].to_numpy(), ctx))
    M.update(htf_map(t15, d60["time"].to_numpy(), st))
    return run_engine(d15, M, pair["tick"], P), int(t15[-1])


# ----------------------------------------------------------------------------
# Meldingen
# ----------------------------------------------------------------------------
def build_message(pair, ev):
    dec = pair["dec"]
    f = lambda x: f"{x:.{dec}f}"
    pct = lambda a, b: f"{(a / b - 1) * 100:+.1f}%"
    k, name = ev["kind"], pair["name"]
    if k in ("BUY", "STRONG BUY", "BREAKOUT BUY"):
        e, s, t1, t2 = ev["entry"], ev["stop"], ev["tp1"], ev["tp2"]
        head = f"{k} {name}" + (" (upgrade)" if ev.get("upgrade") else "")
        body = (f"Entry: {f(e)}\nSL: {f(s)} ({pct(s, e)})\nTP1: {f(t1)} ({pct(t1, e)}, {P['tp1R']:g}R)\n"
                f"TP2: {f(t2)} ({pct(t2, e)}, {P['tp2R']:g}R)\nScore {ev['score']}/4 - 15m close")
        if ev.get("upgrade"):
            body += "\nBestaande trade: entry/SL ongewijzigd."
        return head, body, "high", "chart_with_upwards_trend"
    if k == "EARLY":
        return f"EARLY {name}", f"Setup in opbouw (score {ev['score']}/4). Alleen kijken, nog geen entry.", "default", "eyes"
    if k == "TP1":
        return f"TP1 geraakt {name}", f"TP1 {f(ev['tp1'])} bereikt. Entry was {f(ev['entry'])}, SL {f(ev['stop'])}.", "high", "white_check_mark"
    if k == "TP2":
        return f"TP2 geraakt {name}", f"TP2 {f(ev['tp2'])} bereikt. Trade afgerond.", "high", "tada"
    return (f"EXIT {name}", f"{ev['reason']}. Koers {f(ev['price'])}. Entry was {f(ev['entry'])}, "
            f"SL {f(ev['stop'])}. Check handmatig.", "urgent", "warning")


def push(title, body, priority, tags):
    r = requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=body.encode("utf-8"),
                      headers={"Title": title, "Priority": priority, "Tags": tags}, timeout=15)
    r.raise_for_status()


def load_sent():
    try:
        return set(json.loads(STATE_FILE.read_text()))
    except Exception:
        return set()


def save_sent(sent):
    STATE_FILE.write_text(json.dumps(sorted(sent)[-1000:]))


def want(ev):
    if ev["kind"] == "EARLY":
        return SEND_EARLY
    if ev["kind"] in ("TP1", "TP2"):
        return SEND_TP_UPDATES
    return True


def cycle(pairs, dry=False):
    sent = load_sent()
    for p in pairs:
        try:
            events, last_t = analyze(p)
        except Exception as e:
            print(f"[{p['name']}] fout: {e}")
            continue
        for ev in events:
            if ev["time"] < last_t - 900 or not want(ev):  # alleen de laatste 2 gesloten bars
                continue
            key = f"{p['name']}|{ev['time']}|{ev['kind']}"
            if key in sent:
                continue
            title, body, prio, tags = build_message(p, ev)
            print(f"[{datetime.now():%H:%M}] {title}\n{body}\n")
            if not dry:
                try:
                    push(title, body, prio, tags)
                    sent.add(key)
                except Exception as e:
                    print(f"  push mislukt: {e}")
    save_sent(sent)


def history(pairs):
    for p in pairs:
        try:
            events, _ = analyze(p)
        except Exception as e:
            print(f"[{p['name']}] fout: {e}")
            continue
        for ev in events:
            close = datetime.fromtimestamp(ev["time"] + 900)
            _, body, _, _ = build_message(p, ev)
            print(f"{close:%d-%m %H:%M}  {p['name']:<10} {ev['kind']:<13} " + body.replace("\n", " | "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="stuur een testmelding")
    ap.add_argument("--history", action="store_true", help="toon signalen van de afgelopen dagen (geen push)")
    ap.add_argument("--once", action="store_true", help="draai één keer en stop")
    ap.add_argument("--dry", action="store_true", help="print meldingen, maar push niet")
    args = ap.parse_args()

    if args.test:
        push("Test Quantum alerts", "Als je dit ziet, werken de pushmeldingen.", "default", "white_check_mark")
        print("Testmelding verstuurd naar topic:", NTFY_TOPIC)
        return
    if NTFY_TOPIC.endswith("VERANDER-MIJ") and not (args.history or args.dry):
        sys.exit("Zet eerst je eigen geheime topicnaam: export NTFY_TOPIC=...")

    print("Paren laden...")
    pairs = load_pairs()
    print(f"{len(pairs)} paren: " + ", ".join(p["name"] for p in pairs))
    if args.history:
        return history(pairs)
    while True:
        cycle(pairs, dry=args.dry)
        if args.once:
            return
        now = time.time()
        time.sleep(max(5, (int(now // 900) + 1) * 900 + 20 - now))  # 20s na elke 15m-close


if __name__ == "__main__":
    main()
