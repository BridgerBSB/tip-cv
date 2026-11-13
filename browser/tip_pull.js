/* tip_pull.js - clip lists for pitch-tipping work, pulled in Chrome on statsapi.mlb.com.
 *
 * Why the browser: the analytics(video) endpoint is Okta-gated. The clips themselves are not.
 * Sign-in: open https://statsapi.mlb.com/, Login with Okta -> okta button -> Chrome autofills ->
 * ZAC clicks Sign In (Claude does not submit a password sign-in). Everything else is Claude's.
 *
 * Usage (paste this file through javascript_tool, then):
 *   window.__tip2Run = window.__tip2Pull([pk1, pk2, ...], <pitcher id or 0 for all>, "R", "tip2_<who>.csv"); "started"
 *   JSON.stringify(window.__tip2Log.slice(-3))     // progress, one line per game
 * Each call saves one CSV to Downloads. Move it to data/pulls/.
 *
 * One row per pitch, pickoff and step-off (the three things a pitcher does with a runner on).
 * Columns:
 *   on1b/on2b/on3b, balls_pre/strikes_pre/outs_pre - from the analytics payload's `count`, which is the
 *     state BEFORE the event (the feed's playEvents[].count is AFTER it - do not mix them).
 *   begin_of_play_utc / first_move_utc / ball_pitched_utc / ball_released_utc - MLB trackedEvents, UTC.
 *     Clock anchors. A pitch is stamped BALL_WAS_PITCHED; a pickoff throw is stamped BALL_WAS_RELEASED.
 *     tracked_times keeps every stamp as NAME=time;... so nothing new is lost.
 *   url_<ANGLE> - one column per known camera; angles_all lists every angle offered, so a new one is visible.
 * MiLB games offer only BROADCAST; MLB games offer 9-12 angles (verified 2026-11-12).
 */
(function () {
  const ANGLES = ["PITCHCAST","CENTERFIELD","BROADCAST","ALTERNATE_BROADCAST","3B_PITCHER_SIDEVIEW","1B_PITCHER_SIDEVIEW","3B_BATTER_SIDEVIEW","1B_BATTER_SIDEVIEW","HIGH_HOME","MID_HOME"];
  const BASE = ["game_pk","date","game_type","at_bat_index","event_index","pitch_number","inning","top","event_type","is_pitch","play_id","description","pitch_code","pitch_name","start_speed","bat_side","pitch_hand","batter_id","pitcher_id","balls_pre","strikes_pre","outs_pre","on1b","on2b","on3b","start_time_utc","begin_of_play_utc","first_move_utc","ball_pitched_utc","ball_released_utc","tracked_times","tracked_events","angles_all"];
  const COLS = BASE.concat(ANGLES.map(a => "url_" + a));
  window.__tip2Log = []; window.__tip2Rows = [];
  const log = m => window.__tip2Log.push(m);
  const esc = v => { v = (v === null || v === undefined) ? "" : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  async function analytics(pk, id) {
    for (let a = 0; a < 3; a++) {
      try { const r = await fetch(`/api/v1/game/${pk}/${id}/analytics?hydrate=analytics(video)`, {credentials: "include"});
            if (r.status === 200) return await r.json(); if (r.status === 401) return {__auth: true}; } catch (e) {}
      await new Promise(s => setTimeout(s, 800));
    }
    return null;
  }
  window.__tip2Pull = async function (games, pitcherId, gameType, name) {
    const rows = [];
    for (const pk of games) {
      try {
        const feed = await fetch(`/api/v1.1/game/${pk}/feed/live`).then(r => r.json());
        const date = feed.gameData.datetime.officialDate;
        const evs = [];
        feed.liveData.plays.allPlays.forEach(p => (p.playEvents || []).forEach(e => {
          if (pitcherId && p.matchup.pitcher.id !== pitcherId) return;
          if (!(e.isPitch || e.type === "pickoff" || e.type === "stepoff")) return;
          evs.push({p, e});
        }));
        let i = 0, nUrl = 0, auth = false;
        async function worker() {
          while (i < evs.length) {
            const {p, e} = evs[i++];
            let a = e.playId ? await analytics(pk, e.playId) : null;
            if (a && a.__auth) { auth = true; a = null; }
            const c = (a && a.metaData && a.metaData.stat && a.metaData.stat.play && a.metaData.stat.play.count) || {};
            const te = (a && a.trackedEvents) || [];
            const tt = n => { const x = te.find(t => t.playEvent === n); return x ? x.timeStamp : ""; };
            const urls = {}; const all = [];
            ((a && a.video && a.video.playbackGroups) || []).forEach(g => { all.push(g.mediaSourceType); const rd = (g.playbackRenditions || [])[0]; if (rd && rd.playbackUrl) urls[g.mediaSourceType] = rd.playbackUrl; });
            if (Object.keys(urls).length) nUrl++;
            const d = e.details || {};
            const row = {game_pk: pk, date, game_type: gameType, at_bat_index: p.atBatIndex, event_index: e.index, pitch_number: e.pitchNumber,
              inning: p.about.inning, top: p.about.isTopInning, event_type: e.type, is_pitch: !!e.isPitch, play_id: e.playId,
              description: d.description, pitch_code: d.type ? d.type.code : "", pitch_name: d.type ? d.type.description : "",
              start_speed: e.pitchData ? e.pitchData.startSpeed : "", bat_side: p.matchup.batSide.code, pitch_hand: p.matchup.pitchHand.code,
              batter_id: p.matchup.batter.id, pitcher_id: p.matchup.pitcher.id, balls_pre: c.balls, strikes_pre: c.strikes, outs_pre: c.outs,
              on1b: c.runnerOn1b, on2b: c.runnerOn2b, on3b: c.runnerOn3b, start_time_utc: e.startTime,
              begin_of_play_utc: tt("BEGIN_OF_PLAY"), first_move_utc: tt("PITCHER_FIRST_MOVEMENT"), ball_pitched_utc: tt("BALL_WAS_PITCHED"), ball_released_utc: tt("BALL_WAS_RELEASED"),
              tracked_times: te.map(t => t.playEvent + "=" + t.timeStamp).join(";"),
              tracked_events: te.map(t => t.playEvent).join(";"), angles_all: all.join(";")};
            ANGLES.forEach(k => row["url_" + k] = urls[k] || "");
            rows.push(row);
          }
        }
        await Promise.all([worker(), worker(), worker(), worker()]);
        log({pk, date, events: evs.length, with_url: nUrl, auth_fail: auth});
      } catch (err) { log({pk, error: String(err)}); }
    }
    window.__tip2Rows = window.__tip2Rows.concat(rows);
    const text = COLS.join(",") + "\n" + rows.map(r => COLS.map(k => esc(r[k])).join(",")).join("\n");
    const el = document.createElement("a"); el.href = URL.createObjectURL(new Blob([text], {type: "text/csv"}));
    el.download = name; document.body.appendChild(el); el.click(); el.remove();
    log({done: name, rows: rows.length});
    return rows.length;
  };
  return "tip_pull ready";
})();
