"""
compare_variants.py: does keeping the duplicate rows change the S4 conclusion?

Compares the main analysis (duplicates dropped) with the keep_duplicates
sensitivity analysis on every quantity a conclusion rests on:
  - data: rows, clicks, CTR of each log
  - ranking and mean CTR of every algorithm
  - which pairs are significant after Holm-Bonferroni
  - external-validity hypotheses H1 to H3b
Verdicts are computed, not written by hand, at two levels:
  strict  - identical set of significant pairs and hypothesis
            outcomes in both variants (both streams);
  per conclusion - the specific claims the thesis makes:
    K1 BernoulliTS_Zozotown ranks first
    K2 no significant difference among the algorithms that learn from a
       uniform prior or from scratch (pairs not involving BernoulliTS_Zozotown)
    K3 external validity: H2, H3a and H3b
    K4 H1 (BernoulliTS_Zozotown > B0 Random, one-sided)
    K5 the order of the from-scratch learners (descriptive only)

Run after both variants exist:
  python obd_preprocessing/compare_variants.py
Outputs: outputs/variant_comparison.json / .md
"""

from __future__ import annotations

import json

import common as C


def load(variant):
    C.set_variant(variant)
    res = json.loads((C.OUT_DIR / "s4_replay_results.json").read_text())
    integ = json.loads((C.OUT_DIR / "fix5_integrity_report.json").read_text())
    return res, {r["policy"]: r for r in integ["reports"]}


def sig_set(analysis):
    return sorted(f"{p['a']} vs {p['b']} -> {p['higher_ctr']}" for p in analysis["pairwise"] if p["significant"])


def hyp(ev):
    return {k: ev[k]["supported"] for k in ("H1_bts_beats_random", "H2_rank_top_two",
                                             "H3a_direction_matches", "H3b_ratio_matches")}


def main():
    (m, mi), (k, ki) = load("main"), load("keep_duplicates")
    C.set_variant("main")
    out = {"data": {}, "streams": {}}
    for pol in C.POLICIES:
        out["data"][pol] = {v: {"rows": d[pol]["rows_clean"], "clicks": d[pol]["clicks_clean"],
                                "ctr": d[pol]["ctr_clean"]} for v, d in (("main", mi), ("keep_duplicates", ki))}
    unchanged_all = True
    concl = {}
    for stream in ("pooled", "random_only"):
        A, B = m["analyses"][stream], k["analyses"][stream]
        EA, EB = m["external_validity"][stream], k["external_validity"][stream]
        same = {"significant_pairs": sig_set(A) == sig_set(B),
                "hypotheses": hyp(EA) == hyp(EB)}
        unchanged_all &= all(same.values())
        z = "BernoulliTS_Zozotown"
        others = lambda R: [p for p in R["pairwise"] if p["significant"] and z not in (p["a"], p["b"])]
        order = lambda R: [a for a in R["ranking"] if a != z]
        concl[stream] = {
            "K1_bts_zozotown_ranks_first": {"main": A["ranking"][0] == z, "keep_duplicates": B["ranking"][0] == z},
            "K2_no_significant_pair_among_other_learners": {"main": not others(A), "keep_duplicates": not others(B)},
            "K3_external_validity_H2_H3a_H3b": {
                "main": all(EA[h]["supported"] for h in ("H2_rank_top_two", "H3a_direction_matches", "H3b_ratio_matches")),
                "keep_duplicates": all(EB[h]["supported"] for h in ("H2_rank_top_two", "H3a_direction_matches", "H3b_ratio_matches"))},
            "K4_H1_bts_beats_random": {"main": EA["H1_bts_beats_random"]["supported"],
                                       "keep_duplicates": EB["H1_bts_beats_random"]["supported"],
                                       "p_main": EA["H1_bts_beats_random"]["p"],
                                       "p_keep_duplicates": EB["H1_bts_beats_random"]["p"]},
            "K5_order_of_other_learners": {"main": order(A), "keep_duplicates": order(B)}}
        for kk, v in concl[stream].items():
            v["same"] = v["main"] == v["keep_duplicates"]
        out["streams"][stream] = {
            "algorithms": {a: {"main_ctr": A["summary"][a]["mean_ctr"], "kd_ctr": B["summary"][a]["mean_ctr"],
                               "main_rank": A["ranking"].index(a) + 1, "kd_rank": B["ranking"].index(a) + 1}
                           for a in A["ranking"]},
            "significant_pairs": {"main": sig_set(A), "keep_duplicates": sig_set(B)},
            "smallest_p": {"main": min(p["p_raw"] for p in A["pairwise"]),
                           "keep_duplicates": min(p["p_raw"] for p in B["pairwise"])},
            "external_validity": {"main": EA, "keep_duplicates": EB},
            "same": same}
    out["conclusions"] = concl
    out["verdict_strict"] = ("UNCHANGED: keeping the duplicates changes no significant pair and no "
                      "hypothesis outcome." if unchanged_all else
                      "CHANGED: at least one significant pair or hypothesis outcome differs; "
                      "see the 'same' flags.")
    C.write_json(out, C.OUT_DIR / "variant_comparison.json")

    names = {"K1_bts_zozotown_ranks_first": "K1 BernoulliTS_Zozotown ranks first",
             "K2_no_significant_pair_among_other_learners": "K2 No significant pair among the other learners",
             "K3_external_validity_H2_H3a_H3b": "K3 External validity (H2, H3a, H3b) holds",
             "K4_H1_bts_beats_random": "K4 H1: BernoulliTS_Zozotown > B0 Random",
             "K5_order_of_other_learners": "K5 Order of the other (non-significant) learners"}
    L = ["# Sensitivity: duplicates dropped (main) vs kept", "",
         f"Strict check: {out['verdict_strict']}", "",
         "## Conclusion by conclusion", "",
         "| Conclusion | Stream | Main | Kept | Same |", "|---|---|---|---|---|"]
    for stream, K in concl.items():
        for kk, v in K.items():
            if kk == "K5_order_of_other_learners":
                a, b = "see table below", "see table below"
            elif kk == "K4_H1_bts_beats_random":
                a = f"{'yes' if v['main'] else 'no'} (p = {v['p_main']:.4f})"
                b = f"{'yes' if v['keep_duplicates'] else 'no'} (p = {v['p_keep_duplicates']:.4f})"
            else:
                a, b = (str(v["main"]) if isinstance(v["main"], str) else ("yes" if v["main"] else "no"),
                        str(v["keep_duplicates"]) if isinstance(v["keep_duplicates"], str)
                        else ("yes" if v["keep_duplicates"] else "no"))
            L.append(f"| {names[kk]} | {stream} | {a} | {b} | {'yes' if v['same'] else '**no**'} |")
    L += ["",
         "| Log | Rows main | Rows kept | Clicks main | Clicks kept | CTR main | CTR kept |",
         "|---|---|---|---|---|---|---|"]
    for pol, d in out["data"].items():
        a, b = d["main"], d["keep_duplicates"]
        L.append(f"| {pol} | {a['rows']:,} | {b['rows']:,} | {a['clicks']:,} | {b['clicks']:,} | "
                 f"{a['ctr']:.4%} | {b['ctr']:.4%} |")
    for stream, S in out["streams"].items():
        L += ["", f"## Stream: {stream}", "",
              "| Algorithm | Mean CTR main | Mean CTR kept | Rank main | Rank kept |", "|---|---|---|---|---|"]
        for a, v in S["algorithms"].items():
            L.append(f"| {a} | {v['main_ctr']:.4%} | {v['kd_ctr']:.4%} | {v['main_rank']} | {v['kd_rank']} |")
        em, ek = S["external_validity"]["main"], S["external_validity"]["keep_duplicates"]
        L += ["", "| Quantity | Main | Kept | Same conclusion |", "|---|---|---|---|",
              f"| Significant pairs (Holm) | {len(S['significant_pairs']['main'])} | "
              f"{len(S['significant_pairs']['keep_duplicates'])} | {'yes' if S['same']['significant_pairs'] else 'no'} |",
              f"| Smallest raw p | {S['smallest_p']['main']:.4g} | {S['smallest_p']['keep_duplicates']:.4g} | |",
              f"| Replay ratio BTS-Z / Random | {em['replay_ratio']:.3f}x | {ek['replay_ratio']:.3f}x | |",
              f"| Ground-truth ratio | {em['ground_truth_ratio_this_data']:.3f}x | "
              f"{ek['ground_truth_ratio_this_data']:.3f}x | |"]
        for h in ("H1_bts_beats_random", "H2_rank_top_two", "H3a_direction_matches", "H3b_ratio_matches"):
            L.append(f"| {h} | {'supported' if em[h]['supported'] else 'not supported'} | "
                     f"{'supported' if ek[h]['supported'] else 'not supported'} | "
                     f"{'yes' if em[h]['supported'] == ek[h]['supported'] else 'no'} |")
    (C.OUT_DIR / "variant_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
