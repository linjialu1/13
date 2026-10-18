"""Portable audit of saved scalar records. No model/data import or training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    record_root = root / 'records/s3'
    inp = record_root / 'test_s4/test_per_seed_raw.csv'
    with inp.open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    stored = load_json(record_root / 'test_s4/test_main_table.json')
    aggregate = load_json(record_root / 'aggregate.json')
    groups = defaultdict(list)
    for row in raw:
        groups[row['cell']].append(row)
    problems = []
    checks = []
    def check(ok, label):
        checks.append({'label': label, 'pass': bool(ok)})
        if not ok:
            problems.append(label)
    def close(actual, expected, label, atol=1e-12):
        check(expected is not None and actual is not None and math.isfinite(float(actual))
              and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=atol), label)
    expected_rows = {r['cell']: r for r in stored['rows']}
    check(len(raw) == 117, '117 saved cell/seed scalar observations')
    check(len(groups) == len(expected_rows) == 24, '24 cells')
    check(stored['holm_family'] == 24, 'fixed Holm family is 24')
    tcrit = float(stats.t.ppf(0.975, df=7))
    close(tcrit, stored['t_crit_df7'], 'df=7 two-sided 95% t critical value')
    computed = []
    for cell in sorted(groups):
        rows = sorted(groups[cell], key=lambda r: int(r['seed']))
        seeds = [int(r['seed']) for r in rows]
        check(len(seeds) == len(set(seeds)), f'{cell}: no duplicated seed')
        n = len(rows)
        check(seeds == ([11,22,33,44,55,66,77,88] if n == 8 else [11,22,33]), f'{cell}: expected seed identities')
        d = np.array([float(r['candidate_test_macro']) - float(r['s1_test_macro']) for r in rows], dtype=np.float64)
        for i, row in enumerate(rows):
            close(d[i], float(row['paired_gain']), f'{cell}/seed{seeds[i]}: candidate minus paired baseline')
            close(d[i] * 100, float(row['paired_gain_pp']), f'{cell}/seed{seeds[i]}: percentage-point conversion')
        mean = float(d.mean())
        sd = float(d.std(ddof=1))
        ref = expected_rows[cell]
        agg = aggregate['cells'][cell]
        close(mean, ref['raw_mean'], f'{cell}: saved full-precision mean')
        close(sd, ref['raw_sd'], f'{cell}: saved sample SD ddof=1')
        close(mean, agg['mean'], f'{cell}: aggregate mean')
        close(sd, agg['sd'], f'{cell}: aggregate SD')
        check(int((d > 0).sum()) == ref['strictly_positive'], f'{cell}: strictly-positive count')
        check(n == ref['n_seeds'] == agg['n_seeds'], f'{cell}: sample size')
        check(ref['mean_gain_pp'] == float(f'{100*mean:.6f}'), f'{cell}: rounded table mean')
        check(ref['sd_gain_pp'] == float(f'{100*sd:.6f}'), f'{cell}: rounded table SD')
        rec = {'cell': cell, 'n': n, 'seeds': seeds, 'mean': mean, 'mean_pp': 100*mean,
               'sd_ddof1': sd, 'sd_pp': 100*sd, 'strictly_positive': int((d > 0).sum()),
               'df': None, 't': None, 'ci95': None, 'raw_p': None, 'holm24_p': None}
        if n == 8:
            se = sd / math.sqrt(n)
            lo, hi = mean - tcrit*se, mean + tcrit*se
            tv = None if sd == 0 else mean/se
            p = (1.0 if mean == 0 else 0.0) if sd == 0 else float(2*stats.t.sf(abs(tv), df=7))
            rec.update({'df': 7, 't': tv, 'ci95': [lo, hi], 'raw_p': p})
            close(lo, ref['raw_ci_low'], f'{cell}: nominal t CI lower')
            close(hi, ref['raw_ci_high'], f'{cell}: nominal t CI upper')
            close(lo, agg['ci95'][0], f'{cell}: aggregate CI lower')
            close(hi, agg['ci95'][1], f'{cell}: aggregate CI upper')
            close(p, ref['raw_p'], f'{cell}: raw p rounded to 12 decimals', atol=5.1e-13)
            close(p, aggregate['holm_family_24']['results'][cell]['raw_p'], f'{cell}: aggregate unrounded raw p', atol=1e-14)
            if tv is not None:
                close(float(f'{tv:.6f}'), ref['t_stat'], f'{cell}: rounded t statistic')
        else:
            check(n == 3, f'{cell}: only n=3 descriptive or n=8 full rows allowed')
            for field in ['raw_ci_low', 'raw_ci_high', 'ci95_low', 'ci95_high',
                          'ci95_low_pp', 'ci95_high_pp', 't_stat', 'raw_p', 'p_adjusted_holm24', 'holm_rank']:
                check(ref[field] is None, f'{cell}: n=3 {field} remains null')
            check(agg['p'] is None and agg['ci95'] is None, f'{cell}: aggregate n=3 p and CI remain null')
        computed.append(rec)
    inferential = [r for r in computed if r['n'] == 8]
    check(len(inferential) == 9, '9 eight-seed rows')
    check(len(computed) - len(inferential) == 15, '15 three-seed descriptive rows')
    ordered = sorted(inferential, key=lambda r: r['raw_p'])
    running = 0.0
    for rank, row in enumerate(ordered, 1):
        running = max(running, (24-rank+1)*row['raw_p'])
        row['holm24_p'] = min(1.0, running)
        row['holm_rank'] = rank
        row['holm_weight'] = 25-rank
        ref = expected_rows[row['cell']]
        close(row['holm24_p'], ref['p_adjusted_holm24'], f"{row['cell']}: Holm24 stored rounded p", atol=5.1e-13)
        close(row['holm24_p'], aggregate['holm_family_24']['results'][row['cell']]['adjusted_p'], f"{row['cell']}: aggregate full precision Holm24", atol=1e-14)
        check(rank == ref['holm_rank'], f"{row['cell']}: Holm rank")
    # Algebraic placeholder check only; original stored nulls are never rewritten as 1.
    full_p = sorted([r['raw_p'] for r in inferential] + [1.0]*15)
    adjusted_full = np.minimum(1.0, np.maximum.accumulate([(24-i)*p for i,p in enumerate(full_p)]))
    for i,row in enumerate(ordered):
        close(row['holm24_p'], adjusted_full[i], f"{row['cell']}: reserved-slot p=1 equivalence")
    # Independent consistency checks on existing readout contrasts and descriptive S4.
    s4 = load_json(record_root / 'test_s4/s4_summary.json')
    s2 = load_json(root / 'records/s2/endpoint_summary.json')
    endpoint = {r['dataset']: r for r in s2['endpoints']}
    for candidate, key in [('R1_analytic_head', 'r1'), ('R2_pca16_readout', 'r2')]:
        cells = [r for r in computed if r['cell'].startswith(candidate+'__')]
        x = [endpoint[r['cell'].split('__')[1]]['mean'] for r in cells]
        y = [r['mean'] for r in cells]
        rho = float(stats.spearmanr(x, y).statistic)
        close(rho, s4[key]['rho'], f'{key}: S4 descriptive Spearman over four datasets')
        check(len(cells) == s4[key]['n'] == 4, f'{key}: four dataset units')
    report = {'status': 'PASS' if not problems else 'FAIL', 'scope': 'Offline recomputation of saved scalar results only; no raw-data retraining or independent experimental replication.',
              'rows': computed, 'checks_passed': sum(c['pass'] for c in checks), 'checks_total': len(checks),
              'problems': problems, 'checks': checks,
              'input_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [inp, record_root/'test_s4/test_main_table.json', record_root/'aggregate.json', record_root/'test_s4/s4_summary.json', root/'records/s2/endpoint_summary.json']},
              'null_policy': 'All 15 n=3 rows retain null p/CI in the reported main table. Appending p=1 is a calculation-equivalence check, never a result-file replacement.',
              'inference_limits': 'Nominal seed-level t intervals and tests condition on the fixed data split/order/protocol. The selected C1/WISDM row is not adjusted for validation screening. No new confirmatory test is introduced.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'checks_passed', 'checks_total', 'problems']}, indent=2))
    return 0 if not problems else 1

if __name__ == '__main__':
    raise SystemExit(main())
