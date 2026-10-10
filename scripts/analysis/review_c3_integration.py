"""Đối chiếu hồ sơ C3 đã có sau tích hợp; không train hoặc sinh dự đoán mới."""
import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.paths import ROOT
OUT = ROOT / 'reports/c3_distilbert/post_merge_20261009'
EXPORT = ROOT / 'reports/reproducibility'
METRICS = ('macro_f1', 'micro_f1', 'macro_precision', 'macro_recall',
           'micro_precision', 'micro_recall', 'hamming_loss')


def main():
    inputs = {}

    def read(path):
        path = ROOT / path
        content = path.read_bytes()
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
        return content.decode('utf-8-sig')

    def js(path):
        return json.loads(read(path))

    def table(path):
        return list(csv.DictReader(read(path).splitlines()))

    def equal(a, b):
        assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), (a, b)

    labels = js('data/labels.json')
    manifest = js('reports/reproducibility/manifest.json')
    exports = {}
    for item in manifest['artifacts']:
        if item['source_path'].startswith('data/processed/transformers/'):
            assert item['status'] == 'exported'
            path = EXPORT / item['export_path']
            read(path)
            assert inputs[path.relative_to(ROOT).as_posix()] == item['export_sha256'] == item['source_sha256']
            exports[item['source_path']] = path

    base = 'data/processed/transformers/'
    selection = js(exports[base + 'selected_model.json'])
    candidates = []
    for arch in ('bert', 'roberta', 'distilbert'):
        values = []
        for seed in (42, 123, 2026):
            folder = f'{base}{arch}/seed_{seed}/full/standard/'
            meta = js(exports[folder + 'run_metadata.json'])
            metric = js(exports[folder + 'validation_metrics.json'])
            assert meta['completed'] and not meta['smoke']
            assert meta['architecture'] == arch and meta['seed'] == seed
            assert meta['label_names'] == labels and metric['n_labels'] == 28
            assert metric['n_samples'] == 5426 and metric['threshold'] == .5
            values.append((seed, metric['macro_f1']))
        mean = statistics.mean(v for _, v in values)
        std = statistics.stdev(v for _, v in values)
        candidates.append((arch, mean, std, meta['parameter_count'], values))
        recorded = next(x for x in selection['architecture_summary'] if x['architecture'] == arch)
        equal(mean, recorded['metrics']['macro_f1']['mean'])
        equal(std, recorded['metrics']['macro_f1']['std'])
    winner = sorted(candidates, key=lambda x: (-x[1], x[2], x[3], x[0]))[0]
    best_seed = sorted(winner[4], key=lambda x: (-x[1], x[0]))[0][0]
    assert (winner[0], best_seed) == (selection['architecture'], selection['seed'])

    rows = [r for r in table('reports/project_results/all_runs.csv') if r['system'] == 'C_distilbert']
    assert len(rows) == 18
    expected_keys = {(s, split, mode) for s in (42, 123, 2026)
                     for split in ('validation', 'test') for mode in ('fixed', 'global', 'tuned')}
    assert {(int(float(r['seed'])), r['split'], r['threshold_mode']) for r in rows} == expected_keys
    for row in rows:
        seed = int(float(row['seed']))
        folder = f'{base}distilbert/seed_{seed}/full/standard/'
        protocol_path = exports[folder + 'final_protocol.json']
        protocol = js(protocol_path)
        test = js(exports[folder + 'test_results.json'])
        meta_path = exports[folder + 'run_metadata.json']
        meta = js(meta_path)
        assert protocol['label_names'] == labels and protocol['data_revision'] == manifest['data_revision']
        assert protocol['run_metadata_sha256'] == inputs[meta_path.relative_to(ROOT).as_posix()]
        assert protocol['validation_scores_sha256'] == meta['artifact_sha256']['validation_scores.npz']
        assert test['protocol_sha256'] == inputs[protocol_path.relative_to(ROOT).as_posix()]
        source = protocol['configurations'] if row['split'] == 'validation' else test['results']
        entry = next(x for x in source if x['threshold_mode'] == row['threshold_mode'])
        metrics = entry['validation_metrics' if row['split'] == 'validation' else 'metrics']
        assert int(row['n_samples']) == metrics['n_samples'] == (5426 if row['split'] == 'validation' else 5427)
        for name in METRICS:
            equal(row[name], metrics[name])
    aggregates = [r for r in table('reports/project_results/mean_std.csv') if r['system'] == 'C_distilbert']
    assert len(aggregates) == 6
    for row in aggregates:
        group = [r for r in rows if (r['split'], r['threshold_mode']) == (row['split'], row['threshold_mode'])]
        assert len(group) == int(row['n_runs']) == 3
        for name in METRICS:
            values = [float(r[name]) for r in group]
            equal(row[name + '_mean'], statistics.mean(values))
            equal(row[name + '_std'], statistics.stdev(values))

    error_manifest = js('reports/errors_test_standard_fixed/manifest.json')
    assert error_manifest['label_names'] == labels and error_manifest['threshold_mode'] == 'fixed'
    rare = {r['label'] for r in error_manifest['rare_labels']}
    examples = table('reports/errors_test_standard_fixed/examples.csv')
    for row in examples:
        run = next(r for r in error_manifest['runs'] if r['architecture'] == row['architecture'])
        assert int(row['seed']) == run['seed'] and run['thresholds'] == .5
        test = js(exports[run['run_dir'] + '/test_results.json'])
        meta_path = exports[run['run_dir'] + '/run_metadata.json']
        assert run['run_metadata_sha256'] == inputs[meta_path.relative_to(ROOT).as_posix()]
        assert run['scores_sha256'] == test['scores_sha256']
        scores = json.loads(row['scores_by_label'])
        assert list(scores) == labels and all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values())
        gold = set(json.loads(row['true_labels']))
        pred = {n for n, score in scores.items() if score >= .5}
        assert pred == set(json.loads(row['predicted_labels']))
        fn, fp = gold - pred, pred - gold
        assert fn == set(json.loads(row['missed_labels'])) and fp == set(json.loads(row['extra_labels']))
        flags = {'partial_multi_label': len(gold) >= 2 and bool(gold & pred) and bool(fn),
                 'rare_false_negative': bool(fn & rare), 'missed_extra_pair': bool(fn and fp)}
        assert flags[row['category']] == (row['error_present'].lower() == 'true')
        peers = [r for r in examples if (r['category'], r['id']) == (row['category'], row['id'])]
        assert len(peers) == 3 and {r['architecture'] for r in peers} == {'bert', 'roberta', 'distilbert'}
        assert len({(r['text'], r['true_labels']) for r in peers}) == 1

    case_text = read('reports/error_case_studies.md')
    case_ids = ['eczj48j', 'ed0jr9i', 'eczcvgx']
    for ident in case_ids:
        assert ident in case_text
        assert len([r for r in examples if r['id'] == ident]) == 3
    own = table('reports/c3_distilbert/full/validation_runs.csv')
    own_mean = statistics.mean(float(r['macro_f1']) for r in own)
    own_std = statistics.stdev(float(r['macro_f1']) for r in own)
    # Never combine independent historical runs with the shared experiment.
    report = {
        'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'C3 report consistency and shared selection; no training or shared-model inference',
        'exported_transformer_json_hashes_checked': len(exports),
        'c3_shared_rows_checked': len(rows), 'c3_shared_aggregate_rows_checked': len(aggregates),
        'metrics_per_row': len(METRICS), 'error_rows_checked': len(examples),
        'existing_case_ids_reviewed': case_ids,
        'shared_demo_selection': {'architecture': winner[0], 'seed': best_seed},
        'validation_macro_f1_fixed': {
            'huy_independent': {'mean': own_mean, 'sample_std': own_std, 'n': len(own)},
            'shared_distilbert': {'mean': candidates[2][1], 'sample_std': candidates[2][2], 'n': 3}},
        'new_predictions_generated': False, 'raw_test_dataset_read': False,
        'shared_npz_or_weights_verified': False,
        'limitations': ['Shared metrics were checked against exported JSON, not recomputed from missing full NPZ.',
                       'Error labels/text were checked for cross-architecture consistency in existing CSV; raw test was not reopened.',
                       'Selection was recomputed from exported validation metrics; winning BERT weights are absent locally.'],
        'input_sha256': inputs,
        'review_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'shared_c3_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(rows)} C3 rows, {len(aggregates)} aggregates, {len(examples)} existing error rows; selected {winner[0]} seed {best_seed}.')


if __name__ == '__main__':
    main()
