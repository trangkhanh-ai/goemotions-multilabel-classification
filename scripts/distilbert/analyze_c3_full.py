"""Tổng hợp C3 full, tuning 3 seed và lỗi validation thật; tuyệt đối không đọc test."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.models.distilbert_study import ROOT, read_json, write_json, prepare_frames
from src.datasets.goemotions import sha256, multi_hot
from src.evaluation.metrics import evaluate_multilabel
from src.models.baseline import load_aligned_scores, tune_thresholds, label_error_pairs
from scripts.distilbert.summarize_distilbert import summarize

METRICS = ['macro_f1','micro_f1','macro_precision','macro_recall','micro_precision','micro_recall','hamming_loss']


def main():
    output = ROOT/'reports/c3_distilbert/full'
    protocol = read_json(output/'protocol.json')
    runs = [ROOT/f'data/processed/c3_distilbert/full/seed_{s}' for s in protocol['seeds']]
    summary, stats = summarize(runs)  # Kiểm hash/config/code và đủ ba seed.
    summary.to_csv(output/'validation_runs.csv',index=False)
    stats.to_csv(output/'validation_mean_std.csv',index=False)
    representative = int(summary.sort_values(['macro_f1','seed'],ascending=[False,True]).iloc[0].seed)
    frames, names, _ = prepare_frames(protocol['config'])
    truth = multi_hot(frames['validation'].labels.tolist())
    train_counts = multi_hot(frames['train'].labels.tolist()).sum(0)
    rare = protocol['rare_labels_train_defined']
    assert [names[i] for i in np.argsort(train_counts)[:5]] == rare
    write_json(output/'representative_c3.json',{
        'seed':representative, 'run':f'data/processed/c3_distilbert/full/seed_{representative}',
        'selection':protocol['c3_representative_seed'],
        'architecture_winner':None, 'note':'Representative C3 only; C1/C2 results still required to select final demo.'})
    write_json(output/'demo_examples.json',[{'id':row.id,'text':row.text,
        'true_labels':[names[i] for i in row.labels], 'split':'validation'}
        for row in frames['validation'].head(5).itertuples(index=False)])
    all_metrics, per_label, histories, errors, resources = [], [], [], [], []
    for seed,run in zip(protocol['seeds'],runs):
        meta=read_json(run/'run.json')
        assert meta['source_sha256']==protocol['source_sha256']
        scores=load_aligned_scores(run/'validation_scores.npz',frames['validation'].id,names)
        fixed=evaluate_multilabel(truth,scores,names,.5)
        assert fixed==read_json(run/'validation_metrics.json')
        thresholds=tune_thresholds(truth,scores)
        tuned=evaluate_multilabel(truth,scores,names,thresholds)
        artifact={'run_metadata_sha256':sha256(run/'run.json'),'seed':seed,'data_revision':meta['data_revision'],
            'label_names':names,'model_sha256':meta['artifact_sha256']['best/model.safetensors'],
            'scores_sha256':meta['artifact_sha256']['validation_scores.npz'],
            'thresholds':thresholds.tolist(),'grid':protocol['threshold_grid'],'tie_rule':protocol['threshold_tie_rule'],
            'selection_split':'validation','evaluation_split':'same validation; optimistic estimate, not independent',
            'test_used':False}
        write_json(run/'thresholds_validation.json',artifact)
        write_json(output/f'thresholds_seed_{seed}.json',artifact)
        write_json(output/f'run_seed_{seed}.json',meta)
        for mode,metric in [('fixed_0.5',fixed),('per_label_tuned_on_validation',tuned)]:
            all_metrics.append({'seed':seed,'mode':mode,**{k:metric[k] for k in METRICS},
                'empty_prediction_count':metric['empty_prediction_count']})
            for row in metric['per_label']:
                per_label.append({'seed':seed,'mode':mode,'train_support':int(train_counts[row['label_id']]),
                    'rare_train_defined':row['label'] in rare,**row})
        write_json(output/f'metrics_fixed_seed_{seed}.json',fixed)
        write_json(output/f'metrics_tuned_seed_{seed}.json',tuned)
        for h in read_json(run/'history.json'):
            histories.append({'seed':seed,**h})
        resources.append({'seed':seed,'skipped_amp_steps':len(meta.get('skipped_amp_steps',[])),
            **{k:meta[k] for k in ['best_epoch','optimizer_steps','effective_batch_size',
            'trainable_parameters','fit_and_validation_seconds','peak_cuda_allocated_mib','peak_cuda_reserved_mib',
            'reload_max_abs_score_diff']}})
        pred=scores>=.5
        fn=truth.astype(bool)&~pred
        fp=~truth.astype(bool)&pred
        groups={
            'FN_va_FP_cung_cau':fn.any(1)&fp.any(1),
            'thieu_mot_phan_nhan_that':(truth.sum(1)>=2)&((pred&truth.astype(bool)).sum(1)>0)&fn.any(1),
            'sai_nhan_hiem_hoac_neutral':(fn|fp)[:,[names.index(n) for n in rare+['neutral']]].any(1)}
        for group,mask in groups.items():
            errors.append({'seed':seed,'group':group,'n_samples':int(mask.sum()),'n_validation':len(truth),
                           'note':'Groups may overlap; counts are not additive'})
        pairs=label_error_pairs(truth,scores,names)
        pd.DataFrame(pairs).to_csv(output/f'error_pairs_seed_{seed}.csv',index=False)
        # Nguyên văn thật, ID thật, đúng label scores; chọn mẫu theo thứ tự ID để không chỉ lấy ví dụ đẹp.
        if seed==representative:
            examples=[]
            for group,mask in groups.items():
                indices=sorted(np.flatnonzero(mask),key=lambda i:frames['validation'].id.iloc[i])[:12]
                for i in indices:
                    examples.append({'group':group,'seed':seed,'id':frames['validation'].id.iloc[i],
                        'text':frames['validation'].text.iloc[i],
                        'true_labels':[names[j] for j in np.flatnonzero(truth[i])],
                        'predicted_labels':[names[j] for j in np.flatnonzero(pred[i])],
                        'missed_labels':[names[j] for j in np.flatnonzero(fn[i])],
                        'extra_labels':[names[j] for j in np.flatnonzero(fp[i])],
                        'scores':{name:float(value) for name,value in zip(names,scores[i])},
                        'threshold':.5})
            write_json(output/'error_examples.json',examples)
            readable=pd.DataFrame([{k:v for k,v in row.items() if k!='scores'} for row in examples])
            readable.to_csv(output/'error_examples.csv',index=False,encoding='utf-8-sig')
    rows=pd.DataFrame(all_metrics)
    rows.to_csv(output/'threshold_comparison_runs.csv',index=False)
    aggregated=[]
    for mode,part in rows.groupby('mode'):
        for metric in METRICS:
            aggregated.append({'mode':mode,'metric':metric,'mean':float(part[metric].mean()),
                               'sample_std':float(part[metric].std(ddof=1)),'n':len(part)})
    pd.DataFrame(aggregated).to_csv(output/'threshold_comparison_mean_std.csv',index=False)
    labels=pd.DataFrame(per_label)
    labels.to_csv(output/'per_label_all_seeds.csv',index=False)
    labels[labels.rare_train_defined].to_csv(output/'rare_labels_before_after.csv',index=False)
    means=labels[labels.rare_train_defined].groupby(['label','mode'],sort=False).agg(
        train_support=('train_support','first'),validation_support=('support','first'),
        f1_mean=('f1','mean'),f1_std=('f1','std'),precision_mean=('precision','mean'),recall_mean=('recall','mean')).reset_index()
    means.to_csv(output/'rare_labels_mean_std.csv',index=False)
    pd.DataFrame(histories).to_csv(output/'history_all_seeds.csv',index=False)
    pd.DataFrame(resources).to_csv(output/'resources_all_seeds.csv',index=False)
    pd.DataFrame(errors).to_csv(output/'error_group_counts.csv',index=False)
    write_json(output/'analysis_manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'test_used':False,'n_train':len(frames['train']),'n_validation':len(frames['validation']),
        'seeds':protocol['seeds'],'representative_seed':representative,
        'source_sha256':{p:sha256(ROOT/p) for p in ['scripts/distilbert/analyze_c3_full.py','src/models/baseline.py','src/evaluation/distilbert_thresholds.py']},
        'run_metadata_sha256':{str(seed):sha256(run/'run.json') for seed,run in zip(protocol['seeds'],runs)}})
    print(summary.to_string(index=False))
    print('PASS: real full validation results, 3 seeds, thresholds and error evidence exported.')


if __name__=='__main__':
    main()
