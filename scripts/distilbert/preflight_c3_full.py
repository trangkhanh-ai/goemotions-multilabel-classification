"""Đo bộ nhớ trên batch dữ liệu train thật trước khi khóa cấu hình full; không báo F1."""
import gc
from time import perf_counter
from src.models.distilbert_study import ROOT, read_json, write_json, prepare_frames, seed_everything
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.datasets.goemotions import multi_hot


def main():
    torch.set_num_threads(4)
    config = read_json(ROOT/'configs/distilbert_full_candidate.json')
    frames, names, _ = prepare_frames(config)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'], local_files_only=True)
    # Chỉ dùng 256 bình luận train có thật; padding tới 128 để đo trường hợp nặng.
    frame = frames['train'].iloc[:256]
    encoded = tokenizer(frame.text.tolist(), max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    labels = torch.tensor(multi_hot(frame.labels.tolist()), dtype=torch.float32)
    results = []
    for checkpointing in (False, True):
        seed_everything(42)
        model = AutoModelForSequenceClassification.from_pretrained(config['model_id'], revision=config['revision'],
            local_files_only=True, use_safetensors=True, num_labels=28, problem_type='multi_label_classification').cuda().train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler('cuda')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = perf_counter()
        for offset in range(0,256,16):
            optimizer.zero_grad(set_to_none=True)
            batch = {k:v[offset:offset+16].cuda() for k,v in encoded.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**batch, labels=labels[offset:offset+16].cuda()).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        results.append({'gradient_checkpointing':checkpointing,'batch_size':16,'padded_length':128,
            'n_real_train_samples':256,'seconds':perf_counter()-start,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'finite_last_loss':bool(torch.isfinite(loss))})
        print(results[-1],flush=True)
        del model,optimizer,scaler,batch,loss
        gc.collect()
        torch.cuda.empty_cache()
    write_json(ROOT/'reports/c3_distilbert/full_preflight.json',{
        'purpose':'Memory/time check only; discarded weights; no validation/test and no quality comparison',
        'source':'first 256 official train rows; actual texts and labels; padding to max_length',
        'ids':frame.id.tolist(),'cpu_threads':4,'results':results})


if __name__=='__main__':
    main()
