"""Lean P1 probe: reuse #121 numeric export + #117 rev3 curves. No training, no frontier regen."""
from __future__ import annotations
import gzip, importlib.util, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
REV3 = ROOT / 'experiments/psem_small_model_probe'
EXP121 = ROOT / 'experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1'
OUT = ROOT / 'experiments/psem_evidence_delivery_gap/traces'

TH_H_C = 0.5887844788775033
TH_H_M = 0.3631036176874745
TH_F0 = 0.5
HORIZON = 100

CASES = [  # predeclared 2026-09-08T10:12:42Z, mechanical frame-disagreement scan only
    ('G01', 'ami_ES2015d', 10504, 15874, 'H-only'), ('G02', 'ami_ES2009b', 829, 832, 'H-only'),
    ('G03', 'ami_ES2002b', 2888, 2901, 'H-only'), ('G04', 'ami_ES2009a', 5016, 5022, 'H-only'),
    ('G05', 'ami_EN2009d', 3752, 3762, 'H-only'), ('G06', 'ami_EN2009d', 17742, 17749, 'H-only'),
    ('G07', 'ami_ES2009a', 5088, 5096, 'H-only'), ('G08', 'ami_ES2009d', 5955, 5963, 'H-only'),
    ('G09', 'ami_EN2009d', 20064, 20091, 'F0-only'), ('G10', 'ami_ES2009a', 390, 412, 'F0-only'),
    ('G11', 'ami_ES2009d', 10861, 10883, 'F0-only'), ('G12', 'ami_ES2009c', 3706, 3724, 'F0-only'),
    ('G13', 'alimeeting_R8009_M8019', 5185, 5200, 'F0-only'),
    ('G14', 'alimeeting_R1021_M4073', 1956, 1968, 'F0-only'),
    ('G15', 'ami_ES2002b', 5310, 5321, 'F0-only'), ('G16', 'ami_ES2009a', 4529, 4541, 'F0-only'),
]


def load_pa():
    spec = importlib.util.spec_from_file_location(
        'pa', str(EXP121 / 'persistence_analysis.py'))
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    return pa
def text_diff(members):
    import difflib
    real = json.loads((OUT / 'realcheck.json').read_text(encoding='utf-8'))
    out = {}
    for clip in real:
        whole = ' '.join(clip['whole']['finals'])
        halves = (' '.join(clip['first_half']['finals']) + ' ||| '
                  + ' '.join(clip['second_half']['finals']))
        gid = clip['case']
        m = members[next(s for g, s, a, b, d in CASES if g == gid)]
        dev = m['dev']
        n = m['frames']
        split_frame = next(
            (i for i in range(n)
             if int(np.asarray(dev.starts).reshape(-1)[i]) >= clip['frozen_ref_boundary_sample']),
            n - 1)
        lo, hi = max(0, split_frame - 30), min(n, split_frame + 30)
        idx = np.arange(lo, hi)
        espk = np.asarray(dev.episode_speakers).reshape(-1)[idx]
        out[gid] = {
            'whole_final': whole,
            'halves_finals_joined': halves,
            'word_diff': list(difflib.unified_diff(
                whole.split(), halves.split(), lineterm=''))[:40],
            'frame_window': [lo, hi],
            'split_frame': split_frame,
            'speakers_window': ''.join('1' if s == espk[0] else '*' for s in espk),
            'speaker_ids_window': sorted(set(str(s) for s in espk)),
            'overlap_window': ''.join(
                '1' if v else '.' for v in np.asarray(dev.overlap).reshape(-1)[idx]),
            'speech_window': ''.join(
                '1' if v else '.' for v in np.asarray(dev.speech_present).reshape(-1)[idx]),
            'target_window': ''.join(
                '1' if float(v) > 0 else '.' for v in np.asarray(m['target'])[idx]),
        }
    return out
def owner_alignment():
    import xml.etree.ElementTree as ET
    ann = OUT / 'annotations_reuse'
    segs = {}
    for clip, files in (('G04', ['ES2009a.A', 'ES2009a.B', 'ES2009a.C', 'ES2009a.D']),
                        ('G05', ['EN2009d.A', 'EN2009d.B', 'EN2009d.C', 'EN2009d.D'])):
        intervals = []
        for name in files:
            speaker = name.split('.')[-1]
            root = ET.parse(str(ann / (name + '.segments.xml'))).getroot()
            for seg in root.iter():
                if not seg.tag.endswith('segment'):
                    continue
                try:
                    s = float(seg.attrib['transcriber_start'])
                    e = float(seg.attrib['transcriber_end'])
                except (KeyError, ValueError):
                    continue
                intervals.append((s * 1000.0, e * 1000.0, speaker))
        segs[clip] = intervals
    real = {c['case']: c for c in json.loads(
        (OUT / 'realcheck.json').read_text(encoding='utf-8'))}
    timed = {c['case']: c for c in json.loads(
        (OUT / 'token_timing.json').read_text(encoding='utf-8'))}
    out = {}
    for gid in ('G04', 'G05'):
        excerpt_start = real[gid]['excerpt_ms'][0]
        split_excerpt = real[gid]['frozen_ref_boundary_excerpt_ms']
        split_source = excerpt_start + split_excerpt
        labeled = []
        for tok in timed[gid]['final_tokens']:
            if tok['text'] in ('<fin>', '<end>'):
                continue
            end_ms = tok['end_ms']
            if not isinstance(end_ms, (int, float)) or end_ms <= 0:
                continue
            source_ms = excerpt_start + float(end_ms)
            owners = sorted({sp for s, e, sp in segs[gid] if s <= source_ms <= e})
            labeled.append({'text': tok['text'], 'source_ms': round(source_ms, 1),
                            'owners': owners})
        pre = {t['owners'][0] for t in labeled
               if t['source_ms'] < split_source and len(t['owners']) == 1}
        post = {t['owners'][0] for t in labeled
                if t['source_ms'] >= split_source and len(t['owners']) == 1}
        mixed = sum(1 for t in labeled if len(t['owners']) > 1)
        unlabeled = sum(1 for t in labeled if not t['owners'])
        out[gid] = {'tokens_labeled': len(labeled), 'mixed_owner_tokens': mixed,
                    'unlabeled_tokens': unlabeled,
                    'owners_before_split': sorted(pre), 'owners_after_split': sorted(post),
                    'split_source_ms': split_source, 'tokens': labeled}
    return out
def emission_bounds(members):
    from experiments.psem_state_corrected_adaptation_gate.frontier_sweep import (
        episode_runs,
        simulate_episode,
    )
    out = {}
    for gid, sid, a, b, d in CASES:
        if gid not in ('G02', 'G04', 'G05'):
            continue
        m = members[sid]
        dev = m['dev']
        n = m['frames']
        starts = [int(v) for v in np.asarray(dev.starts).reshape(-1)]
        ends = [int(v) for v in np.asarray(dev.ends).reshape(-1)]
        valid = [bool(v) for v in np.asarray(dev.valid).reshape(-1)]
        masked = [bool(v) for v in np.asarray(dev.masked).reshape(-1)]
        speech = [bool(v) for v in np.asarray(dev.speech_present).reshape(-1)]
        frontiers = [int(v) for v in np.asarray(dev.frontiers).reshape(-1)]
        speakers = [str(v) for v in np.asarray(dev.episode_speakers).reshape(-1)]
        epids = [str(v) for v in np.asarray(dev.episode_ids).reshape(-1)]
        case_eps = sorted(set(epids[a:b + 1]))
        bounds = {}
        for key, label in (('cand_raw_prob', 'H'), ('f0_prob', 'F0')):
            scores = [float(v) for v in np.asarray(m[key])]
            thr = TH_H_C if key == 'cand_raw_prob' else TH_F0
            hits = []
            for epkey, frames in episode_runs(epids):
                if epkey not in case_eps:
                    continue
                res = simulate_episode(frames, epkey, sid, speakers, starts, ends,
                                       valid, masked, speech, scores, frontiers,
                                       thr, HORIZON)
                if res is not None:
                    hits.append({'episode': res[1], 'speaker': res[2],
                                 'boundary_sample': res[3], 'frontier_sample': res[4],
                                 'emit_sample': res[5],
                                 'boundary_ms': round(res[3] / 16.0, 1),
                                 'emit_ms': round(res[5] / 16.0, 1)})
            bounds[label] = hits
        out[gid] = {'source': sid, 'frames': [a, b], 'case_episodes': case_eps,
                    'canonical': 'frontier_sweep.simulate_episode first-event return',
                    'bounds': bounds}
    return out


def main() -> None:
    from experiments.psem_state_corrected_adaptation_gate import h_postprocess as hp
    pa = load_pa()
    v = hp.load_validated_export(EXP121 / 'export/gpu_export')
    cal_f0, cal_cand, _ = hp.fit_calib_from_export(v['calib'])
    members = {sid: hp.prepare_dev_member(sid, v['dev'][sid], cal_f0, cal_cand)
               for sid in sorted(v['dev'])}
    cases = {}
    for gid, sid, a, b, d in CASES:
        m = members[sid]
        dev = m['dev']
        idx = np.arange(a, b + 1)
        st = np.asarray(dev.starts).reshape(-1)
        en = np.asarray(dev.ends).reshape(-1)
        sp = np.asarray(dev.speech_present).reshape(-1).astype(bool)
        mk = np.asarray(dev.masked).reshape(-1).astype(bool)
        ep = np.asarray(dev.episode_ids).reshape(-1)
        ov = np.asarray(dev.overlap).reshape(-1)
        tgt = np.asarray(m['target'])
        cp = np.asarray(m['cand_raw_prob'])
        refs = sorted(int(e.boundary_source_sample) for e in dev.reference.events)
        s0, e1 = int(st[a]), int(en[b])
        inside = [r for r in refs if s0 <= r <= e1]
        near = min([min(abs(s0 - r), abs(e1 - r)) for r in refs]) if refs else None
        rC = pa.classify_runs(m, pa.positive_runs(m, TH_H_C, HORIZON))
        rM = pa.classify_runs(m, pa.positive_runs(m, TH_H_M, HORIZON))
        mF0 = dict(m)
        mF0['cand_raw_prob'] = m['f0_prob']
        rF = pa.classify_runs(m, pa.positive_runs(mF0, TH_F0, HORIZON))

        def covering(rows):
            ovr = []
            for r in rows:
                rs = int(r['boundary_source_sample'])
                re_ = rs + int(round(float(r['duration_ms']) * 16))
                if rs <= e1 and re_ >= s0:
                    ovr.append({'boundary_sample': rs, 'duration_ms': r['duration_ms'],
                                'matched': bool(r['matched']), 'episode': r['episode_id']})
            return ovr

        # episode-split diagnostic: longest H-C-high contiguous stretch vs episode ids
        hi = (cp[idx] >= TH_H_C) & sp[idx]
        eps = [str(x) for x in ep[idx]]
        cases[gid] = {
            'source': sid, 'frames': [a, b], 'direction': d,
            'target_positive_frames': int((tgt[idx] > 0).sum()),
            'speech_frac': round(float(sp[idx].mean()), 3),
            'masked_frac': round(float(mk[idx].mean()), 3),
            'overlap_frac': round(float(np.asarray(ov[idx]).mean()), 3),
            'speakers_in_span': sorted(set(str(x) for x in np.asarray(dev.episode_speakers).reshape(-1)[idx])),
            'reference_inside_detail': [
                {'boundary_sample': int(e.boundary_source_sample),
                 'anchor_id': str(e.anchor_id),
                 'anchor_episode_id': str(e.anchor_episode_id),
                 'frontier_sample': int(e.model_evidence_frontier_sample),
                 'decoder_emit_sample': int(e.decoder_emit_sample)}
                for e in dev.reference.events if s0 <= int(e.boundary_source_sample) <= e1][:8],
            'n_episodes': len(set(eps)),
            'high_and_speech_frames': int(hi.sum()),
            'high_episodes': sorted(set(e for e, h in zip(eps, hi) if h)),
            'src_span_ms': [round(s0 / 16, 1), round(e1 / 16, 1)],
            'reference_inside': len(inside),
            'reference_near_edge_ms': round(near / 16, 1) if near is not None else None,
            'H100C_runs_covering': covering(rC)[:4],
            'H100M_runs_covering': covering(rM)[:4],
            'F0_runs_covering': covering(rF)[:4],
        }
    # Phase C: same-threshold O/C from existing rev3 MAIN calibration rows (no rescore)
    def cal_rows(p):
        return [json.loads(line) for line in open(REV3 / p, encoding='utf-8') if line.strip()]
    o = {r['tau']: r for r in cal_rows('main/results_repaired_v3/firered_O_calibration.jsonl')}
    c = {r['tau']: r for r in cal_rows('main/results_repaired_v3/firered_C_calibration.jsonl')}
    keep = ('tau', 'n_keep', 'n_cut', 'false_cuts', 'missed', 'missed_rate',
            'contamination_s_per_speech_h')
    rev3 = {'O_at_0.05': {k: o[0.05][k] for k in keep},
            'C_at_0.05': {k: c[0.05][k] for k in keep},
            'O_at_selected_0.85': {k: o[0.85][k] for k in keep},
            'C_at_selected_0.05': {k: c[0.05][k] for k in keep}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'case_enrichment.json').write_text(
        json.dumps(cases, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'rev3_same_threshold.json').write_text(
        json.dumps(rev3, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'text_diff.json').write_text(
        json.dumps(text_diff(members), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'owner_alignment.json').write_text(
        json.dumps(owner_alignment(), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'emission_bounds.json').write_text(
        json.dumps(emission_bounds(members), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'cases={len(cases)} rev3_keys={sorted(rev3)}')


if __name__ == '__main__':
    main()
