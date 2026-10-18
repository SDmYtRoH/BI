"""RehaPiano week 2: timestamp-aware, participant-level exploratory analysis.
Run: python analysis.py [--project /path/to/folder/containing/RehaPiano]  (default: <repo>/data)
Only derived files under this script's directory are written.
"""
from pathlib import Path
import argparse, base64, hashlib, io, json, pickle, platform, re, warnings
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
FIG = HERE / 'figures'
DATA = HERE.parents[1] / 'data'  # gitignored: raw recordings and the questionnaire snapshot
FS = 100.0
CHANNELS = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12])
STEPS = ['d2e6b7bd-a952-4652-afe3-9888e29ac205', '5867ebbc-a463-4d20-8b20-50df7c64ace7']
BANDS = {'slow': (.2, 3), 'boundary': (3, 4), 'tremor': (4, 15.00001),
         'tremor_4_12': (4, 12), 'upper': (12, 15.00001),
         'sensitivity_3': (3, 15.00001), 'sensitivity_3p5': (3.5, 15.00001)}
LEVEL_FLOOR = .002  # relative device units; heuristic, not a calibrated noise floor
MIN_OCCUPANCY = .2
MIN_RELATIVE_HAND_LEVEL = .1
MAX_GAP_SECONDS = .05
COLORS = {'patient': '#cf5c36', 'reference': '#2377a4'}
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'font.size': 10,
 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
 'grid.alpha': .18, 'figure.facecolor': 'white'})

class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Class loading is disabled')
    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent objects are disabled')

def load_raw(path):
    payload = json.loads(Path(path).read_text())['payload']
    packed = zlib_decompress(base64.b64decode(payload['raw'], validate=True))
    iterations = DataOnlyUnpickler(io.BytesIO(packed)).load()
    if not isinstance(iterations, (list, tuple)):
        raise ValueError('Expected a sequence of iterations')
    result = []
    for iteration in iterations:
        if not isinstance(iteration, (list, tuple)):
            raise ValueError('Invalid iteration')
        samples = [s for s in iteration if s is not None]
        if not samples or any(not isinstance(s, (bytes, bytearray)) or len(s) != 72 for s in samples):
            raise ValueError('Expected 72-byte samples')
        a = np.frombuffer(b''.join(samples), dtype=[('t', '<u8'), ('x', '<f4', (16,))])
        result.append((a['t'].astype(np.int64), a['x'][:, CHANNELS].astype(float), len(iteration)-len(samples)))
    return payload, result

def zlib_decompress(data):
    import zlib
    return zlib.decompress(data)

def prepare(t_ms, x):
    """Average duplicate timestamps, interpolate short gaps, anti-alias 1000 ->100 Hz.
    Large gaps split segments. Invalid channels remain NaN, never become zeros.
    """
    t_ms = np.asarray(t_ms, dtype=np.int64)
    if np.any(np.diff(t_ms) < 0):
        raise ValueError('Backward timestamp: reject rather than reorder')
    dt = np.diff(t_ms)
    duration = (t_ms[-1]-t_ms[0])/1000
    if duration <= 0 or not 50 <= (len(t_ms)-1)/duration <= 2000:
        raise ValueError('Implausible timestamp scale / sample rate')
    frame = pd.DataFrame(x, index=(t_ms-t_ms[0])/1000).replace([np.inf, -np.inf], np.nan)
    frame = frame.groupby(level=0).mean()
    ts, vals = frame.index.to_numpy(), frame.to_numpy()
    segments = []
    bounds = np.r_[0, np.where(np.diff(ts) > MAX_GAP_SECONDS)[0]+1, len(ts)]
    for start, end in zip(bounds[:-1], bounds[1:]):
        tt, xx = ts[start:end], vals[start:end]
        if len(tt)<2 or tt[-1]-tt[0]<3:
            continue
        grid = np.arange(np.ceil(tt[0]*1000), np.floor(tt[-1]*1000)+1)/1000
        reg = np.full((len(grid), x.shape[1]), np.nan)
        for c in range(x.shape[1]):
            ok = np.isfinite(xx[:, c])
            if ok.sum()<2 or 1-ok.mean()>.01:
                continue
            valid_t = tt[ok]
            if np.max(np.diff(valid_t))>MAX_GAP_SECONDS or valid_t[0]-tt[0]>.01 or tt[-1]-valid_t[-1]>.01:
                continue
            reg[:, c] = np.interp(grid, valid_t, xx[ok,c])
        down = signal.resample_poly(reg, 1, 10, axis=0, padtype='line')
        times = grid[0]+np.arange(len(down))/FS
        segments.append((times, down))
    qa = dict(samples=len(x), duration_s=duration, effective_packet_rate_hz=(len(x)-1)/duration,
              duplicate_timestamps=int((dt==0).sum()), backward_timestamps=int((dt<0).sum()),
              max_gap_ms=int(dt.max()), nonfinite_fraction=float((~np.isfinite(x)).mean()),
              segment_count=len(segments))
    return segments, qa

def spectrum(x, fs=FS):
    x = np.asarray(x)
    if x.ndim == 2:
        ok = np.isfinite(x).all(axis=0)
        f = np.fft.rfftfreq(len(x), 1/fs)
        p = np.full((len(f), x.shape[1]), np.nan)
        if ok.any():
            _,p[:,ok] = signal.periodogram(x[:,ok], fs=fs, window='hann', detrend='linear', scaling='density', axis=0)
        return f,p
    f,p = signal.periodogram(x, fs=fs, window='hann', detrend='linear', scaling='density', axis=0)
    return f,p

def spectral_features(f, p):
    """Integrate one-sided PSD with a rectangular DFT-bin sum; no zero padding."""
    df = f[1]-f[0]
    m = (f>=4)&(f<=15)
    pt = p[m]; ft=f[m]; total=float(pt.sum())
    out = {}
    for name,(lo,hi) in BANDS.items():
        power=float(p[(f>=lo)&(f<hi)].sum()*df)
        out[name+'_power']=power
        out[name+'_rms']=np.sqrt(max(power,0))
    whole=out['slow_power']+out['boundary_power']+out['tremor_power']
    out['tremor_fraction']=out['tremor_power']/whole if whole>1e-16 else np.nan
    out['tremor_to_slow']=out['tremor_power']/out['slow_power'] if out['slow_power']>1e-16 else np.nan
    if total>1e-16:
        weights=pt/total
        centroid=float(ft@weights)
        out.update(centroid_hz=centroid, spectral_spread_hz=float(np.sqrt(((ft-centroid)**2)@weights)),
                   peak_hz=float(ft[np.argmax(pt)]), peak_to_median_psd=float(pt.max()/max(np.median(pt),1e-20)),
                   spectral_entropy=float(-np.sum(weights*np.log(weights+1e-30))/np.log(len(weights))))
    else:
        out.update({k:np.nan for k in ['centroid_hz','spectral_spread_hz','peak_hz','peak_to_median_psd','spectral_entropy']})
    return out

def activity(x, floor=LEVEL_FLOOR, ratio=MIN_RELATIVE_HAND_LEVEL):
    """Heuristic low-use flag. Absolute force is used because channels can be signed."""
    finite=np.isfinite(x).all(axis=0)
    safe=np.where(np.isfinite(x),x,0)
    levels=np.median(np.abs(safe),axis=0)
    occupancy=(np.abs(safe)>floor).mean(axis=0)
    sd=np.std(safe,axis=0)
    clip=(np.abs(safe)>=.999).mean(axis=0)
    eligible=finite&(levels>=floor)&(occupancy>=MIN_OCCUPANCY)&(sd>1e-7)&(clip<.01)
    hand_level=np.array([np.sqrt(np.mean(levels[:5]**2)),np.sqrt(np.mean(levels[5:]**2))])
    hand_ok=[]
    reasons=[]
    for h,ix in enumerate([np.arange(5),np.arange(5,10)]):
        why=[]
        if not finite[ix].any():why.append('invalid_channels')
        if not eligible[ix].any():why.append('no_eligible_finger')
        if hand_level[h]<floor:why.append('low_absolute_force')
        if hand_level[h]<ratio*max(hand_level):why.append('low_relative_force')
        hand_ok.append(not why)
        reasons.append('|'.join(why) or 'usable')
    return eligible, hand_ok, levels, occupancy, hand_level, reasons, clip

def build_manifest(project):
    root=project/'RehaPiano'
    lookup={}
    for path in sorted(root.rglob('results_*')):
        if not path.is_file():continue
        with path.open('rb') as h:head=h.read(1000).decode('utf-8',errors='ignore')
        e=re.search(r'"examination_id"\s*:\s*"([^"]+)"',head)
        s=re.search(r'"step_id"\s*:\s*"([^"]+)"',head)
        if e and s:lookup.setdefault((e[1],s[1]),[]).append(path)
    rows=[]
    for filename in ['dataset_patients.csv','dataset_tuke.csv','dataset_99.csv']:
        d=pd.read_csv(root/'datasets'/filename,sep=';',dtype={'patient_id':str}).drop_duplicates()
        d=d[d.step_id.isin(STEPS)]
        for row in d.to_dict('records'):
            paths=lookup.get((row['measurement_id'],row['step_id']),[])
            paths.sort(key=lambda p:('22_08_2025' not in str(p),len(str(p)),str(p)))
            # Exact content duplicates are harmless; divergent copies require review.
            digests={hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            row.update(cohort='patient' if filename=='dataset_patients.csv' else 'reference',
                       source_cohort=filename, source_path=str(paths[0].relative_to(project)) if paths else '',
                       duplicate_files=len(paths), divergent_copies=len(digests)>1,
                       sha256=next(iter(digests)) if len(digests)==1 else '')
            rows.append(row)
    return pd.DataFrame(rows).drop(columns=['additional_data'])

def questionnaire_labels(manifest):
    q=pd.read_csv(DATA/'sources/questionnaire.csv',dtype=str)
    ids=q.iloc[:,2].str.strip()
    records=[]
    for pid,g in manifest.groupby('patient_id'):
        matches=q.loc[ids==str(pid)]
        answers=matches.iloc[:,7].dropna().str.strip().unique().tolist()
        health='no_reported_condition' if answers==['Nie'] else ('reported_condition' if answers==['Áno'] else 'unknown_or_conflicting')
        if len(matches)>1:health='duplicate_questionnaire_id'
        cohort=g.cohort.iloc[0]
        status='patient_cohort' if cohort=='patient' else health
        records.append(dict(patient_id=pid,cohort=cohort,label_status=status,
                            eligible_comparison=cohort=='patient' or health=='no_reported_condition',
                            questionnaire_matches=len(matches),age=g.age.iloc[0],gender=g.gender.iloc[0],
                            source_cohort=g.source_cohort.iloc[0]))
    return pd.DataFrame(records)

def analyse(project, manifest):
    fingers=[];hands=[];quality=[];usage=[];errors=[];demos={};sensitivity=[]
    demo_ids={'33eacf6b-6231-4dd5-863e-c3aa1e95d79b','d482c2df-d0be-4c89-bbdf-da1b2308cedc',
              '340209e1-ecf2-4446-a52a-16c5a830241b','de72d25f-8f60-4820-a784-68c0f3d0b7a3'}
    for ix,row in enumerate(manifest.to_dict('records')):
        base={k:row[k] for k in ['patient_id','measurement_id','step_id','cohort','source_cohort']}
        if not row['source_path'] or row['divergent_copies']:
            errors.append({**base,'reason':'missing file or divergent copies'});continue
        try:
            payload,iterations=load_raw(project/row['source_path'])
            assert payload['examination_id']==row['measurement_id'] and payload['step_id']==row['step_id']
            for iteration,(t,x,n_none) in enumerate(iterations,1):
                b={**base,'iteration':iteration}
                try:segments,qa=prepare(t,x)
                except ValueError as exc:
                    errors.append({**b,'reason':str(exc)});continue
                quality.append({**b,**qa,'none_samples':n_none})
                for segno,(tt,xx) in enumerate(segments):
                    # Full iterations are ~16 s. Use 3–13 s for the primary comparison.
                    # Avoid filter/resampling edges and never bridge a temporal gap.
                    lo=max(3.,tt[0]+.5);hi=min(15.,tt[-1]-.5)
                    if hi-lo<3:continue
                    if row['measurement_id'] in demo_ids:
                        demos[(row['measurement_id'],iteration,segno)]=(tt,xx)
                    for window_s in [3,5,10]:
                        starts=np.arange(lo,hi-window_s+1e-8,window_s)
                        for wi,start in enumerate(starts):
                            k=int(np.searchsorted(tt,start-1e-9));block=xx[k:k+int(window_s*FS)]
                            if len(block)!=int(window_s*FS):continue
                            wbase={**b,'segment':segno,'window_s':window_s,'window_index':wi,'start_s':float(tt[k])}
                            eligible,hok,levels,occ,hl,reasons,clip=activity(block)
                            f,p=spectrum(block)
                            for c in range(10):
                                cb={**wbase,'hand':'left' if c<5 else 'right','channel':int(CHANNELS[c]),
                                    'eligible_channel':bool(eligible[c]),'median_abs_force':levels[c],
                                    'active_fraction':occ[c],'clip_fraction':clip[c]}
                                if np.isfinite(p[:,c]).all():cb.update(spectral_features(f,p[:,c]))
                                fingers.append(cb)
                            for h,(hand,ci) in enumerate([('left',np.arange(5)),('right',np.arange(5,10))]):
                                active=ci[eligible[ci]]
                                hb={**wbase,'hand':hand,'usable':bool(hok[h]),'reason':reasons[h],
                                    'eligible_fingers':len(active),'force_level':hl[h],
                                    'relative_hand_level':hl[h]/max(hl) if max(hl)>0 else np.nan}
                                if hok[h]:
                                    hp=p[:,active].mean(axis=1)
                                    hb.update(spectral_features(f,hp))
                                    hf,meanp=spectrum(block[:,active].mean(axis=1))
                                    hb['mean_signal_to_mean_power_ratio']=spectral_features(hf,meanp)['tremor_power']/max(hb['tremor_power'],1e-20)
                                hands.append(hb)
                                if window_s==5:usage.append(hb.copy())
                            if window_s==5:
                                for floor in [.001,.002,.005]:
                                    for ratio in [.05,.1,.2]:
                                        _,ok,_,_,_,_,_=activity(block,floor,ratio)
                                        for hand,h in [('left',0),('right',1)]:
                                            sensitivity.append({**wbase,'hand':hand,'floor':floor,'relative_threshold':ratio,'usable':ok[h]})
        except Exception as exc:
            errors.append({**base,'reason':repr(exc)})
        if (ix+1)%25==0:print(f'Processed {ix+1}/{len(manifest)} sessions',flush=True)
    return [pd.DataFrame(z) for z in [fingers,hands,quality,usage,errors,sensitivity]],demos

def aggregate(hands, manifest, labels):
    metric_cols=[c for c in hands if c.endswith(('_power','_rms'))]+['centroid_hz','spectral_spread_hz','peak_hz','tremor_fraction','spectral_entropy','peak_to_median_psd','mean_signal_to_mean_power_ratio']
    valid=hands[hands.usable].copy()
    # Equal weighting: windows within iteration, iterations within session, sessions within person.
    iter_means=valid.groupby(['patient_id','measurement_id','iteration','hand','window_s'])[metric_cols].mean().reset_index()
    session_means=iter_means.groupby(['patient_id','measurement_id','hand','window_s'])[metric_cols].mean().reset_index()
    person= session_means.groupby(['patient_id','hand','window_s'])[metric_cols].mean().reset_index()
    stats=valid.groupby(['patient_id','hand','window_s']).agg(valid_windows=('tremor_rms','size'),
         tremor_rms_window_sd=('tremor_rms','std'),centroid_window_sd_hz=('centroid_hz','std'),peak_window_sd_hz=('peak_hz','std')).reset_index()
    eligibility=hands.groupby(['patient_id','hand','window_s']).agg(total_windows=('usable','size'),usable_fraction=('usable','mean'),
                   median_force_level=('force_level','median'),median_relative_hand_level=('relative_hand_level','median')).reset_index()
    skeleton=pd.MultiIndex.from_product([manifest.patient_id.unique(),['left','right'],[3,5,10]],names=['patient_id','hand','window_s']).to_frame(index=False)
    person=skeleton.merge(person,how='left').merge(stats,how='left').merge(eligibility,how='left').merge(labels,on='patient_id',how='left')
    person['valid_windows']=person.valid_windows.fillna(0).astype(int)
    person['status']=np.where(person.valid_windows>=4,'usable',np.where(person.valid_windows>0,'limited_windows','unassessable'))
    return person,iter_means,session_means

def auc(y,score):
    from scipy.stats import rankdata
    n1=y.sum();n0=len(y)-n1
    return float((rankdata(score)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def baseline(person):
    """Predefined 5-fold participant split; training-only univariate threshold.
    Cohort discrimination only. Not a validated tremor detector.
    """
    p=person[(person.window_s==5)&(person.valid_windows>=4)]
    wide=p.pivot(index='patient_id',columns='hand',values='tremor_rms')
    score=wide.max(axis=1).rename('tremor_rms_max_hand')
    meta=person.drop_duplicates('patient_id').set_index('patient_id')
    data=meta.join(score).reset_index()
    data=data[data.eligible_comparison & data.tremor_rms_max_hand.notna()].copy()
    data['y']=(data.cohort=='patient').astype(int)
    rng=np.random.default_rng(20261010)
    data['fold']=-1
    for y in [0,1]:
        idx=data.index[data.y==y].to_numpy();rng.shuffle(idx)
        data.loc[idx,'fold']=np.arange(len(idx))%5
    metrics=[];predictions=[]
    for feature in ['tremor_rms_max_hand','age']:
        pred=np.zeros(len(data),int);scores=np.zeros(len(data));thresholds=[]
        arr=data[feature].to_numpy();target=data.y.to_numpy()
        for fold in range(5):
            train=data.fold.to_numpy()!=fold;test=~train
            vals=np.unique(arr[train]);cuts=np.r_[-np.inf,(vals[:-1]+vals[1:])/2,np.inf]
            best=(-1,None,None)
            for direction in [1,-1]:
                for cut in cuts:
                    yh=(direction*arr[train]>=direction*cut).astype(int)
                    yy=target[train]
                    ba=.5*((yh[yy==1]==1).mean()+(yh[yy==0]==0).mean())
                    if ba>best[0]:best=(ba,cut,direction)
            _,cut,direction=best
            pred[test]=(direction*arr[test]>=direction*cut)
            # AUC calculated from original feature below; thresholds vary by fold.
            scores[test]=direction*arr[test]
            thresholds.append(dict(feature=feature,fold=fold,threshold=float(cut),direction=direction,training_balanced_accuracy=float(best[0])))
        y=target;tp=int(((pred==1)&(y==1)).sum());tn=int(((pred==0)&(y==0)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
        metrics.append(dict(feature=feature,n=len(y),patients=int(y.sum()),references=int((1-y).sum()),
                            sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
                            tp=tp,tn=tn,fp=fp,fn=fn,descriptive_raw_feature_auc=auc(y,arr)))
        for j,(_,r) in enumerate(data.iterrows()):predictions.append(dict(patient_id=r.patient_id,feature=feature,fold=r.fold,y=r.y,prediction=int(pred[j]),feature_value=arr[j]))
        pd.DataFrame(thresholds).to_csv(OUT/f'baseline_thresholds_{feature}.csv',index=False)
    return pd.DataFrame(metrics),pd.DataFrame(predictions)

def savefig(fig,name):
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def plot_results(demos,person,hands):
    # Reuse the pair from task_1, without selecting on tremor magnitude.
    ids=[('reference','d482c2df-d0be-4c89-bbdf-da1b2308cedc','992601'),('patient','33eacf6b-6231-4dd5-863e-c3aa1e95d79b','UQIS-DXIH')]
    pair_labels={'reference':'Healthy participant','patient':'Sick participant'}
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
    specs={}
    for col,(cohort,exam,pid) in enumerate(ids):
        t,x=demos[(exam,2,0)]
        low=signal.sosfiltfilt(signal.butter(4,3,fs=FS,output='sos'),x,axis=0)
        fast=signal.sosfiltfilt(signal.butter(4,[4,15],btype='bandpass',fs=FS,output='sos'),x,axis=0)
        # Fixed physical channel 10, chosen in advance, not by maximum tremor.
        c=7;m=(t>=3)&(t<13)
        axs[0,col].plot(t[m],x[m,c],color='#aaa',lw=.7,label='Measured force')
        axs[0,col].plot(t[m],low[m,c],color=COLORS[cohort],label='Slow component (<3 Hz)')
        axs[1,col].plot(t[m],fast[m,c],color=COLORS[cohort],label='Candidate tremor (4–15 Hz)')
        axs[0,col].set_title(f'{pair_labels[cohort]} · one fixed right-hand sensor')
        for a in axs[:,col]:a.legend(fontsize=8);a.set_ylabel('Force [device units]')
        axs[1,col].set_xlabel('Time within attempt 2 [s]')
    for r in [0,1]:
        lower=min(axs[r,c].get_ylim()[0] for c in [0,1]);upper=max(axs[r,c].get_ylim()[1] for c in [0,1])
        for c in [0,1]:axs[r,c].set_ylim(lower,upper)
    fig.suptitle('Separate slow force variation from faster oscillations\nWorking frequency bands; the filter cannot identify intention',fontsize=13)
    fig.tight_layout();savefig(fig,'01_decomposition')
    fig,axs=plt.subplots(3,2,figsize=(12,9),sharex=True,sharey=True)
    fftrows=[]
    for r,w in enumerate([3,5,10]):
        for col,hand in enumerate(['left','right']):
            ci=np.arange(5) if col==0 else np.arange(5,10)
            for cohort,exam,pid in ids:
                t,x=demos[(exam,2,0)];k=np.searchsorted(t,3);b=x[k:k+int(w*FS)]
                elig,ok,*_=activity(b);active=ci[elig[ci]]
                f,p=spectrum(b[:,active]);p=p.mean(axis=1)
                axs[r,col].semilogy(f,p,color=COLORS[cohort],label=pair_labels[cohort])
                fftrows.append(dict(patient_id=pid,hand=hand,window_s=w,**spectral_features(f,p)))
            a=axs[r,col];a.axvspan(0,3,alpha=.06,color='blue');a.axvspan(3,4,alpha=.15,color='grey');a.axvspan(4,15,alpha=.06,color='orange')
            a.set_title(f'{hand.title()} hand · {w} second view\nFrequency detail: {1/w:.3f} Hz');a.set_xlim(0,15);a.set_ylim(1e-10,1e-2)
            a.set_ylabel('Oscillation power (log scale)')
    for a in axs[-1]:a.set_xlabel('Frequency [Hz]')
    axs[0,0].legend();fig.suptitle('How the spectrum changes when we observe 3, 5 or 10 seconds\nBlue area = slow force changes · grey = uncertain · orange = faster oscillations',fontsize=13)
    fig.tight_layout();savefig(fig,'02_window_comparison')
    pd.DataFrame(fftrows).to_csv(OUT/'example_window_comparison.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
    for col,(cohort,exam,pid) in enumerate(ids):
        t,x=demos[(exam,2,0)];m=(t>=3)&(t<15);x=x[m];elig,ok,*_=activity(x)
        for r,ci in enumerate([np.arange(5),np.arange(5,10)]):
            active=ci[elig[ci]]
            f,st,p=signal.spectrogram(x[:,active],fs=FS,nperseg=200,noverlap=180,window='hann',detrend='linear',scaling='density',axis=0)
            db=10*np.log10(np.maximum(p.mean(axis=1),1e-20));mask=f<=15
            specs[(r,col)]=(f[mask],st+3,db[mask])
    vals=np.concatenate([v[2].ravel() for v in specs.values()]);vmin,vmax=np.percentile(vals,[2,99])
    for (r,col),(f,t,db) in specs.items():
        im=axs[r,col].pcolormesh(t,f,db,shading='auto',cmap='magma',vmin=vmin,vmax=vmax)
        axs[r,col].axhline(3,color='white',ls=':',lw=.8);axs[r,col].axhline(4,color='white',ls='--',lw=.8)
        axs[r,col].set_title(f'{pair_labels[ids[col][0]]} · {["left","right"][r]} hand')
        axs[r,col].set_ylabel('Frequency [Hz]');axs[r,col].set_xlabel('Time within attempt 2 [s]');axs[r,col].grid(False)
    fig.colorbar(im,ax=axs,label='Oscillation strength (brighter = stronger)',shrink=.85)
    fig.suptitle('When do slow and faster force changes occur?\nBelow 3 Hz = slow control · 3–4 Hz = uncertain · 4–15 Hz = possible tremor',fontsize=13)
    savefig(fig,'03_spectrograms')
    # All repetitions of the primary pair, both hands.
    fig,axs=plt.subplots(4,4,figsize=(15,10),sharex=True,sharey=True,layout='constrained')
    for g,(cohort,exam,pid) in enumerate(ids):
        for it in range(1,5):
            t,x=demos[(exam,it,0)];m=(t>=3)&(t<15);x=x[m];elig,ok,*_=activity(x)
            for h,ci in enumerate([np.arange(5),np.arange(5,10)]):
                ax=axs[2*g+h,it-1];active=ci[elig[ci]]
                if not len(active):ax.text(.5,.5,'Low use / unassessable',transform=ax.transAxes,ha='center');continue
                f,st,p=signal.spectrogram(x[:,active],fs=FS,nperseg=200,noverlap=180,window='hann',detrend='linear',scaling='density',axis=0)
                db=10*np.log10(np.maximum(p.mean(axis=1),1e-20));m=f<=15
                im=ax.pcolormesh(st+3,f[m],db[m],shading='auto',cmap='magma',vmin=vmin,vmax=vmax);ax.grid(False)
                ax.axhline(3,color='white',ls=':',lw=.7);ax.axhline(4,color='white',ls='--',lw=.7)
                ax.set_title(f'{pair_labels[cohort]} · {["left","right"][h]} hand · attempt {it}',fontsize=9)
                if it==1:ax.set_ylabel('Frequency [Hz]')
                if 2*g+h==3:ax.set_xlabel('Time [s]')
    fig.colorbar(im,ax=axs,label='Oscillation strength (brighter = stronger)',shrink=.7)
    fig.suptitle('Are the patterns repeated across all four attempts?\nTop two rows = healthy participant · bottom two rows = sick participant · same colour scale',fontsize=13)
    savefig(fig,'04_all_repetitions')
    p=person[(person.window_s==5)&(person.status!='unassessable')]
    fig,axs=plt.subplots(1,3,figsize=(13,4.5))
    rng=np.random.default_rng(13)
    for ax,metric,title,log in zip(axs,['tremor_rms','centroid_hz','usable_fraction'],['Faster-oscillation amplitude (4–15 Hz)','Typical faster-oscillation frequency','Share of usable time windows'],[True,False,False]):
        for k,cohort in enumerate(['reference','patient']):
            q=p[p.cohort==cohort].groupby('patient_id')[metric].mean().dropna()
            ax.scatter(k+rng.uniform(-.15,.15,len(q)),q,s=20,color=COLORS[cohort],alpha=.65)
            ax.plot([k-.22,k+.22],[q.median()]*2,color='black',lw=2)
        ax.set_xticks([0,1],['Healthy group','Sick group']);ax.set_title(title);ax.set_ylabel('Relative force units' if log else ('Frequency [Hz]' if metric=='centroid_hz' else 'Share from 0 to 1'))
        if log:ax.set_yscale('log')
    fig.suptitle('How the measured features differ between groups\nOne dot = one person · black line = group median',fontsize=12)
    fig.tight_layout();savefig(fig,'05_cohort_features')
    # A concrete hand-underuse example selected by force level only.
    u=hands[(hands.window_s==5)&(hands.cohort=='patient')]
    ratios=u.groupby(['patient_id','hand']).relative_hand_level.median().sort_values()
    flagged=u.loc[~u.usable].sort_values('relative_hand_level')
    chosen=flagged.patient_id.iloc[0] if len(flagged) else ratios.index[0][0]
    sub=u[u.patient_id==chosen]
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for hand,color in [('left','#2377a4'),('right','#cf5c36')]:
        d=sub[sub.hand==hand].reset_index(drop=True)
        axs[0].plot(np.arange(len(d)),d.force_level,'o-',label=hand,color=color)
        axs[1].plot(np.arange(len(d)),d.usable.astype(int),'o-',label=hand,color=color)
    axs[0].axhline(LEVEL_FLOOR,color='grey',ls='--',label='Absolute floor')
    axs[0].set_yscale('log');axs[0].set_ylabel('Median-based force level [device units]');axs[0].legend()
    axs[1].set_yticks([0,1],['Flagged','Usable']);axs[1].legend()
    for a in axs:a.set_xlabel('5 s window index (sessions and repetitions kept separate)')
    fig.suptitle(f'Low-use example: {chosen} · missing evidence must not become zero tremor')
    fig.tight_layout();savefig(fig,'06_low_use_hand')
    return chosen

def validate_methods():
    t=np.arange(2000)/FS
    x=.03+.01*np.sin(2*np.pi*1*t)+.002*np.sin(2*np.pi*6*t)
    f,p=spectrum(x);m=spectral_features(f,p)
    assert abs(m['tremor_rms']-.002/np.sqrt(2))<1e-5
    assert abs(m['slow_rms']-.01/np.sqrt(2))<1e-4
    assert m['peak_hz']==6
    pair=np.c_[np.sin(2*np.pi*6*t),-np.sin(2*np.pi*6*t)]
    _,sp=spectrum(pair);_,av=spectrum(pair.mean(axis=1))
    assert sp.mean(axis=1).sum()>.1 and av.sum()<1e-15
    xx=np.zeros((1000,10));xx[:,5:]=.02+np.sin(2*np.pi*6*np.arange(1000)/FS)[:,None]*.001
    _,ok,*_=activity(xx);assert ok==[False,True]
    _,zero=spectrum(np.zeros(1000));assert np.isnan(spectral_features(*spectrum(np.zeros(1000)))['centroid_hz'])
    # Verify filtering retains 6 Hz and suppresses the 1 Hz component.
    b=signal.sosfiltfilt(signal.butter(4,[4,15],btype='bandpass',fs=FS,output='sos'),x)
    ff,pp=spectrum(b[300:-300]);assert np.sqrt(pp[(ff>.5)&(ff<2)].sum()*(ff[1]-ff[0]))<1e-5
    # A 200 ms gap must create two segments, not an interpolated bridge.
    tt=np.r_[np.arange(4000),np.arange(4200,8200)]
    segs,_=prepare(tt,np.ones((len(tt),10))*.02);assert len(segs)==2
    # A timestamp reset must be rejected.
    try:prepare(np.array([10,11,9]),np.ones((3,10)))
    except ValueError:pass
    else:raise AssertionError('Backward timestamps accepted')
    # Packet jitter, duplicates and a short missing burst preserve an injected peak.
    rng=np.random.default_rng(4);tt=np.maximum.accumulate(np.arange(16000)+rng.integers(-1,2,16000));tt=tt-tt[0]
    xx=np.tile((.03+.003*np.sin(2*np.pi*6*tt/1000))[:,None],(1,10));xx[500:504]=np.nan
    segs,_=prepare(tt,xx);assert len(segs)==1
    f,p=spectrum(segs[0][1][300:1300,0]);assert abs(spectral_features(f,p)['peak_hz']-6)<.11
    bad=np.c_[np.full(1000,np.nan),np.sin(2*np.pi*6*np.arange(1000)/FS)]
    _,bp=spectrum(bad);assert np.isnan(bp[:,0]).all() and np.isfinite(bp[:,1]).all()
    return dict(status='passed',checks=['known sine amplitudes and frequency','opposite-phase cancellation','unused hand exclusion','zero-power undefined frequency','filter separation','large-gap splitting','timestamp reset rejection','jitter and NaN resampling','invalid channel isolation'])

def main(project):
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    checks=validate_methods();print('Method checks passed',flush=True)
    manifest=build_manifest(project);manifest.to_csv(OUT/'recording_manifest.csv',index=False)
    labels=questionnaire_labels(manifest);labels.to_csv(OUT/'participant_labels.csv',index=False)
    (tables,demos)=analyse(project,manifest)
    names=['finger_windows','hand_windows','recording_quality','hand_usage','processing_errors','activity_sensitivity']
    for name,table in zip(names,tables):table.to_csv(OUT/(name+'.csv'),index=False)
    fingers,hands,quality,usage,errors,sensitivity=tables
    person,iters,sessions=aggregate(hands,manifest,labels)
    for name,table in [('participant_hand_features',person),('iteration_features',iters),('session_features',sessions)]:table.to_csv(OUT/(name+'.csv'),index=False)
    primary=person[person.window_s==5].copy()
    primary.to_csv(OUT/'participant_hand_features_primary.csv',index=False)
    metrics,predictions=baseline(person)
    metrics.to_csv(OUT/'baseline_metrics.csv',index=False);predictions.to_csv(OUT/'baseline_predictions.csv',index=False)
    low_use=plot_results(demos,person,hands)
    sensitivity.groupby(['floor','relative_threshold']).usable.agg(['mean','size']).to_csv(OUT/'activity_sensitivity_summary.csv')
    summary={'participants':int(manifest.patient_id.nunique()),'sessions':len(manifest),'cohort_people':labels.cohort.value_counts().to_dict(),
       'label_status':labels.label_status.value_counts().to_dict(),'quality_iterations':len(quality),'processing_errors':len(errors),
       'divergent_files':int(manifest.divergent_copies.sum()),'primary_hand_status':primary.status.value_counts().to_dict(),
       'primary_windows':int((hands.window_s==5).sum()),'primary_usable_windows':int(hands.loc[hands.window_s==5,'usable'].sum()),
       'low_use_example':low_use,'method_checks':checks,'baseline':metrics.to_dict('records'),
       'packet_rate_summary':quality.effective_packet_rate_hz.describe().to_dict(),
       'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
       'config':{'resampled_fs_hz':FS,'crop_start_s':3,'crop_end_s':15,'primary_window_s':5,'bands':BANDS,'level_floor':LEVEL_FLOOR,'relative_hand_floor':MIN_RELATIVE_HAND_LEVEL,'max_interpolated_gap_s':MAX_GAP_SECONDS}}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=DATA);args=ap.parse_args()
    main(args.project.resolve())
