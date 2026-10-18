"""Additional participant exports, sensitivity analysis and presentation figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import analysis as a

def enrich(project):
    out=a.OUT
    p=pd.read_csv(out/'participant_hand_features.csv',dtype={'patient_id':str})
    h=pd.read_csv(out/'hand_windows.csv',dtype={'patient_id':str})
    manifest=pd.read_csv(out/'recording_manifest.csv',dtype={'patient_id':str})
    labels=pd.read_csv(out/'participant_labels.csv',dtype={'patient_id':str})
    primary=p[p.window_s==5]
    # A compact, one-row-per-person table; side-specific missingness is retained.
    cols=['slow_rms','boundary_rms','tremor_rms','tremor_4_12_rms','upper_rms','centroid_hz','spectral_spread_hz',
          'peak_hz','tremor_fraction','tremor_rms_window_sd','centroid_window_sd_hz','peak_window_sd_hz',
          'valid_windows','total_windows','usable_fraction','status']
    wide=primary.pivot(index='patient_id',columns='hand',values=cols)
    wide.columns=[f'{hand}_{metric}' for metric,hand in wide.columns]
    wide=labels.set_index('patient_id').join(wide)
    wide['number_of_sessions']=manifest.groupby('patient_id').size()
    wide.to_csv(out/'participant_features.csv')
    # Recalculate the same participant features using only late 8–13 s windows.
    late=h[(h.window_s==5)&(h.start_s>=7.99)].copy()
    late_person,_,_=a.aggregate(late,manifest,labels)
    late_person=late_person[late_person.window_s==5]
    late_person.to_csv(out/'participant_features_late_windows.csv',index=False)
    # Baseline writes fold thresholds; preserve primary thresholds first.
    original={f.name:f.read_bytes() for f in out.glob('baseline_thresholds_*.csv')}
    late_metrics,late_pred=a.baseline(late_person)
    for f in out.glob('baseline_thresholds_*.csv'):
        (out/('late_'+f.name)).write_bytes(f.read_bytes())
    for name,content in original.items():(out/name).write_bytes(content)
    late_metrics.to_csv(out/'late_baseline_metrics.csv',index=False)
    late_pred.to_csv(out/'late_baseline_predictions.csv',index=False)
    early=h[(h.window_s==5)&(h.start_s<7.99)].copy()
    early_person,_,_=a.aggregate(early,manifest,labels)
    early_person=early_person[early_person.window_s==5]
    rows=[]
    for label,frame in [('primary: 3–8 and 8–13 s',primary),('early: 3–8 s',early_person),('late: 8–13 s',late_person)]:
        # Reduce to one equally weighted value per person before cohort summaries.
        pp=frame.groupby(['patient_id','cohort'])[['tremor_rms','slow_rms']].mean().reset_index()
        for cohort,g in pp.groupby('cohort'):
            rows.append(dict(interval=label,cohort=cohort,people=len(g),median_tremor_rms=g.tremor_rms.median(),median_slow_rms=g.slow_rms.median()))
    pd.DataFrame(rows).to_csv(out/'transient_sensitivity.csv',index=False)
    # Participant bootstrap of fixed out-of-fold predictions: descriptive uncertainty.
    pred=pd.read_csv(out/'baseline_predictions.csv');cis=[];rng=np.random.default_rng(31)
    for feature,g in pred.groupby('feature'):
        y=g.y.to_numpy();yh=g.prediction.to_numpy();vals=[]
        for _ in range(2000):
            ix=np.r_[rng.choice(np.where(y==1)[0],sum(y==1),replace=True),rng.choice(np.where(y==0)[0],sum(y==0),replace=True)]
            yy=y[ix];yp=yh[ix];vals.append(.5*((yp[yy==1]==1).mean()+(yp[yy==0]==0).mean()))
        lo,hi=np.quantile(vals,[.025,.975]);cis.append(dict(feature=feature,balanced_accuracy_low=lo,balanced_accuracy_high=hi))
    pd.DataFrame(cis).to_csv(out/'baseline_bootstrap_intervals.csv',index=False)
    # Fixed split confound inspection: no fitted age adjustment / no causal claim.
    from scipy.stats import spearmanr
    people=primary.pivot(index='patient_id',columns='hand',values='tremor_rms').max(axis=1).rename('score').to_frame().join(labels.set_index('patient_id'))
    eligible=people[people.eligible_comparison]
    rho,pvalue=spearmanr(eligible.age,eligible.score)
    age_subset=eligible[eligible.age>=50]
    age_summary=labels.groupby('cohort').age.agg(['count','median','min','max'])
    age_summary.to_csv(out/'age_summary.csv')
    confounds={'age_tremor_spearman_rho':float(rho),'age50plus_patients':int((age_subset.cohort=='patient').sum()),
               'age50plus_references':int((age_subset.cohort=='reference').sum()),
               'age50plus_descriptive_tremor_auc':a.auc((age_subset.cohort=='patient').astype(int).to_numpy(),age_subset.score.to_numpy()),
               'age_summary':age_summary.reset_index().to_dict('records')}
    (out/'confound_checks.json').write_text(json.dumps(confounds,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    e=early_person.set_index(['patient_id','hand']).tremor_rms
    l=late_person.set_index(['patient_id','hand']).tremor_rms
    paired=pd.concat([e.rename('early'),l.rename('late')],axis=1).groupby('patient_id').mean().join(labels.set_index('patient_id'))
    group_labels={'reference':'Healthy group','patient':'Sick group'}
    for cohort in ['reference','patient']:
        d=paired[paired.cohort==cohort]
        axs[0].scatter(d.early,d.late,s=20,alpha=.65,label=group_labels[cohort],color=a.COLORS[cohort])
        d=eligible[eligible.cohort==cohort]
        axs[1].scatter(d.age,d.score,s=22,alpha=.7,label=group_labels[cohort],color=a.COLORS[cohort])
    axs[0].plot([1e-5,.01],[1e-5,.01],'k--',lw=1);axs[0].set_xscale('log');axs[0].set_yscale('log')
    axs[0].set_xlabel('Faster-oscillation amplitude in first 5 seconds');axs[0].set_ylabel('Amplitude in next 5 seconds');axs[0].set_title('Does the beginning of the task look different?');axs[0].legend()
    axs[1].set_yscale('log');axs[1].set_xlabel('Age [years]');axs[1].set_ylabel('Faster-oscillation amplitude');axs[1].set_title(f'Older participants often show larger values (ρ = {rho:.2f})');axs[1].legend()
    fig.tight_layout();a.savefig(fig,'07_transients_and_age')
    # Existing illustrative pair: selected by the earlier notebook, not validation data.
    ids=[('reference','de72d25f-8f60-4820-a784-68c0f3d0b7a3','992612'),('patient','340209e1-ecf2-4446-a52a-16c5a830241b','ILHC-BZCP')]
    pair_labels={'reference':'Healthy participant','patient':'Sick participant'}
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ps=[];specs=[]
    for col,(cohort,exam,pid) in enumerate(ids):
        r=manifest[manifest.measurement_id==exam].iloc[0]
        _,raw=a.load_raw(project/r.source_path)
        for it,(tt,xx,_) in enumerate(raw,1):
            seg,_=a.prepare(tt,xx);t,x=seg[0];mask=(t>=3)&(t<13);f,power=a.spectrum(x[mask,7]);maskf=f<=15
            axs[0,col].semilogy(f[maskf],power[maskf],label=f'Attempt {it}',lw=1)
            ps.append(dict(patient_id=pid,iteration=it,channel=10,**a.spectral_features(f,power)))
        # Iteration 2 spectrogram, fixed channel 10 shared by both people.
        seg,_=a.prepare(raw[1][0],raw[1][1]);t,x=seg[0];mask=(t>=3)&(t<15)
        f,st,power=signal.spectrogram(x[mask,7],fs=a.FS,window='hann',nperseg=200,noverlap=180,detrend='linear',scaling='density')
        m=f<=15;specs.append((f[m],st+3,10*np.log10(np.maximum(power[m],1e-20))))
        axs[0,col].set_title(f'{pair_labels[cohort]} · one fixed sensor channel');axs[0,col].set_xlabel('Frequency [Hz]');axs[0,col].set_ylabel('Oscillation power (log scale)');axs[0,col].legend(fontsize=8);axs[0,col].set_ylim(1e-12,1e-4)
    vals=np.concatenate([s[2].ravel() for s in specs]);lo,hi=np.percentile(vals,[2,99])
    for col,(f,t,power) in enumerate(specs):
        im=axs[1,col].pcolormesh(t,f,power,shading='auto',cmap='magma',vmin=lo,vmax=hi);axs[1,col].grid(False);axs[1,col].axhline(3,color='white',ls=':',lw=.7);axs[1,col].axhline(4,color='white',ls='--',lw=.7);axs[1,col].set_xlabel('Time within attempt 2 [s]');axs[1,col].set_ylabel('Frequency [Hz]')
    fig.colorbar(im,ax=axs[1],label='Oscillation strength (brighter = stronger)',shrink=.8)
    fig.suptitle('A clear teaching example: healthy participant compared with sick participant\nA repeated bright ridge near 4 Hz is visible for the sick participant',fontsize=12)
    a.savefig(fig,'08_illustrative_tremor_case')
    pd.DataFrame(ps).to_csv(out/'illustrative_case_channel10.csv',index=False)
    # Cutoff/window sensitivity uses participant-level values, not pooled sessions.
    sensitivity=[]
    for w in [3,5,10]:
        d=p[p.window_s==w]
        for metric in ['tremor_rms','sensitivity_3_rms','sensitivity_3p5_rms']:
            pp=d.groupby(['patient_id','cohort'])[metric].mean().reset_index()
            for cohort,g in pp.groupby('cohort'):sensitivity.append(dict(window_s=w,metric=metric,cohort=cohort,median=g[metric].median()))
    pd.DataFrame(sensitivity).to_csv(out/'band_window_sensitivity.csv',index=False)
    print('Additional outputs complete',flush=True)
    return confounds
