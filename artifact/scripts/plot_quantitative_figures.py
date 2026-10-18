"""Portable wrapper for the three quantitative figure functions; no manuscript edits.
Functions effects/selection/sensitivity/save retain the original plotting operations,
with the sensitivity-table location mapped to the distribution layout.
No model inference, fitting, data downloading, or statistical selection occurs.
"""
from pathlib import Path
import argparse,csv,json
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SCI=ROOT/'records/s3'
OUT=ROOT
FIG=None
NAVY,CORAL,SAND='#1A3A5C','#E05A47','#D4A96A'
INK,GRAY,LIGHT,FAINT=NAVY,'#7A8A99','#DDE2E6','#F8F9FA'
DATA=['uci_har','wisdm','uwave','cwru']
NAMES=['UCI-HAR','WISDM','UWave','CWRU']
SEEDS=[11,22,33,44,55,66,77,88]
RID=['R1_analytic_head','R2_pca16_readout']
def jread(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def cread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',facecolor='white')
    plt.close(fig)

def effects(raw,main):
    fig=plt.figure(figsize=(7.15,2.90))
    ax=fig.add_axes([.235,.16,.525,.71]);ax.set(xlim=(-18,10),ylim=(-.6,9.65))
    positions=[9,8,6.5,5.5,4,3,1.5,.5]
    for i in range(4):
        if i%2==0:ax.axhspan(positions[2*i+1]-.43,positions[2*i]+.43,color=FAINT,zorder=0)
    ax.axvline(0,color=GRAY,lw=.8,ls=(0,(3,3)),zorder=0)
    records=[]
    for i,(d,name) in enumerate(zip(DATA,NAMES)):
        ym=(positions[2*i]+positions[2*i+1])/2
        ax.text(-.43,ym,name,transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8.8,weight='bold',clip_on=False)
        for k,(rid,color) in enumerate(zip(RID,[NAVY,CORAL])):
            y=positions[2*i+k]
            rr=sorted([r for r in raw if r['candidate']==rid and r['dataset']==d],key=lambda r:int(r['seed']))
            assert [int(r['seed']) for r in rr]==SEEDS
            values=[float(r['paired_gain_pp']) for r in rr];assert min(values)>-18 and max(values)<10
            ax.scatter(values,y+np.linspace(-.22,.22,8),s=11,color='#AAB2B9',alpha=.85,linewidth=0,zorder=2)
            m=main[rid+'__'+d];mean=m['raw_mean']*100;lo=m['raw_ci_low']*100;hi=m['raw_ci_high']*100;sig=m['p_adjusted_holm24']<.05
            ax.plot([lo,hi],[y,y],color=color,lw=1.65,zorder=3)
            ax.plot([lo,lo],[y-.075,y+.075],color=color,lw=.85);ax.plot([hi,hi],[y-.075,y+.075],color=color,lw=.85)
            ax.plot(mean,y,marker='D',ms=5.0,mfc=color if sig else 'white',mec=color,mew=1.05,zorder=4)
            ax.text(-.06,y,'R'+str(k+1),transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8.2,color=color,clip_on=False)
            display=Decimal(str(m['mean_gain_pp'])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
            ax.text(1.10,y,f'{display:+.2f}',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8.5,color=color,clip_on=False)
            ax.text(1.40,y,f"{m['strictly_positive']}/8",transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8.5,clip_on=False)
            records.append({'cell':m['cell'],'seed_gains_pp':values,'mean_pp':mean,'nominal_ci_pp':[lo,hi],'holm_p':m['p_adjusted_holm24']})
    ax.set_yticks([]);ax.set_xticks([-15,-10,-5,0,5,10]);ax.tick_params(axis='x',length=2.5)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(LIGHT)
    ax.set_xlabel('Paired test macro difference vs B0 (pp)',labelpad=7)
    ax.text(-.43,10.25,'DATASET',transform=ax.get_yaxis_transform(),fontsize=7.5,color=GRAY,clip_on=False)
    ax.text(1.10,10.25,'GAIN',transform=ax.get_yaxis_transform(),ha='right',fontsize=7.5,color=GRAY,clip_on=False)
    ax.text(1.40,10.25,'POSITIVE',transform=ax.get_yaxis_transform(),ha='right',fontsize=7.5,color=GRAY,clip_on=False)
    handles=[Line2D([],[],marker='o',ls='',ms=3,color='#AAB2B9',label='Seed'),Line2D([],[],color=NAVY,lw=1.2,marker='D',mfc='white',ms=4,label='Mean + nominal 95% CI'),Line2D([],[],marker='D',ls='',ms=4,color=NAVY,label='Filled: Holm p < 0.05')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,1.0),ncol=3,frameon=False,fontsize=7.5,handletextpad=.4,columnspacing=1.25)
    save(fig,'fig2_readout_effects');return records

def selection(raw):
    vv={int(r['seed']):r['diff_pp'] for r in jread(SCI/'expand_8seed_record.json')['seeds']}
    tt={int(r['seed']):float(r['paired_gain_pp']) for r in raw if r['cell']=='C1_backbone_freeze__wisdm'}
    fig=plt.figure(figsize=(7.15,2.50));ax=fig.add_axes([.085,.21,.575,.64]);ax.set(xlim=(-7,12),ylim=(-.6,7.6))
    ax.axvline(0,color=GRAY,lw=.75,ls=(0,(3,3)))
    for i,s in enumerate(SEEDS):
        y=7-i;ax.plot([vv[s],tt[s]],[y,y],color='#AAB2B9',lw=.9,zorder=1)
    ax.scatter([vv[s] for s in SEEDS],range(7,-1,-1),s=25,color=NAVY,marker='o',label='Validation',zorder=3)
    ax.scatter([tt[s] for s in SEEDS],range(7,-1,-1),s=23,color=CORAL,marker='s',label='Held-out test',zorder=3)
    ax.set_yticks(range(7,-1,-1),[str(s)+('*' if s in SEEDS[:3] else '') for s in SEEDS],fontsize=8)
    ax.set_xticks([-5,0,5,10]);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(LIGHT);ax.tick_params(axis='y',length=0)
    ax.set_xlabel('Gain vs B0 (pp)',labelpad=7);ax.set_ylabel('Paired seed',labelpad=6)
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.02,1.03),ncol=2,fontsize=8,handletextpad=.4,columnspacing=1.6)
    sx=fig.add_axes([.73,.15,.255,.73]);sx.set(xlim=(0,1),ylim=(-.15,1.10));sx.axis('off')
    sx.text(0,1.02,'POST-SCREENING OBSERVATION',fontsize=6.9,color=GRAY)
    sx.plot([0,1],[.94,.94],color=LIGHT,lw=.7)
    sx.text(0,.82,'VALIDATION',fontsize=7.5,color=NAVY);sx.text(0,.64,'+3.30 pp',fontsize=15.5,color=NAVY,weight='bold');sx.text(0,.51,'6/8 positive',fontsize=8.2,color=GRAY)
    sx.plot([0,1],[.43,.43],color=LIGHT,lw=.7)
    sx.text(0,.32,'HELD-OUT TEST',fontsize=7.5,color=CORAL);sx.text(0,.14,'−0.42 pp',fontsize=15.5,color=CORAL,weight='bold');sx.text(0,.01,'2/8 positive · Holm p = 1',fontsize=7.4,color=GRAY)
    sx.text(0,-.11,'95% CI [−2.77, +1.93]',fontsize=7.4,color=GRAY)
    fig.text(.085,.006,'* Original screening seeds retained in the expanded set. Connectors pair observations only.',fontsize=6.7,color=GRAY,va='bottom')
    save(fig,'fig3_selection_case')
    return [{'seed':s,'validation_gain_pp':vv[s],'test_gain_pp':tt[s]} for s in SEEDS]

def sensitivity():
    rr=cread(ROOT/'paper_tables/s4_units.csv');fig,axes=plt.subplots(1,2,figsize=(7.15,2.40),sharex=True,sharey=True)
    fig.subplots_adjust(left=.085,right=.975,bottom=.27,top=.76,wspace=.17)
    shapes=['o','s','D','^']
    for i,(ax,rid,color,title) in enumerate(zip(axes,RID,[NAVY,CORAL],['R1  |  RIDGE56','R2  |  PCA16 + RIDGE'])):
        ax.axhline(0,color=GRAY,lw=.7,ls=(0,(3,3)))
        for d,name,mark in zip(DATA,NAMES,shapes):
            r=next(r for r in rr if r['candidate']==rid and r['dataset']==d);x=float(r['s2_range4_mean_pp']);y=float(r['test_gain_mean_pp'])
            ax.scatter(x,y,marker=mark,color=color,s=30,zorder=3)
            dx,dy={'uci_har':(-6,7),'wisdm':(8,1),'uwave':(-6,8),'cwru':(-5,-14)}[d]
            ax.annotate(name,(x,y),xytext=(dx,dy),textcoords='offset points',ha='left' if dx>0 else 'right',fontsize=8)
        ax.set_title(title,loc='left',fontsize=9,color=color,weight='bold',pad=25)
        ax.text(0,1.10,r'$\rho=0.40$  ·  four dataset units',transform=ax.transAxes,fontsize=8,color=GRAY)
        ax.set(xlim=(0,19),ylim=(-14,7.5),xticks=[0,5,10,15],yticks=[-10,-5,0,5]);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(LIGHT);ax.tick_params(axis='y',length=0)
    axes[0].set_ylabel('Mean test gain (pp)',labelpad=6)
    fig.text(.53,.095,'Mean validation range across four readout views (pp)',ha='center',fontsize=9)
    fig.text(.53,.015,'Descriptive association only; no fitted line or out-of-sample predictive evaluation.',ha='center',fontsize=7.4,color=GRAY)
    save(fig,'fig4_sensitivity');return rr
def main():
    global FIG
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    FIG=args.output.resolve()
    FIG.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.labelsize':9,'axes.titlesize':9.5,'xtick.labelsize':8,
        'ytick.labelsize':9,'text.color':INK,'axes.labelcolor':INK,
        'axes.edgecolor':GRAY,'xtick.color':GRAY,'ytick.color':INK,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.dpi':350,'figure.facecolor':'white','axes.facecolor':'white'})
    raw=cread(SCI/'test_s4/test_per_seed_raw.csv')
    ms={r['cell']:r for r in jread(SCI/'test_s4/test_main_table.json')['rows'] if r['n_seeds']==8}
    got={'figure2_rows':effects(raw,ms),'figure3_rows':selection(raw),'figure4_rows':sensitivity()}
    stored=jread(ROOT/'figure_sources/FIGURE_SOURCES.json')
    assert all(got[k]==stored[k] for k in got), 'Figure source values differ from saved manuscript inputs'
    (FIG/'figure_value_verification.json').write_text(json.dumps({'status':'PASS','scope':'All numeric plotting inputs match the saved figure source; exported image bytes can vary with fonts/library versions.'},indent=2)+'\n',encoding='utf-8')
    print('Rendered quantitative figures 2-4; source values PASS.')
if __name__=='__main__':main()
