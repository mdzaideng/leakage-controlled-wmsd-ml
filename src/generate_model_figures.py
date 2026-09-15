#!/usr/bin/env python3
"""Generate manuscript Figures 4, 5, 10 and 11 as 600-dpi PNG and vector PDF."""
from pathlib import Path
import argparse
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS=["GBT","RF","AB","SVM","LR","DT","NB","kNN"]

def style():
    names={f.name for f in fm.fontManager.ttflist}
    family="Times New Roman" if "Times New Roman" in names else "Nimbus Roman"
    plt.rcParams.update({"font.family":family,"font.size":10,"axes.titlesize":11,
        "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
        "savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42})
    return family

def save(fig,out,name):
    fig.savefig(out/f"{name}.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}.pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)

def box(ax,x,y,w,h,text,fc,ec,fs=9):
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,lw=1.25))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs)

def fig4(out):
    fig,ax=plt.subplots(figsize=(7.2,3.0)); ax.set(xlim=(0,10),ylim=(0,4.5)); ax.axis("off")
    labels=["60 analytical\nrecords","Remove identifiers\nand outcome proxies","Repeated outer\nheld-out testing","Inner tuning on\ntraining data only","Out-of-fold\npredictions"]
    xs=[.15,2.10,4.05,6.00,7.95]
    for x,t in zip(xs,labels): box(ax,x,2.75,1.7,1,t,"#EAF2F8","#24527A")
    for x1,x2 in zip(xs[:-1],xs[1:]): ax.annotate("",(x2-.05,3.25),(x1+1.75,3.25),arrowprops=dict(arrowstyle="->",lw=1.2,color="#24527A"))
    bl=["Fold-contained\npreprocessing","Selection by mean\nbalanced accuracy","Aggregate results\nacross 10 repeats"]; bx=[1.85,4.05,6.25]
    for x,t in zip(bx,bl): box(ax,x,.55,1.9,.95,t,"#F3F6F4","#477A5B")
    for x1,x2 in zip(bx[:-1],bx[1:]): ax.annotate("",(x2-.05,1.025),(x1+1.95,1.025),arrowprops=dict(arrowstyle="->",lw=1.1,color="#477A5B"))
    ax.annotate("",(2.8,1.55),(4.9,2.70),arrowprops=dict(arrowstyle="->",color="#666"))
    ax.annotate("",(7.2,1.55),(8.75,2.70),arrowprops=dict(arrowstyle="->",color="#666"))
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98); save(fig,out,"Figure_4_leakage_controlled_workflow")

def fig5(out):
    fig,ax=plt.subplots(figsize=(7.2,3.7)); ax.set(xlim=(0,10),ylim=(0,6)); ax.axis("off")
    ax.text(.25,5.55,"Repeat 10 times with prespecified seeds",fontsize=11,fontweight="bold",color="#244A68")
    ax.add_patch(plt.Rectangle((.25,3.65),9.4,1.35,facecolor="#F7FAFC",edgecolor="#244A68",lw=1.3))
    ax.text(.48,4.65,"Outer stratified five-fold cross-validation",fontsize=10,fontweight="bold")
    xs=[.55+i*1.67 for i in range(5)]
    for i,x in enumerate(xs): box(ax,x,3.88,1.48,.55,"Held-out test" if i==4 else f"Training fold {i+1}","#F8CBAD" if i==4 else "#DDEBF7","white",8)
    ax.text(8.10,4.64,"Rotate held-out fold",fontsize=8.5,color="#555")
    ax.annotate("",(7.45,3.60),(7.45,2.80),arrowprops=dict(arrowstyle="->",lw=1.2,color="#244A68"))
    ax.add_patch(plt.Rectangle((.75,.75),7.2,1.75,facecolor="#F5F9F6",edgecolor="#477A5B",lw=1.3))
    ax.text(1,2.18,"Within each outer-training set: stratified inner three-fold CV",fontsize=10,fontweight="bold")
    labs=["Fit preprocessing","Tune candidate\nhyperparameters","Select by balanced\naccuracy","Refit on complete\nouter-training set"]; ix=[1.05,2.72,4.55,6.25]
    for x,t in zip(ix,labs): box(ax,x,1.03,1.37,.72,t,"white","#477A5B",7.8)
    for x1,x2 in zip(ix[:-1],ix[1:]): ax.annotate("",(x2-.04,1.39),(x1+1.41,1.39),arrowprops=dict(arrowstyle="->",color="#477A5B"))
    box(ax,8.25,.90,1.4,1.45,"Evaluate once on\nouter test fold\n(no fitting)","#FFF2CC","#A67C00",8.5)
    ax.annotate("",(8.20,1.625),(7.98,1.625),arrowprops=dict(arrowstyle="->",lw=1.2,color="#A67C00"))
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98); save(fig,out,"Figure_5_repeated_nested_cv")

def fig11(results,out):
    d=pd.read_csv(results/"primary/model_summary_primary.csv").set_index("model").loc[MODELS]
    assert (d.n_repeats==10).all()
    m=d.balanced_accuracy_mean.to_numpy(); lo=d.balanced_accuracy_ci_low.to_numpy(); hi=d.balanced_accuracy_ci_high.to_numpy(); assert np.all(lo<=m) and np.all(m<=hi)
    y=np.arange(len(d)); fig,ax=plt.subplots(figsize=(7.2,4.3))
    ax.barh(y,m,color=["#B24C3D"]+["#2F6B8A"]*7,height=.64,zorder=2)
    ax.errorbar(m,y,xerr=np.vstack([m-lo,hi-m]),fmt="none",ecolor="#222",capsize=3,lw=1,zorder=3)
    ax.set_yticks(y,d.index); ax.invert_yaxis(); ax.set_xlim(.70,1); ax.set_xlabel("Mean balanced accuracy"); ax.grid(axis="x",alpha=.25,zorder=0)
    for yi,v in zip(y,m): ax.text(.715,yi,f"{v:.3f}",ha="left",va="center",color="white",fontweight="bold",fontsize=9)
    ax.text(.70,-.82,"Error bars show 95% intervals across 10 repeated outer-CV estimates.",fontsize=9,color="#444")
    fig.subplots_adjust(left=.12,right=.98,top=.90,bottom=.14); save(fig,out,"Figure_10_model_balanced_accuracy")

def fig12(results,out):
    a=pd.read_csv(results/"primary/feature_ablation_summary.csv").set_index("feature_set")
    s=pd.read_csv(results/"secondary/sensitivity_summary.csv").set_index("analysis")
    ao=["postural","combined","nonpostural"]; so=["combined_class_weighted","combined_unweighted_rerun","combined_without_exercise","combined_without_lifestyle"]
    av=a.loc[ao,"balanced_accuracy_mean"].to_numpy(); ae=a.loc[ao,"balanced_accuracy_sd"].to_numpy(); sv=s.loc[so,"balanced_accuracy_mean"].to_numpy(); se=s.loc[so,"balanced_accuracy_std"].to_numpy()
    assert np.all((av>=0)&(av<=1)) and np.all((sv>=0)&(sv<=1))
    fig,(x,y)=plt.subplots(1,2,figsize=(7.4,4),sharey=True,gridspec_kw={"wspace":.22})
    x.bar(range(3),av,yerr=ae,capsize=3,color=["#90A4AE","#4F81BD","#70AD47"],zorder=2); x.set_xticks(range(3),["Postural only","Combined","Non-postural only"],rotation=18,ha="right"); x.set_title("(a) Feature-set ablation"); x.set_ylabel("Mean balanced accuracy")
    y.bar(range(4),sv,yerr=se,capsize=3,color=["#8064A2","#4F81BD","#C0504D","#F79646"],zorder=2); y.set_xticks(range(4),["Class weighted","Unweighted","Without exercise","Without lifestyle"],rotation=24,ha="right"); y.set_title("(b) Sensitivity analyses")
    for ax,vals in ((x,av),(y,sv)):
        ax.set_ylim(.55,1); ax.grid(axis="y",alpha=.25,zorder=0)
        for i,v in enumerate(vals): ax.text(i-.29,v-.014,f"{v:.3f}",ha="left",va="top",color="white",fontweight="bold",fontsize=9)
    fig.subplots_adjust(left=.09,right=.99,top=.88,bottom=.25); save(fig,out,"Figure_11_ablation_and_sensitivity")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--package-root",type=Path,default=Path(__file__).resolve().parents[1]); root=p.parse_args().package_root.resolve()
    out=root/"figures/manuscript"; out.mkdir(parents=True,exist_ok=True); family=style()
    fig4(out); fig5(out); fig11(root/"results",out); fig12(root/"results",out)
    files=sorted(x.name for x in out.iterdir() if x.suffix in {".png",".pdf"})
    print("Font family used:",family); print("\n".join(files))
if __name__=="__main__": main()
