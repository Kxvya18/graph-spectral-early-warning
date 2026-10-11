# Synthetic GSEWS Experiment Report

The synthetic experiments use known destabilizing graph eigenmodes.
This validates the central theory claim before moving to real datasets.

## Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
Energy,0.5772270114942528,0.12517497122384905,22.5,0.0,,
GSEWS,0.7216235632183908,0.1373476388457978,295.0,0.0,0.53125,0.39543613541315326
NodeAvg,0.58441091954023,0.12781869985283298,32.5,0.0,,
PCA,0.5463362068965517,0.10814390370097451,55.0,0.0,,

## GSEWS by Graph and Model

graph,model,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
modular,fhn_reduced,0.7169540229885057,0.11093033417791738,280.0,0.0,0.0,0.003673746891418935
modular,modal_ou,0.6408045977011494,0.1814615924985794,280.0,0.0,1.0,1.0
ring,fhn_reduced,0.7528735632183908,0.0813654892802746,330.0,0.0,0.125,-0.4219292052388059
ring,modal_ou,0.7758620689655172,0.19024203871305098,290.0,0.0,1.0,1.0

## Recommended Chapter Figures

- `fig01_representative_modal_warning.pdf`: modal variance and global warning before transition.
- `fig02_benchmark_summary.pdf`: GSEWS versus node-average, PCA, and signal-energy baselines.
- `fig03_topology_comparison.pdf`: topology-dependent warning behavior.
