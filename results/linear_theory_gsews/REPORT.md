# Synthetic GSEWS Experiment Report

The synthetic experiments use known destabilizing graph eigenmodes.
This validates the central theory claim before moving to real datasets.

## Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
EdgeCorr,0.5,0.1614965485950686,169.0,0.0020833333333333333,,
Energy,0.4988888888888889,0.1581718712654393,350.5,0.008333333333333333,,
GSEWS,0.7270039682539682,0.10199467558824016,346.5,0.00625,1.0,1.0
GraphSmooth,0.48664682539682536,0.17423138048258585,0.0,0.0,,
NodeAvg,0.5068849206349206,0.1542067506029086,183.0,0.0034722222222222225,,
NodeMax,0.5148809523809523,0.13888775509741305,262.5,0.0034722222222222225,,
PCA,0.5018055555555556,0.1567450129537274,352.5,0.008333333333333333,,

## GSEWS by Graph and Model

graph,model,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
modular,modal_ou,0.7244444444444444,0.16055860467654573,386.0,0.01111111111111111,1.0,1.0
ring,modal_ou,0.7226190476190476,0.05230239686998049,312.0,0.0027777777777777775,1.0,1.0
scale_free,modal_ou,0.7281746031746033,0.0686101305799412,344.0,0.0027777777777777775,1.0,1.0
small_world,modal_ou,0.7327777777777778,0.10858236714805203,344.0,0.008333333333333333,1.0,1.0

## Recommended Chapter Figures

- `fig01_representative_modal_warning.pdf`: modal variance and global warning before transition.
- `fig02_benchmark_summary.pdf`: GSEWS versus node-average, PCA, and signal-energy baselines.
- `fig03_topology_comparison.pdf`: topology-dependent warning behavior.
