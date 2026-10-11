# GSEWS Experiment Evidence Summary

This file summarizes the experiment evidence currently available for the separate GSEWS book chapter.

## Claim Status

- Linearized graph-spectral theorem: supported by controlled modal OU experiments with known critical eigenmode.
- Global early warning: supported in both linearized and nonlinear reduced simulations.
- Exact eigenmode localization: supported in the linearized theorem setting; not claimed as exact in nonlinear reduced FHN simulations.
- Actual clinical dataset validation: pending dataset download/preprocessing.

## Linearized modal OU theorem test

### Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
Energy,0.4988888888888889,0.1581718712654393,350.5,0.0083333333333333,,
GSEWS,0.7270039682539682,0.1019946755882401,346.5,0.00625,1.0,1.0
NodeAvg,0.5068849206349206,0.1542067506029086,183.0,0.0034722222222222,,
PCA,0.5018055555555556,0.1567450129537274,352.5,0.0083333333333333,,

### By Graph/Model

graph,model,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
modular,modal_ou,0.7244444444444444,0.1605586046765457,386.0,0.0111111111111111,1.0,1.0
ring,modal_ou,0.7226190476190476,0.0523023968699804,312.0,0.0027777777777777,1.0,1.0
scale_free,modal_ou,0.7281746031746033,0.0686101305799412,344.0,0.0027777777777777,1.0,1.0
small_world,modal_ou,0.7327777777777778,0.108582367148052,344.0,0.0083333333333333,1.0,1.0

## Nonlinear reduced excitable physiology test

### Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
Energy,0.6628472222222223,0.0591161797638555,1320.0,0.0442708333333333,,
GSEWS,0.736904761904762,0.0676787824297266,513.75,0.0052083333333333,0.0734375,0.1676898532100732
NodeAvg,0.6698164682539682,0.0656229977249086,330.0,0.0069444444444444,,
PCA,0.6482638888888889,0.0691008433210478,1320.0,0.046875,,

### By Graph/Model

graph,model,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
modular,fhn_reduced,0.732638888888889,0.0645914421868105,515.0,0.0,0.025,0.0772238441074965
ring,fhn_reduced,0.7561507936507936,0.0884258163914767,555.0,0.0104166666666666,0.1625,0.3765839110380297
scale_free,fhn_reduced,0.7198412698412698,0.0466067766314506,507.5,0.0069444444444444,0.03125,0.2345780422337826
small_world,fhn_reduced,0.7389880952380953,0.0730718476908453,477.5,0.0034722222222222,0.075,-0.0176263845390159

## Smoke mixed test

### Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
Energy,0.5772270114942528,0.125174971223849,22.5,0.0,,
GSEWS,0.7216235632183908,0.1373476388457978,295.0,0.0,0.53125,0.3954361354131532
NodeAvg,0.58441091954023,0.1278186998528329,32.5,0.0,,
PCA,0.5463362068965517,0.1081439037009745,55.0,0.0,,

### By Graph/Model

graph,model,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_acc_mean,loc_corr_mean
modular,fhn_reduced,0.7169540229885057,0.1109303341779173,280.0,0.0,0.0,0.0036737468914189
modular,modal_ou,0.6408045977011494,0.1814615924985794,280.0,0.0,1.0,1.0
ring,fhn_reduced,0.7528735632183908,0.0813654892802746,330.0,0.0,0.125,-0.4219292052388059
ring,modal_ou,0.7758620689655172,0.1902420387130509,290.0,0.0,1.0,1.0

## Figure Inventory

- `results/linear_theory_gsews/figures/fig01_representative_modal_warning.pdf`
- `results/linear_theory_gsews/figures/fig02_benchmark_summary.pdf`
- `results/linear_theory_gsews/figures/fig03_topology_comparison.pdf`
- `results/nonlinear_physiology_gsews/figures/fig02_benchmark_summary.pdf`
- `results/nonlinear_physiology_gsews/figures/fig03_topology_comparison.pdf`

## Next Clinical Dataset Step

Use CHB-MIT scalp EEG first because seizure onset is a canonical coupled-network transition and channels naturally define graph nodes. The clinical claim should be early-warning discrimination and interpretable modal patterns, not exact ground-truth eigenmode recovery.
