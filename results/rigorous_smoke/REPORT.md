# Rigorous causal GSEWS synthetic benchmark

This benchmark uses equal isotropic node noise, a graph-commuting drift
operator, trailing-only processing, and window-end eligibility.

## Locked configuration

```json
{
  "seed": 20260804,
  "n_nodes": 24,
  "n_steps": 900,
  "dt": 0.05,
  "transition_step": 720,
  "ramp_start": 320,
  "baseline_end": 300,
  "pretransition_start": 420,
  "prediction_horizon": 100,
  "window": 100,
  "stride": 20,
  "trailing_smooth": 3,
  "trend_span": 6,
  "noise_sigma": 0.2,
  "spectral_center": 0.35,
  "cubic_strength": 1.0,
  "heterogeneity": 0.0,
  "graph_perturbation": 0.2,
  "n_seeds_per_graph": 2,
  "bootstrap_replicates": 500,
  "graphs": [
    "ring",
    "small_world"
  ],
  "models": [
    "graph_commuting_ou",
    "graph_pattern_normal_form"
  ],
  "output_dir": "results/rigorous_smoke"
}
```

## Summary

model,method,n_runs,auc_mean,auc_std,lead_mean,false_alarm_mean,mode_accuracy_mean,localization_corr_mean
graph_commuting_ou,IdentityBasis:GSEWS,4,1.0,0.0,280.0,0.0,0.0,-0.10839839348191876
graph_commuting_ou,NodeMax,4,1.0,0.0,280.0,0.0,,
graph_commuting_ou,PermutedGraph:GSEWS,4,1.0,0.0,280.0,0.0,0.0,-0.14754676682412884
graph_commuting_ou,RandomGraph:GSEWS,4,1.0,0.0,280.0,0.0,0.05,-0.1497296769753173
graph_commuting_ou,GSEWS,4,0.9977272727272727,0.004545454545454577,280.0,0.0,0.1,0.1115900355870223
graph_commuting_ou,GSEWS_Level,4,0.9977272727272727,0.0045454545454546025,280.0,0.0,,
graph_commuting_ou,GSEWS_Var,4,0.9977272727272727,0.004545454545454577,280.0,0.0,,
graph_commuting_ou,PerturbedGraph:GSEWS,4,0.9977272727272727,0.004545454545454577,280.0,0.0,0.0,0.217692361383306
graph_commuting_ou,GSEWS_AC,4,0.9954545454545455,0.009090909090909094,280.0,0.0,,
graph_commuting_ou,NodeAvg,4,0.5431818181818182,0.18004820291216686,95.0,0.0,,
graph_commuting_ou,GraphSmooth,4,0.42727272727272725,0.31553007566654834,0.0,0.0,,
graph_commuting_ou,Energy,4,0.3818181818181818,0.29345477075579957,0.0,0.0,,
graph_commuting_ou,PCA,4,0.37272727272727274,0.33418623665757075,70.0,0.0,,
graph_commuting_ou,EdgeCorr,4,0.3295454545454546,0.27316883446178875,100.0,0.0,,
graph_pattern_normal_form,GSEWS_Level,4,1.0,0.0,280.0,0.0,,
graph_pattern_normal_form,GSEWS_Var,4,1.0,0.0,280.0,0.0,,
graph_pattern_normal_form,IdentityBasis:GSEWS,4,1.0,0.0,280.0,0.0,0.125,0.28016540629589426
graph_pattern_normal_form,NodeMax,4,1.0,0.0,280.0,0.0,,
graph_pattern_normal_form,PermutedGraph:GSEWS,4,1.0,0.0,280.0,0.0,0.05,0.05704668721306837
graph_pattern_normal_form,PerturbedGraph:GSEWS,4,1.0,0.0,280.0,0.0,0.175,0.2427149070898188
graph_pattern_normal_form,RandomGraph:GSEWS,4,1.0,0.0,280.0,0.0,0.1,0.16622032562792008
graph_pattern_normal_form,GSEWS,4,0.9795454545454545,0.040909090909090944,280.0,0.0,0.1,-0.24292996751113327
graph_pattern_normal_form,GSEWS_AC,4,0.9318181818181819,0.10820346103852807,280.0,0.0,,
graph_pattern_normal_form,EdgeCorr,4,0.5113636363636364,0.2974061418719198,0.0,0.0,,
graph_pattern_normal_form,GraphSmooth,4,0.4954545454545455,0.27995473856926045,170.0,0.0,,
graph_pattern_normal_form,NodeAvg,4,0.49318181818181817,0.25716991199664796,60.0,0.0,,
graph_pattern_normal_form,Energy,4,0.41590909090909095,0.2822943191858952,0.0,0.0,,
graph_pattern_normal_form,PCA,4,0.3704545454545454,0.2781654778326226,0.0,0.0,,

## Paired AUC differences (GSEWS minus comparator)

model,reference,comparator,n_pairs,auc_difference_mean,ci95_low,ci95_high,probability_difference_gt_zero
graph_commuting_ou,GSEWS,EdgeCorr,4,0.6681818181818182,0.43636363636363634,0.8727272727272728,1.0
graph_commuting_ou,GSEWS,PCA,4,0.625,0.3409090909090909,0.9090909090909091,1.0
graph_commuting_ou,GSEWS,Energy,4,0.615909090909091,0.36818181818181817,0.8636363636363636,1.0
graph_commuting_ou,GSEWS,GraphSmooth,4,0.5704545454545454,0.2772727272727273,0.7886363636363636,1.0
graph_commuting_ou,GSEWS,NodeAvg,4,0.45454545454545453,0.3363636363636364,0.615909090909091,1.0
graph_commuting_ou,GSEWS,GSEWS_AC,4,0.0022727272727272596,0.0,0.006818181818181779,0.712
graph_commuting_ou,GSEWS,GSEWS_Level,4,0.0,-0.006818181818181862,0.006818181818181862,0.37
graph_commuting_ou,GSEWS,GSEWS_Var,4,0.0,0.0,0.0,0.0
graph_commuting_ou,GSEWS,PerturbedGraph:GSEWS,4,0.0,0.0,0.0,0.0
graph_commuting_ou,GSEWS,IdentityBasis:GSEWS,4,-0.0022727272727272874,-0.006818181818181862,0.0,0.0
graph_commuting_ou,GSEWS,NodeMax,4,-0.0022727272727272874,-0.006818181818181862,0.0,0.0
graph_commuting_ou,GSEWS,PermutedGraph:GSEWS,4,-0.0022727272727272874,-0.006818181818181862,0.0,0.0
graph_commuting_ou,GSEWS,RandomGraph:GSEWS,4,-0.0022727272727272874,-0.006818181818181862,0.0,0.0
graph_pattern_normal_form,GSEWS,PCA,4,0.6090909090909091,0.40454545454545454,0.8045454545454546,1.0
graph_pattern_normal_form,GSEWS,Energy,4,0.5636363636363635,0.29999999999999993,0.8045454545454546,1.0
graph_pattern_normal_form,GSEWS,NodeAvg,4,0.48636363636363633,0.28863636363636364,0.7457954545454542,1.0
graph_pattern_normal_form,GSEWS,GraphSmooth,4,0.48409090909090907,0.3,0.6863636363636363,1.0
graph_pattern_normal_form,GSEWS,EdgeCorr,4,0.4681818181818182,0.2409090909090909,0.6954545454545454,1.0
graph_pattern_normal_form,GSEWS,GSEWS_AC,4,0.0477272727272727,0.0,0.10909090909090904,0.944
graph_pattern_normal_form,GSEWS,GSEWS_Level,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
graph_pattern_normal_form,GSEWS,GSEWS_Var,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
graph_pattern_normal_form,GSEWS,IdentityBasis:GSEWS,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
graph_pattern_normal_form,GSEWS,NodeMax,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
graph_pattern_normal_form,GSEWS,PermutedGraph:GSEWS,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
graph_pattern_normal_form,GSEWS,PerturbedGraph:GSEWS,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
graph_pattern_normal_form,GSEWS,RandomGraph:GSEWS,4,-0.020454545454545475,-0.061363636363636426,0.0,0.0
