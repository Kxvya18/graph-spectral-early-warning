# CHB-MIT Real EEG GSEWS Report

This is real multichannel EEG validation on downloaded CHB-MIT EDF recordings.
Labels are preictal windows before annotated seizure onset versus early baseline windows from the same record.
Localization is reported as an interpretable dominant modal pattern, not as ground-truth pathological-region accuracy.

## Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean
EdgeCorr,0.47907697583623504,0.10804588303806209,221.0,0.0011337868480725624
Energy,0.5822530864197532,0.08053086632674876,221.33333333333334,0.007936507936507936
GSEWS,0.6714796349055608,0.2533992224384354,264.0,0.011337868480725625
GraphSmooth,0.50838081671415,0.12212925622623563,22.0,0.0
NodeAvg,0.5854146881924659,0.2084223116256981,496.3333333333333,0.012471655328798186
NodeMax,0.6184710351377019,0.19102630936810566,289.0,0.0022675736961451248
PCA,0.5838424606943126,0.05166796028236087,219.66666666666666,0.005668934240362813

## Records

model,method,events
chb01_03,EdgeCorr,1
chb01_03,Energy,1
chb01_03,GSEWS,1
chb01_03,GraphSmooth,1
chb01_03,NodeAvg,1
chb01_03,NodeMax,1
chb01_03,PCA,1
chb01_04,EdgeCorr,1
chb01_04,Energy,1
chb01_04,GSEWS,1
chb01_04,GraphSmooth,1
chb01_04,NodeAvg,1
chb01_04,NodeMax,1
chb01_04,PCA,1
chb01_15,EdgeCorr,1
chb01_15,Energy,1
chb01_15,GSEWS,1
chb01_15,GraphSmooth,1
chb01_15,NodeAvg,1
chb01_15,NodeMax,1
chb01_15,PCA,1
