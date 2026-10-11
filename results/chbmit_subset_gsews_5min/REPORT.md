# CHB-MIT Real EEG GSEWS Report

This is real multichannel EEG validation on downloaded CHB-MIT EDF recordings.
Labels are preictal windows before annotated seizure onset versus early baseline windows from the same record.
Localization is reported as an interpretable dominant modal pattern, not as ground-truth pathological-region accuracy.

## Method Summary

method,auc_mean,auc_std,lead_mean,false_alarm_mean
EdgeCorr,0.5914704477204478,0.22151620809345565,0.0,0.0
Energy,0.5444717444717445,0.11852677140282294,319.0,0.011904761904761904
GSEWS,0.5558268495768496,0.3558481576861908,333.0,0.01870748299319728
GraphSmooth,0.41465670215670214,0.21964612975800804,0.0,0.0
NodeAvg,0.5016072891072891,0.38449293177664096,329.0,0.015306122448979591
NodeMax,0.45659125034125037,0.2961986912961709,311.5,0.003401360544217687
PCA,0.5685981435981435,0.13226159190007178,311.5,0.00510204081632653

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
