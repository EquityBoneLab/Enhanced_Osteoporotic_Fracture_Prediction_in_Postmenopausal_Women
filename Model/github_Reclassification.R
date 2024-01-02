library(PredictABEL)

y_twomodels=read.csv("./xgb_reclassification_mof.csv")
y_grs=y_twomodels$TestScore
y_nogrs_FRAX=y_twomodels$WHOFRAC
reclassification(data=y_twomodels, 
                 cOutcome=1, 
                 predrisk1=y_nogrs_FRAX,
                 predrisk2=y_grs, 
                 cutoff=c(0,0.2,1))
