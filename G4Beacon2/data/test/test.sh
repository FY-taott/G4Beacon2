mkdir HepG2Train_K562Test

g4beacon2 getValidatedG4s \
       --seqCSV minus_seq_1k.csv \
       --atacCSV K562_minus_atac_1k.csv \
       --originBED minus_origin_1k.bed \
       --model zscoreDNABERT2_HepG2_ES00_0517model.checkpoint.joblib \
       -o HepG2Train_K562Test/ES00_minus_prediction_1k.bed
