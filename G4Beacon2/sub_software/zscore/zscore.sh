# conda install bioconda::ucsc-bigwigtowig
# conda install bioconda::bedops
# conda install soil::ucsc-bedgraphtobigwig
#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "Usage: bash norm.sh <input.bigWig>" >&2
  exit 1
fi
IN_BW="$1"

bigWigToWig "$IN_BW" ATAC_pval.wig
wig2bed < ATAC_pval.wig > ATAC_pval.bed
cut -f 1,2,3,5- ATAC_pval.bed > ATAC_pval.bedGraph

python3 bwnorm.py ATAC_pval.bedGraph

cut -f 1-3 ATAC_pval.bedGraph > ATAC_pval_cut.temp
awk '{printf "%.6g\n", $2}' ATAC_pval_z_score_normalized.bedgraph > ATAC_pval_zscore.temp
paste -d'\t' ATAC_pval_cut.temp ATAC_pval_zscore.temp > ATAC_pval_zscore.bedGraph
bedGraphToBigWig ATAC_pval_zscore.bedGraph chrom.sizes ATAC_pval_zscore.bigwig