#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

import os

HOME = os.environ["HOME"]

# PDB_DIR = os.path.join(HOME, "/bigdat1/pub/PDB/divided")
# 
# ALPHAFOLD_DIR = "/data2/users/chenken/db/alphafold"

HUMAN_CHROMS_ALL = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", 
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", 
        "X", "Y", "M", "MT", "chrX", "chrY", "chrM", "chrMT"
        }

HUMAN_CHROMS_NO_MT = HUMAN_CHROMS_ALL.difference({"chrM", "chrMT", "M", "MT"})
HUMAN_CHROMS_NO_Y_MT = HUMAN_CHROMS_ALL.difference({"chrY", "chrM", "chrMT", "Y", "M", "MT"})

ONEHOT_GENOME_DICT = {
        "hg19": os.path.join(HOME, "db/gencode/GRCh37/GRCh37.primary_assembly.genome.int8.pkl"),
        "hg38-shuffle": os.path.join(HOME, "db/gencode/GRCh38/GRCh38.primary_assembly.genome.int8.shuffle.pkl"),
        "hg38-random": os.path.join(HOME, "db/gencode/GRCh38/GRCh38.primary_assembly.genome.int8.random.pkl"),
        "hg38": os.path.join(HOME, "db/gencode/GRCh38/GRCh38.primary_assembly.genome.int8.pkl"),
        "mm10": os.path.join(HOME, "db/gencode/GRCm38/GRCm38.primary_assembly.genome.int8.pkl"),
        "mm9": os.path.join(HOME, "db/UCSC/mm9/mm9.int8.pkl")
        }
ONEHOT_GENOME_DICT["GRCh38"] = ONEHOT_GENOME_DICT["hg38"]
ONEHOT_GENOME_DICT["GRCh37"] = ONEHOT_GENOME_DICT["hg19"]
ONEHOT_GENOME_DICT["GRCm38"] = ONEHOT_GENOME_DICT["mm10"]

BLACKLIST_DICT = {
    "hg19": os.path.join(HOME, "db/blacklist/hg19-blacklist.v2.bed.gz"),
    "hg38": os.path.join(HOME, "db/blacklist/hg38-blacklist.v2.bed.gz"),
    "hg38-shuffle": os.path.join(HOME, "db/blacklist/hg38-blacklist.v2.bed.gz"),
    "hg38-random": os.path.join(HOME, "db/blacklist/hg38-blacklist.v2.bed.gz"),
    "mm10": os.path.join(HOME, "db/blacklist/mm10-blacklist.v2.bed.gz"),
    "mm9":  os.path.join(HOME, "db/blacklist/mm9-blacklist.bed.gz")
}
BLACKLIST_DICT["GRCh38"] = BLACKLIST_DICT["hg38"]
BLACKLIST_DICT["GRCh37"] = BLACKLIST_DICT["hg19"]
BLACKLIST_DICT["GRCm38"] = BLACKLIST_DICT["mm10"]


## liftover chains

LIFTOVER_CHAINS = {
        "hg19-hg38": os.path.join(HOME, "db/UCSC/hg19/liftOver/hg19ToHg38.over.chain.gz"),
        "mm10-mm9": os.path.join(HOME, "db/UCSC/mm10/mm10ToMm9.over.chain.gz"),
        "hg38-hg19": os.path.join(HOME, "db/UCSC/hg38/liftOver/hg38ToHg19.over.chain.gz"),
        "hg18-hg19": os.path.join(HOME, "db/UCSC/hg18/liftOver/hg18ToHg19.over.chain.gz"),
        "hg18-hg38": os.path.join(HOME, "db/UCSC/hg18/liftOver/hg18ToHg38.over.chain.gz")
}



## 
NUCLEOTIDE4 = {
        'N': 0, 'n': 0,
        'A': 1, 'a': 1,
        'C': 2, 'c': 2,
        'G': 3, 'g': 3,
        'T': 4, 't': 4,
        'U': 4, 'u': 4
}


VCF_HEADER_GRCH38 = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=248956422,assembly=gnomAD_GRCh38>
##contig=<ID=chr2,length=242193529,assembly=gnomAD_GRCh38>
##contig=<ID=chr3,length=198295559,assembly=gnomAD_GRCh38>
##contig=<ID=chr4,length=190214555,assembly=gnomAD_GRCh38>
##contig=<ID=chr5,length=181538259,assembly=gnomAD_GRCh38>
##contig=<ID=chr6,length=170805979,assembly=gnomAD_GRCh38>
##contig=<ID=chr7,length=159345973,assembly=gnomAD_GRCh38>
##contig=<ID=chr8,length=145138636,assembly=gnomAD_GRCh38>
##contig=<ID=chr9,length=138394717,assembly=gnomAD_GRCh38>
##contig=<ID=chr10,length=133797422,assembly=gnomAD_GRCh38>
##contig=<ID=chr11,length=135086622,assembly=gnomAD_GRCh38>
##contig=<ID=chr12,length=133275309,assembly=gnomAD_GRCh38>
##contig=<ID=chr13,length=114364328,assembly=gnomAD_GRCh38>
##contig=<ID=chr14,length=107043718,assembly=gnomAD_GRCh38>
##contig=<ID=chr15,length=101991189,assembly=gnomAD_GRCh38>
##contig=<ID=chr16,length=90338345,assembly=gnomAD_GRCh38>
##contig=<ID=chr17,length=83257441,assembly=gnomAD_GRCh38>
##contig=<ID=chr18,length=80373285,assembly=gnomAD_GRCh38>
##contig=<ID=chr19,length=58617616,assembly=gnomAD_GRCh38>
##contig=<ID=chr20,length=64444167,assembly=gnomAD_GRCh38>
##contig=<ID=chr21,length=46709983,assembly=gnomAD_GRCh38>
##contig=<ID=chr22,length=50818468,assembly=gnomAD_GRCh38>
##contig=<ID=chrX,length=156040895,assembly=gnomAD_GRCh38>
##contig=<ID=chrY,length=57227415,assembly=gnomAD_GRCh38>
##contig=<ID=chrM,length=16569,assembly=gnomAD_GRCh38>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""


VCF_HEADER_GRCH37 = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
##contig=<ID=chr1,length=249250621,assembly=gnomAD_GRCh37>
##contig=<ID=chr2,length=243199373,assembly=gnomAD_GRCh37>
##contig=<ID=chr3,length=198022430,assembly=gnomAD_GRCh37>
##contig=<ID=chr4,length=191154276,assembly=gnomAD_GRCh37>
##contig=<ID=chr5,length=180915260,assembly=gnomAD_GRCh37>
##contig=<ID=chr6,length=171115067,assembly=gnomAD_GRCh37>
##contig=<ID=chr7,length=159138663,assembly=gnomAD_GRCh37>
##contig=<ID=chr8,length=146364022,assembly=gnomAD_GRCh37>
##contig=<ID=chr9,length=141213431,assembly=gnomAD_GRCh37>
##contig=<ID=chr10,length=135534747,assembly=gnomAD_GRCh37>
##contig=<ID=chr11,length=135006516,assembly=gnomAD_GRCh37>
##contig=<ID=chr12,length=133851895,assembly=gnomAD_GRCh37>
##contig=<ID=chr13,length=115169878,assembly=gnomAD_GRCh37>
##contig=<ID=chr14,length=107349540,assembly=gnomAD_GRCh37>
##contig=<ID=chr15,length=102531392,assembly=gnomAD_GRCh37>
##contig=<ID=chr16,length=90354753,assembly=gnomAD_GRCh37>
##contig=<ID=chr17,length=81195210,assembly=gnomAD_GRCh37>
##contig=<ID=chr18,length=78077248,assembly=gnomAD_GRCh37>
##contig=<ID=chr19,length=59128983,assembly=gnomAD_GRCh37>
##contig=<ID=chr20,length=63025520,assembly=gnomAD_GRCh37>
##contig=<ID=chr21,length=48129895,assembly=gnomAD_GRCh37>
##contig=<ID=chr22,length=51304566,assembly=gnomAD_GRCh37>
##contig=<ID=chrX,length=155270560,assembly=gnomAD_GRCh37>
##contig=<ID=chrY,length=59373566,assembly=gnomAD_GRCh37>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""

