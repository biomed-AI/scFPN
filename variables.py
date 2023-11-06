#!/usr/bin/env python3
"""
device specific configuration
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

