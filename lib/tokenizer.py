from __future__ import annotations

import gzip
import sys
import os
import re
import typing as tp
import torch

from pathlib        import Path
from contextlib     import closing
from dataclasses    import dataclass


def anti(seq):
	comp = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
	anti = seq.translate(comp)[::-1]
	return anti


#####################
## UTILITY SECTION ##
#####################


def getfp(filename):
    
	if   filename.endswith('.gz'):
		return gzip.open(filename, 'rt', encoding='ISO-8859-1')
	elif filename == '-':
		return sys.stdin
	return open(filename)

def read_fasta(filename):

	name = None
	seqs = []

	fp = getfp(filename)

	for line in fp:
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:].split()[0]
				seqs = []
			else:
				name = line[1:].split()[0]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()

def find_files(input_path):

	fa      = re.compile(r'\.(fasta|fa|fna)(\.gz)?$', re.IGNORECASE)
	gff     = re.compile(r'\.(gff|gff3)(\.gz)?$', re.IGNORECASE)
 
	fastas  = []
	gffs    = []
	
	input_path = Path(input_path)
	
	if not input_path.exists():
		raise FileNotFoundError(f"Input path does not exist: {input_path}")
	
	if not input_path.is_dir():
		raise ValueError(f"Input path must be a directory: {input_path}")
	
	for root, dirs, files in os.walk(input_path):
		for file in files:
			filepath = os.path.join(root, file)
			
			if fa.search(file):
				fastas.append(filepath)
			elif gff.search(file):
				gffs.append(filepath)
	
	return fastas, gffs

@dataclass
class Feature:
    """ Parse Single GFF line"""

    seqid:  str
    source: str
    typ:    str
    beg:    int
    end:    int
    score:  tp.Optional[float]
    strand: str
    phase:  tp.Optional[int]
    att:    tp.Dict[str, str]

def parse_att(att):

    attributes = {}

    if not att or att == ".":
        return attributes

    for stuff in att.split(";"):
        stuff = stuff.strip()
        if not stuff:
            continue
        if "=" in stuff:
            key, value = stuff.split("=", 1)
        elif " " in stuff:
            key, value = stuff.split(" ", 1)
        else:
            key, value = stuff, ""
        attributes[key.strip()] = value.strip()

    return attributes

def parse_gff(filename):
    
    fp          = getfp(filename)
    features    = []

    with closing(fp):
        for line in fp:
            
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("\t")
            if len(line) != 9:
                continue
            
            seqid, source, typ, beg, end, score, strand, phase, att = line
            score = None if score == "." else float(score)
            phase = None if phase == "." else int(phase)
            
            att   = parse_att(att)
            if not att:
                continue
            
            feature = Feature(
                seqid=  seqid,
                source= source,
                typ=    typ.lower(),
                beg=    int(beg),
                end=    int(end),
                score=  score,
                strand= strand,
                phase=  phase,
                att=    att,
            )
            features.append(feature)

    return features

def choose_parent_id(feature):
    
    if "Parent" in feature.att:
        return feature.att["Parent"].split(",")[0]
    if "ID" in feature.att:
        return feature.att["ID"]
    return feature.seqid

def group_features(features, filter):

    group = {}

    for feature in features:
        if feature.typ not in filter: continue
        parent_id = choose_parent_id(feature)
        group.setdefault(parent_id, []).append(feature)

    return group

def build_line(features, seqid, strand, seq):

    line = []

    # iterate through all feature under a parent ID
    for feature in sorted(features, key=lambda x: x.beg):

        if feature.seqid != seqid:
            raise ValueError(
                f"Transcript {seqid} has exons on multiple sequences "
                f"({seqid} vs {feature.seqid})"
            )
            
        if feature.strand != strand:
            raise ValueError(
                f"Transcript {seqid} mixes strands ({strand} vs {feature.strand})"
            )
        
        word = seq[feature.beg-1 : feature.end]
        if strand == "-":
            word = anti(word)
        line.append(word)

    return line

def build_transcript(grouped, sequences):

    transcripts = {}
    
    for parent_id, features in grouped.items():
        if not features:
            continue
        
        seqid   = features[0].seqid
        strand  = features[0].strand
        
        if seqid not in sequences:
            raise KeyError(
                f"Sequence '{seqid}' referenced in GFF but absent from FASTA"
            )

        seq = sequences[seqid]
        transcripts[parent_id] = build_line(features, seqid, strand, seq)

    return transcripts

def tokenize_transcripts(transcripts, tokenizer):

    tokenized = {}
    
    for parent_id, segments in transcripts.items():
        token_ids = []
        for segment in segments:
            token_ids.extend(tokenizer(segment))
        tokenized[parent_id] = token_ids

    return tokenized


######################
#### Tokenisation ####
######################


BASE_PAIR   = ("A", "C", "G", "T")
BASE2IDX    = {base: idx for idx, base in enumerate(BASE_PAIR)}

def apkmer(k: int):

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if k == 1:
        return list(BASE_PAIR)

    prev_kmers = apkmer(k - 1)
    return [prefix + base for prefix in prev_kmers for base in BASE_PAIR]

@dataclass
class KmerTokenizer:
    """DNA seq to kmer ids by sliding window algo"""

    k           : int
    stride      : int = 1
    vocabulary  : list = None
    unk_token   : str = None

    def __post_init__(self):
        
        # map allkmer with a int
        self.token2id = {token: idx for idx, token in enumerate(self.vocabulary)}

        # map the unknown bps as last element
        if self.unk_token is not None and self.unk_token not in self.token2id:
            self.token2id[self.unk_token] = len(self.token2id)

    def __call__(self, seq):
        
        seq     = seq.upper()
        tokens  = []

        # sliding window algo
        for t in range(0, max(len(seq) - self.k + 1, 0), self.stride):
            token = seq[t:t+self.k]
            tokens.append(self.token2id.get(token, self.token2id.get(self.unk_token, 0)))
        if not tokens:
            tokens.append(self.token2id.get(self.unk_token, 0))
        return tokens


################
#### Output ####
################


def write_tokenized_corpus(output_path, tokenized):

    with open(output_path, "w", encoding="utf-8") as fp:
        for parent_id in sorted(tokenized):
            token_line = " ".join(str(token_id) for token_id in tokenized[parent_id])
            fp.write(token_line + "\n")


###################
#### Embedding ####
###################


def parse_glove_matrix(vectors):

    embeddings = {}
    
    with open(vectors, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            if not parts:
                continue
            
            idx     = int(parts[0])
            vector  = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            embeddings[idx] = vector
    
    return embeddings

def embed_sentence(line, embeddings):

    idxs = [int(x) for x in line.strip().split()]
    
    if not idxs:
        return torch.tensor([])

    embeddings = [embeddings[token_id] for token_id in idxs]
    # stack all embedding vectors
    return torch.stack(embeddings, dim=0)

def embed_whole_corpus(corpus, embeddings):

    with open(corpus, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            embedded = embed_sentence(line, embeddings)
            yield embedded