import lib.data._parser     as parser
import lib.data._binary     as binary
import lib.data._merge      as merge
import lib.data._chunking   as chunking
import lib.data._dataload   as dataload
import lib.data._ram        as ram
import lib.data._util       as util


#####################
#####  Parser   #####
#####################


lazy_fasta_open                = parser.lazy_fasta_open
parse_gff_to_gene_index        = parser.parse_gff_to_gene_index
filter_canonical_transcripts   = parser.filter_canonical_transcripts
find_genome_files              = parser.find_genome_files
save_gene_index                = parser.save_gene_index
load_gene_index                = parser.load_gene_index
extract_coding_genes           = parser.extract_coding_genes
build_eval_sample              = parser.build_eval_sample
select_diverse_samples         = parser.select_diverse_samples


#####################
#####  Binary   #####
#####################


BinaryChunk         = binary.BinaryChunk
write_binary        = binary.write_binary
read_binary         = binary.read_binary
get_binary_info     = binary.get_binary_info
get_chunk_count     = binary.get_chunk_count
read_chunk_at_index = binary.read_chunk_at_index
iter_binary           = binary.iter_binary
iter_binary_token_ids = binary.iter_binary_token_ids
merge_binary_files    = merge.merge_binary_files


#####################
#####  Chunking #####
#####################


sliding_window_chunking = chunking.sliding_window_chunking
augment_with_hints      = chunking.augment_with_hints
filter_n_heavy_chunks   = chunking.filter_n_heavy_chunks


#####################
#####  Dataload #####
#####################


create_train_pipeline  = dataload.create_train_pipeline
BinaryTrainDataset     = dataload.BinaryTrainDataset
DEFAULT_MAX_SEQ        = dataload.DEFAULT_MAX_SEQ
BUDGET_SEQ             = dataload.BUDGET_SEQ
PACK_SEQ_LEN           = dataload.PACK_SEQ_LEN
token_budget_batcher   = dataload.token_budget_batcher
packed_collate         = dataload.packed_collate


####################
#####  RAM I/O #####
####################


consolidate          = ram.consolidate
RamDataset           = ram.RamDataset
TokenBudgetSampler   = ram.TokenBudgetSampler
compress_dataset     = ram.compress_dataset
decompress_dataset   = ram.decompress_dataset


###################
#####  Util   #####
###################


format_size     = util.format_size
print_run_stats = util.print_run_stats
