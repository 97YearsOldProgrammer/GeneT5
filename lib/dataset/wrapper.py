import lib.dataset._parser     as parser
import lib.dataset._binary     as binary
import lib.dataset._chunking   as chunking
import lib.dataset._compacting as compacting
import lib.dataset._validation as validation
import lib.dataset._dataload   as dataload
import lib.dataset._util       as util


#####################
#####  Parser   #####
#####################


parse_fasta           = parser.parse_fasta
parse_gff             = parser.parse_gff
build_gene_index      = parser.build_gene_index
extract_feature_types = parser.extract_feature_types
extract_biotypes      = parser.extract_biotypes


#####################
#####  Binary   #####
#####################


BinaryChunk         = binary.BinaryChunk
write_binary        = binary.write_binary
read_binary         = binary.read_binary
get_binary_info     = binary.get_binary_info
read_chunk_at_index = binary.read_chunk_at_index


#####################
#####  Chunking #####
#####################


dynamic_chunking   = chunking.dynamic_chunking
augment_with_hints = chunking.augment_with_hints


######################
#####  Compacting ####
######################


compact_chunks           = compacting.compact_chunks
flatten_groups           = compacting.flatten_groups
pack_with_isolation      = compacting.pack_with_isolation
verify_isolation         = compacting.verify_isolation
build_segment_mask       = compacting.build_segment_mask
estimate_chunk_tokens    = compacting.estimate_chunk_tokens
align_to_block           = compacting.align_to_block
compute_effective_length = compacting.compute_effective_length


######################
#####  Validation ####
######################


build_validation_set       = validation.build_validation_set
save_validation_set        = validation.save_validation_set


#####################
#####  Dataload #####
#####################


BinaryTrainDataset      = dataload.BinaryTrainDataset
DynamicPaddingCollator  = dataload.DynamicPaddingCollator
SmartBatchSampler       = dataload.SmartBatchSampler


###################
#####  Util   #####
###################


append_tokens_to_txt = util.append_tokens_to_txt
format_size          = util.format_size
print_run_stats      = util.print_run_stats
print_validation_stats = validation.print_validation_stats