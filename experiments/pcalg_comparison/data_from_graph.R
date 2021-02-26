library(igraph)
library(optparse)
library(pcalg)

input_graph_file <- "/home/jonas/Code/mpci2/services/executionenvironments/r/generator/gt-download.gml"
output_file <- "/home/jonas/Code/mpci-dag/experiments/pcalg_comparison/out.csv"
graph <- read.graph(input_graph_file, format="gml")
graph_nel <- igraph.to.graphNEL(graph)

nSamples <- 200000
dataset <- rmvDAG(nSamples,graph_nel)
write.csv(dataset, output_file, row.names = FALSE, quote = FALSE)
# option_list_v <- list(
#   # optparse does not support mandatory arguments so I set a value to NA by default to verify later if it was provided.
#   make_option("--nSamples", type = "integer", default = NA, help = "number of samples to be generated"),
#   make_option("--edgeValueLowerBound", type = "double", default = NA, help = "lowest possible edge value"),
#   make_option("--edgeValueUpperBound", type = "double", default = NA, help = "highest possible edge value")
# )
#
# option_parser <- OptionParser(option_list = option_list_v)
# opt <- parse_args(option_parser)
#
# for (name in names(opt)) {
#   if(is.na(opt[[name]])){
#     stop(paste0("Parameter --", name, " is required"))
#   }
# }



