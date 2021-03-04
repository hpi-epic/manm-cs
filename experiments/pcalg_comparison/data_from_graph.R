library(igraph)
library(optparse)
library(pcalg)

option_list_v <- list(
  # optparse does not support mandatory arguments so I set a value to NA by default to verify later if it was provided.
  make_option("--nSamples", type = "integer", default = NA, help = "number of samples to be generated"),
  make_option("--inputFile", type="character", help="", default=NA),
  make_option("--outputFile", type="character", help="", default=NA)
)
option_parser <- OptionParser(option_list = option_list_v)
opt <- parse_args(option_parser)

for (name in names(opt)) {
  if(is.na(opt[[name]])){
    stop(paste0("Parameter --", name, " is required"))
  }
}

input_graph_file <- opt$inputFile
output_file <- opt$outputFile
nSamples <- opt$nSamples

graph <- read.graph(input_graph_file, format="gml")
topo_graph <- topo_sort(graph)
graph_nel <- igraph.to.graphNEL(topo_graph)

dataset <- rmvDAG(nSamples,graph_nel)
colnames(dataset) <- as.character(as.numeric(colnames(dataset)) - 1) # otherwise the columns would be 1 indexed but we need 0 indices
write.csv(dataset, output_file, row.names = FALSE, quote = FALSE)




