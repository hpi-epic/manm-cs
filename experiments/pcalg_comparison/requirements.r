# Before running this create an R virtual environment with
# conda create -n renv -c conda-forge r-base=4.0.3
# Activate it using
# conda activate renv
# You can activate this environment inside your python environment
# Then run
# Rscript requirements.r
# <-------------------------------------------------
# CRAN dependencies
packages <- c('igraph', 'optparse', 'pcalg')
# non-CRAN dependencies
bioc_packages <- c('graph', 'RBGL')
# <-------------------------------------------------
#
#
#

r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(version = "3.12")

not_installed <- function(pkg){
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    return(length(new.pkg) > 0)
}

# source('http://www.bioconductor.org/biocLite.R');
for (pkg in bioc_packages){
    if(not_installed(pkg)){
        BiocManager::install(pkg);
    }
}

for (pkg in packages){
    if(not_installed(pkg)){
        install.packages(pkg)
        sapply(pkg, require, character.only = TRUE)
    }
}
