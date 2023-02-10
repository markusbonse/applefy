# How to get the data

This short tutorial explains how to get the dataset and intermediate results 
necessary to reproduce the results and plots of the 
[Apples with Apples](../05_citation.rst) paper. 

## Downloading the data from Zenodo
The data is publicly available at 
[Zenodo](https://zenodo.org/record/7443481#.Y-acmy2cYUE). Please download and 
unpack the file: `apples_root_dir.zip`. Once unzipped, the directory should 
contain three subdirectories:

1. `30_data`: This is the raw NACO L' Beta Pic dataset used in the paper in two 
versions: `betapic_naco_lp_HR.hdf5` the vanilla dataset after pre-processing with
[PynPoint](https://pynpoint.readthedocs.io/en/latest/) but with the full 
temporal resolution. `betapic_naco_lp_LR.hdf5` is the same dataset but with 
reduced resolution in time (temporal binned / averaged). 
The plots in the paper are calculated for the `betapic_naco_lp_HR.hdf5` dataset.
But all the code should work (and be much faster) with the lower resolution data.
2. `70_results`: Contains all final and intermediate results.
3. `lookup_tables`: Are the lookup tables for the LaplaceBootstrapTest.

## Setting up the environmental variable

Once downloaded, we need to tell applefy where the data is on your local 
machine. You can do this by setting the following environment variable:

```bash
export APPLES_ROOT_DIR="/path/to/datasets/dir" ;
```

You are ready to go!
