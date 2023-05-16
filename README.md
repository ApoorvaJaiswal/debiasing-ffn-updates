# debiasing-ffn-updates

All notebooks in this repo can be run on Google Colab with GPU enabled:

1. To run a given notebook, open a new notebook from github:
File -> Open Notebook -> Github -> Paste the link to this github repo
3. Edit -> Notebook Settings -> Hardware Accelerator: GPU, GPU Type: V100
4. Open a terminal instance 
5. Clone our repo: `git clone https://github.com/apeterson7/debiasing-ffn-updates.git`
6. Change permissions of the setup script `chmod 777 debiasing-fnn-updates/setup.sh`
7. Run the setup script `./debiasing-fnn-updates/setup.sh`
    (this just moves directories to root and installs transformers)
8. Run the notebook cells!

How To Guide:

1. Selecting Value Vectors (Section 5): 
    - `value_vector_scans/value_vector_scans.ipynb` : Identifies value vectors using word lists for protected groups
    - `notebooks/De-EmphasizingWithValueVectors.ipynb`: Experiments done to analyze the impact of de-emphasizing value vectors with various co-efficient values
    - `notebooks/ExperimentingWithValueVectors.ipynb` : Experiments done with various techniques to automate value vector identification. Includes Language detection, sentiment analysis, similarity and Perspective API usage.
    - `notebooks/FindingPositiveValueVectorsForEachGroup.ipynb` : Identifies value vectors using the automated approach for each protected group 
	
2. Running Big Bench Tasks (Section 6): `notebooks/big_bench_tests.ipynb`
3. Fine Tuning GPT-2 (Section 7): `notebooks/fine_tuning_gpt2.ipynb`
4. Examing Results / Visualizations: `examine_results/examine_results.ipynb`
    - Note: the results for each big bench task are pickled in this directory
    the naming convention is <model>-<test_name>-<config>.pkl
5. Qualitative Results (Section 9): `notebooks/unqover_qualitative_results.ipynb`

Note: Please be advised that some of our notebooks generate tabular results in latex format to speed up writing our report!  We didn't change this back to a nice format for jupyter in some places.