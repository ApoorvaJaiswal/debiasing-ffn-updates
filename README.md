# debiasing-ffn-updates

Running an example test in Colab on GPUs
1. Clone debiasing-ffn-updates
2. Edit -> Notebook Settings -> enable GPU
3. move bias_evaluation and model_wrappers to root directory
4. install transformers with pip
5. Run Big_Bench_Example.ipynb 

Pickling Results

The last two cells of Big_Bench_Example.ipynb show how to serialize the results dictionary.
Please follow this naming convention to keep track of results
<model>-<big_bench_test_name>-<config_number>.pkl