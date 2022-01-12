import os
import time

ground_truth = "./ground_truth.gml"
samples = "./samples.csv"

def test_manm_cs_pypi():

    if os.path.isfile(ground_truth):
        os.rename(ground_truth, f"./ground_truth_old_{int(time.time())}.gml")
    if os.path.isfile(samples):
        os.rename(samples, f"./samples_old{int(time.time())}.csv")

    os.system(
        "python3 -m pip install . &&" + # install dependencies from actual pip because they are not on testpypi
        "python3 -m pip install -i https://test.pypi.org/simple/ manm-cs &&" + # install latest manm-cs from testpypi
        "python3 -m manm_cs --num_nodes 10 --edge_density 0.5 --num_samples 10000 --discrete_node_ratio 0.5")

    assert os.path.isfile(ground_truth)
    assert os.path.isfile(samples)
