import os
import time

ground_truth = "graph"
samples = "data"
test_folder = "test_pypi"

def test_manm_cs_pypi():

    os.system(
        "python3 -m pip install . &&" + # install dependencies from actual pip because they are not on testpypi
        f"mkdir -p {test_folder} && cd {test_folder} && rm -f {samples} {ground_truth} && " +
        "python3 -m pip uninstall -y manm-cs &&" + # uninstall the package itself so we can install it from testpypi
        "python3 -m pip install -i https://test.pypi.org/simple/ manm-cs &&" + # install latest manm-cs from testpypi
        f"python3 -m manm_cs --num_nodes 10 --edge_density 0.5 --num_samples 10000 --discrete_node_ratio 0.5 --output_ground_truth_file \"{ground_truth}\" --output_samples_file \"{samples}\"")

    assert os.path.isfile(test_folder + "/" + ground_truth + ".gml")
    assert os.path.isfile(test_folder + "/" + samples + ".csv")
