import os
import time
from dataclasses import dataclass
from typing import Dict

import experiments.src.adapters.backend_adapter as BackendAdapter
from experiments.src.tree.base_node import BaseNode
from experiments.src.tree.node_type import NodeType
import docker


@dataclass
class ResolvedJob:
    job_id: int
    result: Dict
    ground_truth_compare: Dict

class JobNode(BaseNode):
    type = NodeType.JOB
    
    # Information from previous dataset node
    dataset_node = None
    experiment_node = None

    def __init__(self, parent):
        super(JobNode, self).__init__()
        self.parent = parent
        self.name = "Waiting Job"

        self.node = "minikube" #TODO
        self.docker = docker.from_env()

        self.experiment_node = self.get_parent_with_type(NodeType.EXPERIMENT)

    def resolve_impl(self):
        experiment_id = self.experiment_node.resolved_data.experiment_id
        job = BackendAdapter.BackendAdapter.instance().start_job(
            experiment_id=experiment_id
        )
        job_id = job["id"]
        self.name = f"Job #{job_id}"

        #self._run_job_with_docker(job_id)

        job_status = "init"
        job_response_json = None

        while(job_status != "error" and  job_status != "done"):
            job_response = BackendAdapter.BackendAdapter.instance().get_job(job_id)
            job_status = job_response["status"]
            job_response_json= job_response

            time.sleep(10) #TODO env

        ground_truth_compare = None
        if job_status == "done":
            job_result = job_response_json["result"]
            ground_truth_compare = BackendAdapter.BackendAdapter.instance().get_ground_truth_compare(job_result["id"])

        return ResolvedJob(job_id=job_id, result=job_response_json, ground_truth_compare=ground_truth_compare)

    def _run_job_with_docker(self, job_id):
        log_file = os.path.join("./logs", f"{job_id}.log") #TODO env
        experiment_response = self.experiment_node.resolved_data.response

        execution_command = [experiment_response['algorithm']['script_filename'], "-j", job_id, "-d", str(experiment_response['dataset_id']), "--api_host", "localhost:5000", "--send_sepsets", "0"]
        for k, v in experiment_response['parameters'].items():
            execution_command.append('--' + k)
        execution_command.append(str(v))

        with open(log_file, 'w') as log_file_handle:
            log_file_handle.write(" ".join(execution_command))
            log_file_handle.write("\n")
            ls_output = self.docker.containers.run("mpci/mpci_execution_r", execution_command, detach=False)

        return ls_output
