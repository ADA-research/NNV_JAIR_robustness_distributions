import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

from pathlib import Path
import numpy as np
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from autoverify.verifier import AbCrown
import pandas as pd
import onnxruntime as rt
import stat


from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler
from robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import BinarySearchEpsilonValueEstimator
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedShuffleSplit


CONFIG = {'cifar_7_1024': '/home/bosmanaw/experiments_JAIR_revision/CIFAR/config/cifar_7_1024.yaml',
            'conv_big':'/home/bosmanaw/experiments_JAIR_revision/CIFAR/config/cifar_conv_big.yaml',
            'resnet_4b': '/home/bosmanaw/experiments_JAIR_revision/CIFAR/config/resnet_4b.yaml', 
            'resnet_18': '/home/bosmanaw/experiments_JAIR_revision/CIFAR/config/cifar_resnet18.yaml'}

def get_balanced_sample(train_bool =True):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)


    data_transforms = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.ToTensor(), 
    torch.flatten
    ])
    if train_bool:
        torch_dataset= torchvision.datasets.CIFAR10(root="./data", train = True, download=True, transform=data_transforms)
           
    # Extract the labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=seed)  
        for train_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = train_idx  


    else: 
        torch_dataset= torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=data_transforms)
           
        # Extract the labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=seed)  
        for _,test_idx in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = test_idx  


    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(torch_dataset, balanced_sample_idx)

    return balanced_dataset




def submit_job_array(batch_contexts, slurm_script_template, slurm_scripts_path, array_counter, experiment_repository, config):
    
    name = slurm_scripts_path + str(array_counter) + ".txt"
    with open(name, 'w') as f:
        f.write("ArrayTaskID\tfile_verification_context\n")
        j =1 
        for i in batch_contexts:
            f.write(f"{j}\t{i}\n")
            j = j+1 


    array_end = len(batch_contexts)
    slurm_script_content = slurm_script_template.format(slurm_scripts_path=slurm_scripts_path, name = name,number_of_jobs=array_end, array=array_counter, base_path_experiment_repository=experiment_repository.base_path,network_folder=experiment_repository.network_folder, experiment_name=experiment_repository.get_act_experiment_path(), config=config)

    slurm_script_path = f"{slurm_scripts_path}slurmscript_array_{array_counter}.sh"
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script_content)
 
    os.chmod(slurm_script_path, stat.S_IRWXU)
    os.system(f'sbatch {slurm_script_path}')

def find_config(network_identifier, config_dict):
    for key, path in config_dict.items():
        if key in str(network_identifier.path):
            return path
    return None

def create_distribution_array(experiment_repository: ExperimentRepository, dataset: ExperimentDataset, dataset_sampler: DatasetSampler,  epsilon_value_estimator: EpsilonValueEstimator):
    slurm_scripts_path = f"{experiment_repository.get_act_experiment_path()}/slurm/"
    os.makedirs(f"{experiment_repository.get_act_experiment_path()}/slurm/")


    slurm_script_template = ("#!/bin/sh\n"
    "#SBATCH --job-name=robox\n"
    "#SBATCH --partition=graceGPU\n"
    "#SBATCH --exclude=ethnode[07]\n"
    "#SBATCH --array=1-{number_of_jobs}\n"
    "#SBATCH --output={slurm_scripts_path}/{array}_%j.out\n"

    "config={name}\n"
    "file_verification_context=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {{print $2}}' $config)\n"
    "srun python /home/bosmanaw/experiments_JAIR_revision/VERONA/scripts/JAIR/CIFAR/one_multiple_jobs.py --file_verification_context $file_verification_context "
    "--base_path_experiment_repository {base_path_experiment_repository} "
    "--network_folder {network_folder} --experiment_name {experiment_name} --config {config}\n")

    
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    os.makedirs(f"{experiment_repository.get_act_experiment_path()}/yaml")

    batch_contexts = []
    batch_size = 500
    array_counter = 0

    for network in network_list:

        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except:
            logging.info(f"failed for network: {network}")
            failed_networks.append(network)
            continue
        config = find_config(network, CONFIG)
        for data_point in sampled_data:
         
            verification_context = experiment_repository.create_verification_context(network, data_point)
            now = datetime.now()
            now_string = now.strftime("%d-%m-%Y+%H_%M_%S_%f")[:-3]
            file_path = Path(f"{experiment_repository.get_act_experiment_path()}/yaml/verification_context_{now_string}.yaml")
       
            file_verification_context = experiment_repository.save_verification_context_to_yaml(file_path, verification_context)

            batch_contexts.append(file_verification_context)
            if len(batch_contexts) >= batch_size:
                submit_job_array(batch_contexts, slurm_script_template, slurm_scripts_path, array_counter,experiment_repository, config)
                array_counter += 1
                batch_contexts.clear()
           

    # Submit any remaining contexts in the batch
    if batch_contexts:
        submit_job_array(batch_contexts, slurm_script_template, slurm_scripts_path, array_counter, experiment_repository, config)
   
    logging.info(f"Failed for networks: {failed_networks}")




def main():

    torch_dataset = get_balanced_sample(train_bool=True)


    timeout = 3600
    epsilon_list = np.arange(0.00,0.4,0.0039)

    experiment_repository_path = Path(f'/home/bosmanaw/experiments_JAIR_revision/CIFAR')
    network_folder = Path("/home/bosmanaw/experiments_JAIR_revision/CIFAR/networks")

    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    # dataset = ImageFileDataset(image_folder=Path('/home/bosmanaw/experiments_JAIR_revision/GTSRB/images'), label_file=Path('/home/bosmanaw/experiments_JAIR_revision/GTSRB/instances.csv'))
    
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    # Create sampler for correct predictions or incorrect predictions
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
 
    #create the experiment based on a name you choose
    experiment_name = "JAIR_CIFAR_TRAIN_3600"
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(dict(
                                        experiment_name=experiment_name, experiment_repository_path=str(experiment_repository_path),
                                        network_folder=str(network_folder), dataset=str(dataset),
                                        timeout=timeout, epsilon_list=[str(x) for x in epsilon_list]))

    create_distribution_array(experiment_repository, dataset, dataset_sampler, epsilon_list)

if __name__ == "__main__":
    main()



import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


import argparse
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import BinarySearchEpsilonValueEstimator
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from pathlib import Path
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from autoverify.verifier import AbCrown, OvalBab, Nnenum
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--file_verification_context', type =Path)
parser.add_argument('--base_path_experiment_repository')
parser.add_argument('--network_folder')
parser.add_argument('--experiment_name')
parser.add_argument('--config')
args = parser.parse_args()
    
baby_experiment_repository = ExperimentRepository(base_path=Path(args.base_path_experiment_repository), network_folder=Path(args.network_folder))
baby_experiment_repository.load_experiment(experiment_name= args.experiment_name)

verifier = AutoVerifyModule(verifier=AbCrown(), property_generator=One2AnyPropertyGenerator(number_classes=10, data_lb=0, data_ub=1),  timeout=3600, config = Path(args.config))
eps_list = np.arange(0.00,0.4,0.0039)
epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list= eps_list.copy(), verifier=verifier)



verification_context = baby_experiment_repository.load_verification_context_from_yaml(Path(args.file_verification_context))

epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)
baby_experiment_repository.save_result(epsilon_value_result)


