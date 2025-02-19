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
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedShuffleSplit


CONFIG = '/home/bosmanaw/experiments_JAIR_revision/GTSRB/config/gtsrb.yaml'

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
        torch_dataset= torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=data_transforms)
           
    # Extract the labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=seed)  #TODO
        for train_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = train_idx  


    else: 
        torch_dataset= torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=data_transforms)
           
        # Extract the labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=seed)  
        for _,test_idx in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = test_idx  


    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(torch_dataset, balanced_sample_idx)

    return balanced_dataset

def create_distribution(experiment_repository: ExperimentRepository, dataset: ExperimentDataset, dataset_sampler: DatasetSampler, epsilon_value_estimator: EpsilonValueEstimator, property_generator: PropertyGenerator):
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    for network in network_list:
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except:
            logging.info(f"failed for network: {network}")
            failed_networks.append(network)
            continue
        for data_point in sampled_data:
            verification_context = experiment_repository.create_verification_context(network, data_point, property_generator)

            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

            experiment_repository.save_result(epsilon_value_result)

    experiment_repository.save_plots()
    logging.info(f"Failed for networks: {failed_networks}")




def main():
    #TODO: change to "test" if want the test set. 
    split = "train"

    #load dataset
    torch_dataset = get_balanced_sample(train_bool=(split == 'train')) #Added check to see whether the split is train

    #set per-query timeout and the list of epsilons to verify
    timeout = 3600
    epsilon_list = np.arange(0.00,0.4,0.0039)
    
    #TODO: change to correct path
    experiment_repository_path = Path(f'./GTSRB')
    
    #TODO: change network folder path as necessary
    network_folder = Path("../networks_onnx/gtsrb")


    #Create dataset and experiment repo 
    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    # Create sampler for correct predictions or incorrect predictions
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
 
    #create the experiment based on a name you choose
    experiment_name = f"JAIR_GTSRB_{split}" #TODO: adjust name as prefered"

    experiment_repository.initialize_new_experiment(experiment_name)

    #Define verifier, we have used abCrown and the one2any property generator (untargeted verification)
    verifier = AutoVerifyModule(verifier=AbCrown(),  timeout=timeout, config = CONFIG)
    property_generator=One2AnyPropertyGenerator(number_classes=43, data_lb=0, data_ub=1)

    #initiate the binary search 
    epsilon_value_estimator= BinarySearchEpsilonValueEstimator(epsilon_value_list= epsilon_list.copy(), verifier=verifier)
    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)

if __name__ == "__main__":
    main()

