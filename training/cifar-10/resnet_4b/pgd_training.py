from random import shuffle
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from pathlib import Path
import optuna
from optuna.trial import TrialState
\
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from ....networks_pytorch.resnet_4b import ResNet4B
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule

pgd_attack = PGDAttack(epsilon_step_size=2/255, number_iterations=7, random_init=True)

ATTACK_EPSILON= 8/255

def objective(trial):
    network = ResNet4B(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    optimizer_name = "Adam"
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 1, 10, log=True)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.01, 1, log=True)
    attack_epsilon = ATTACK_EPSILON
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    dataset = torchvision.datasets.CIFAR10('../../data', train=True, download=False,
                    transform=torchvision.transforms.ToTensor())

    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, (0.8, 0.2))
    train_sampler = torch.utils.data.RandomSampler(data_source=train_dataset, num_samples=8000)
    validation_sampler = torch.utils.data.RandomSampler(data_source=validation_dataset, num_samples=2000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, sampler=validation_sampler)

    training_module = StandardTrainingModule(criterion=criterion, attack=pgd_attack, epsilon=attack_epsilon)

    for epoch in range(0,40):
        network.train()
        train_accuracy = training_module.train(train_loader, network, optimizer)
        scheduler.step()

        network.eval()
        test_module = StandardTestModule(attack=pgd_attack, epsilon=attack_epsilon)
        attack, epsilon, test_accuracy = test_module.test(validation_loader, network)

        trial.report(test_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_accuracy

if __name__ == "__main__":
    torch.manual_seed(0)

    study = optuna.create_study(direction="maximize", storage="sqlite:///pgd_training.db")
    study.optimize(objective, n_trials=300, timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    training_parameters = AttributeDict(learning_rate=trial.params["lr"], 
                                        weight_decay=trial.params["weight_decay"], 
                                        scheduler_step_size=trial.params["scheduler_step_size"], 
                                        scheduler_gamma=trial.params["scheduler_gamma"],
                                        attack_epsilon=ATTACK_EPSILON,
                                        batch_size=128)
    """ training_parameters = AttributeDict(
        learning_rate = 0.00149,
        weight_decay = 0.00001,
        scheduler_step_size=8,
        scheduler_gamma=0.041,
        attack_epsilon=8/255,
        batch_size=128)  """
    
    network = ResNet4B(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    optimizer = getattr(optim, 'Adam')(network.parameters(), lr=training_parameters.learning_rate, weight_decay=training_parameters.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_parameters.scheduler_step_size, gamma=training_parameters.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    train_dataset = torchvision.datasets.CIFAR10('../../data', train=True, download=False,
                    transform=torchvision.transforms.ToTensor())
    
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, (0.98, 0.02))
    test_dataset = torchvision.datasets.CIFAR10('../../data', train=False, download=False,
                    transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    training_stack = []
    training_stack.append((50, StandardTrainingModule(criterion=criterion, attack=pgd_attack, epsilon=training_parameters.attack_epsilon)))

    testing_stack = [
        StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=1/255),
        StandardTestModule(attack=FGSMAttack(), epsilon=4/255),
        StandardTestModule(attack=FGSMAttack(), epsilon=8/255),
        StandardTestModule(attack=pgd_attack, epsilon=1/255),
        StandardTestModule(attack=pgd_attack, epsilon=4/255),
        StandardTestModule(attack=pgd_attack, epsilon=8/255),
    ]

    """ StandardTestModule(attack=AutoAttackModule(), epsilon=0.1),
        StandardTestModule(attack=AutoAttackModule(), epsilon=0.2),
        StandardTestModule(attack=AutoAttackModule(), epsilon=0.3), """
    
    training_objects = AttributeDict(criterion=str(criterion), 
                                     optimizer=str(optimizer), 
                                     network=str(network),
                                      training_stack=[str(x[0]) + "_" + str(x[1]) for x in training_stack], 
                                      testing_stack=[str(x) for x in testing_stack])
    
    in_training_validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1024)

    validation_module = StandardTestModule(attack=pgd_attack, epsilon=8/255)

    experiment_tracker = ExperimentTracker("resnet_4b-pgd", Path("./generated"), login=True)

    experiment_tracker.initialize_new_experiment("no-batch-norm", training_parameters=training_parameters | training_objects)

    pipeline = Pipeline(experiment_tracker, training_parameters, criterion, optimizer, scheduler)

    pipeline.train(train_loader, network, training_stack, validation_module=validation_module, in_training_validation_loader=in_training_validation_loader)
    #pipeline.train(train_loader, network, training_stack)

    network = experiment_tracker.load_trained_model(network)

    pipeline.test(network, test_loader, testing_stack=testing_stack)
