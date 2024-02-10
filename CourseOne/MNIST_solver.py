import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# MNIST is a default dataset: we can load from torchvision directly
from torchvision import datasets

# used for storing the current best params set
import copy

# for reproducibility
torch_seed = 11
torch.manual_seed(torch_seed)


def create_subplot_image(axis_ref, data_structure, lbl_id):
    """

    :param axis_ref:
    :param data_structure:
    :param lbl_id:
    :return:
    """
    axis_ref.imshow(data_structure.data[lbl_id, :, :].numpy(), cmap='gray')
    axis_ref.axis('off')
    axis_ref.title.set_text(f"Label: {data_structure.targets[lbl_id]}")
    pass  # only side effects (maybe should return the axis?)


def create_panel_of_consecutive_ex_images(init_idx, input_data, panel_dims=(3, 5)):
    fig, ax = plt.subplots(*panel_dims)
    x_plt_index, y_plt_index = np.meshgrid(np.arange(panel_dims[0]),
                                           np.arange(panel_dims[1]))
    sequential_indexed_couple = zip(x_plt_index.flatten(), y_plt_index.flatten())
    [create_subplot_image(ax[plot_tuple_idx[0], plot_tuple_idx[1]],
                          input_data,
                          init_idx + seq_idx) for seq_idx, plot_tuple_idx in enumerate(sequential_indexed_couple)]
    pass


# values inspired by this discussion
# [1] https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset

# to check my reasoning check the values
def get_max_n_normalized_mean_n_std(input_data):
    max_data_value = input_data.data.max()  # should be 255
    img_mean = torch.mean(input_data.data, dtype=torch.float32)/max_data_value  # mean of inputs,
    # when data are scaled between 0 and 1 (from ToTensor)
    img_std = input_data.data.to_dense().numpy().std()/max_data_value
    return max_data_value, img_mean, img_std


def get_data_loaded_in_batches(current_data, batch_size, shuffle=True):
    return DataLoader(current_data, batch_size=batch_size, shuffle=shuffle)


def get_train_and_test_data_w_batch_size(batch_size, train_data, val_data, test_data):
    return get_data_loaded_in_batches(train_data, batch_size, shuffle=True), \
           get_data_loaded_in_batches(val_data, batch_size, shuffle=True), \
           get_data_loaded_in_batches(test_data, batch_size, shuffle=False)


# implementation inspired by the lectures
# and by https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

class MNIST_MLP(nn.Module):
    def __init__(self, network_input_dim, n_nodes, n_classes=10):
        super().__init__()

        self.n_classes = n_classes
        self.activation = F.relu
        self.output = F.log_softmax  # multiple category
        self.fc1 = nn.Linear(network_input_dim, n_nodes[0])
        self.fc2 = nn.Linear(n_nodes[0], n_nodes[1])
        self.fc3 = nn.Linear(n_nodes[1], n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(self.fc3(x), dim=1)
        return x


def get_model_device(model):
    #  adapted from:
    #  https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    return next(model.parameters()).device


def get_num_of_correctly_predicted_samples(predicted, labels):
    predicted_labels = torch.argmax(predicted.detach(), 1)  # possibly detach is not necessary here. Please clarify
    return torch.count_nonzero(torch.eq(predicted_labels, labels))


def evaluation_step(loss, _):
    return loss


def training_step(loss, optimizer):
    # init gradient
    optimizer.zero_grad()

    # compute gradient
    loss.backward()

    # update the model's weights
    optimizer.step()
    return loss


def batch_process(inputs, labels, optimizer, model, processing_fcn,
                  criterion=F.cross_entropy):
    # estimate model's prediction
    output = model(inputs)

    # compute loss
    loss = criterion(output, labels)

    # do nothing or train, depending on the processing function
    loss = processing_fcn(loss, optimizer)

    # get the number of rightly predicted items
    correctly_predicted_samples = get_num_of_correctly_predicted_samples(output, labels)
    return loss, correctly_predicted_samples


def process_all_data_from_loader_n_get_metrics(data_loader, optimizer, model, processing_fcn,
                                               criterion=F.cross_entropy):
    device = get_model_device(model)

    running_loss = 0.0
    train_correct = 0  # init number of correctly classified items

    for index, data in enumerate(data_loader):
        inputs, labels = data

        # dump data to device
        inputs, labels = inputs.to(device), labels.to(device)

        current_loss, correctly_classified_items = batch_process(inputs, labels, optimizer, model,
                                                                 processing_fcn,
                                                                 criterion)  # batch_process
        running_loss += current_loss.item()
        train_correct += correctly_classified_items

    return running_loss, train_correct


def train_net_on_data(data_loader, optimizer, model,
                      criterion=F.cross_entropy):
    model.train()
    return process_all_data_from_loader_n_get_metrics(data_loader, optimizer, model,
                                                      training_step, criterion)


def eval_net_on_data(data_loader, model,
                     criterion=F.cross_entropy):
    model.eval()
    return process_all_data_from_loader_n_get_metrics(data_loader,
                                                      None, model, evaluation_step,
                                                      criterion)


class PerformanceImprover:
    def __init__(self, max_count=5, window=2, loss_threshold=1.0e-2, acc_threshold=1.0e-2):
        self.best_val_loss_mean = np.infty
        self.best_val_acc_mean = 0.0
        self.counter = 0
        self.max_count = max_count
        self.loss_threshold = loss_threshold
        self.acc_threshold = acc_threshold
        self.window = window

    def is_improving(self, val_loss, val_accuracy):

        def get_mean_of_last_n_elements(list_of_values, num_of_elements):
            return np.mean(list_of_values[-num_of_elements:])  # handles automatically list shorter than num_of_elements

        if not val_loss:
            return True

        self.counter += 1

        current_val_loss_mean = get_mean_of_last_n_elements(val_loss, self.window)
        current_val_acc_mean = get_mean_of_last_n_elements(val_accuracy, self.window)

        loss_has_improved = current_val_loss_mean < (self.best_val_loss_mean - self.loss_threshold)
        acc_has_improved = current_val_acc_mean > (self.best_val_acc_mean + self.acc_threshold)

        if loss_has_improved or acc_has_improved:

            self.counter = 0  # reset counter

            self.best_val_loss_mean = np.min([current_val_loss_mean, self.best_val_loss_mean])
            self.best_val_acc_mean = np.max([current_val_acc_mean, self.best_val_acc_mean])
            return True

        elif self.counter > self.max_count:
            return False

        else:
            return True


class TrainingStopper:
    def __init__(self, is_improving, max_epochs=500, hist_horizon=20):
        self.max_epochs = max_epochs
        self.hist_horizon = hist_horizon
        self.current_epoch = 0
        self.is_improving = is_improving

    def __str__(self):
        return f"""Class Parameters:\n
                   Max Epochs: {self.max_epochs} 
                   History Horizon: {self.hist_horizon}
                   Current Epoch:{self.current_epoch}"""

    def keep_training(self, val_loss, val_accuracy):
        if self.current_epoch >= self.max_epochs:  # max epochs reached: stop training
            print('Max Epochs reached')
            return False

        self.current_epoch += 1
        print(f"EPOCH: {self.current_epoch}")

        return self.is_improving(val_loss, val_accuracy)


def train_network_classification(net, train_loader, test_loader, optimizer, stopping_algorithm,
                                 criterion=F.cross_entropy):

    train_loss_history, training_acc_hist = list(), list()
    val_loss_history, val_acc_hist = list(), list()

    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(test_loader.dataset)

    best_test_accuracy = 0.0
    best_net_weights = net.state_dict()

    while stopping_algorithm.keep_training(val_loss_history, val_acc_hist):

        # training step
        training_loss, num_corrected_samples_train = \
            train_net_on_data(train_loader, optimizer, net, criterion)
        training_accuracy = num_corrected_samples_train / num_train_samples * 100.0

        train_loss_history.append(training_loss)
        training_acc_hist.append(training_accuracy)

        # evaluation step
        eval_loss, num_corrected_samples_test = eval_net_on_data(test_loader, net, criterion)
        # eval_loss = eval_loss.to('cpu').numpy()
        testing_accuracy = num_corrected_samples_test / num_val_samples * 100.0
        testing_accuracy = testing_accuracy.to('cpu').numpy()

        val_loss_history.append(eval_loss)
        val_acc_hist.append(testing_accuracy)

        print(f'Training Accuracy: {training_accuracy:.4f}; Validation Accuracy: {testing_accuracy:.4f}')

        if testing_accuracy > best_test_accuracy:
            best_test_accuracy = testing_accuracy

            # Create a deep copy of the model's parameters, otherwise dict is stored by reference
            best_net_weights = copy.deepcopy(net.state_dict())

    #     # load the best net on test data
    net.load_state_dict(best_net_weights)

    return net, (train_loss_history, training_acc_hist), (val_loss_history, val_acc_hist)


def define_objective_fcn_with_params(network_input_dim, train_data, test_data):
    def objective_function(trial):
        power_of_2_batch_size = trial.suggest_int('exp_batch_size', 3, 10)  # Suggest an exponent between 5 and 10
        BATCH_SIZE = 2 ** power_of_2_batch_size

        power_of_2_hidden_node_layer1 = trial.suggest_int('exp_nodes_1', 4, 9)
        number_of_hidden_nodes_1st_layer = 2 ** power_of_2_hidden_node_layer1

        power_of_2_hidden_node_layer2 = trial.suggest_int('exp_nodes_2', 4, 9)
        number_of_hidden_nodes_2nd_layer = 2 ** power_of_2_hidden_node_layer2

        HIDDEN_NODES = (number_of_hidden_nodes_1st_layer, number_of_hidden_nodes_2nd_layer)
        LEARNING_RATE = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        # Create the Data Loaders with given batch_size
        train_data_loaded, test_data_loaded = get_train_and_test_data_w_batch_size(BATCH_SIZE, train_data, test_data)

        # Create the model with given number of hidden architecture
        model = MNIST_MLP(network_input_dim, HIDDEN_NODES)

        # Create optimizer with given learning rate
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        improver = PerformanceImprover().is_improving

        stopping_algo = TrainingStopper(improver)

        model, (_, _), (val_loss, val_acc) = \
            train_network_classification(model, train_data_loaded, test_data_loaded, optimizer, stopping_algo)

        best_accuracy = np.max(val_acc)

        if best_accuracy > trial.user_attrs.get("best_accuracy", -1.0):
            trial.set_user_attr("best_accuracy", best_accuracy)
            trial.set_user_attr("best_state_dict", model.state_dict())

        return best_accuracy

    return objective_function


def get_HW_acceleration_if_available():
    try:
        if torch.backends.mps.is_available():
            return 'mps'
    except AttributeError:
        pass

    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'
