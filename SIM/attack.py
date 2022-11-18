import torch
import torch.nn.functional as F
from .utils import *
from .ImplicitRobustModel import *


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_test(model, device, test_loader, epsilon, attack_num=100, do_print=True):
    # Accuracy counter
    m = (lambda x: model(x, 0.0)[0]) if isinstance(model, (ImplicitRobustModel, ImplicitRobustModelRank1FT)) else model
    correct = 0
    adv_examples = []

    # Loop over all examples in validation set
    for data, target in test_loader:
        #for data ,target in zip(d, t):
        #    if len(losses) >= attack_num:
        #        break

        # Send the data and label to the device
        #data, target = data.to(device)[None, ...], target.to(device)[None, ...]
        data, target = data.to(device)[:attack_num], target.to(device)[:attack_num]

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = m(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        ## If the initial prediction is wrong, dont bother attacking, just move on
        #if init_pred.item() != target.item():
        #    continue

        # Calculate the loss
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = m(perturbed_data)

        # Re-calculate the loss
        loss = F.cross_entropy(output, target)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        for i in range(attack_num):
            if final_pred[i].item() == target[i].item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred[i].item(), final_pred[i].item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred[i].item(), final_pred[i].item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / attack_num
    #loss = torch.mean(torch.tensor(losses))
    if do_print:
        print("FGSM: Epsilon: {}, Test Accuracy = {} / {} = {}, Test Loss = {}".format(epsilon, correct, attack_num, final_acc, loss))

    # Return the accuracy and an adversarial example
    model.zero_grad()
    return loss, final_acc, perturbed_data


def fgsm_purturb(model, data, target, device, epsilon):
    m = (lambda x: model(x, 0.0)[0]) if isinstance(model, ImplicitRobustModel) else model

    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = m(data)
    init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

    ## If the initial prediction is wrong, dont bother attacking, just move on
    # if init_pred.item() != target.item():
    #    continue

    # Calculate the loss
    loss = F.cross_entropy(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)

    # Zero all existing gradients
    model.zero_grad()

    return perturbed_data