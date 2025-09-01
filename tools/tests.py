import torch
from nap_robustness_from_onnx import update_bounds_with_nap

def test_active_neuron_already_positive():
    lbs = [torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[0.5, -0.5]]])]
    ubs = [torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[0.5, 1.0]]])]
    nap = [[1, 0]]  # First neuron active, second inactive

    update_bounds_with_nap(ubs, lbs, nap, label=8)

    assert torch.isclose(lbs[1][0][0], torch.tensor(0.5))
    assert torch.isclose(ubs[1][0][1], torch.tensor(0.0))


def test_active_neuron_negative():
    lbs = [torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[-0.3, -0.5]]])]
    ubs = [torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[0.5, 1.0]]])]
    nap = [[1, 0]]

    update_bounds_with_nap(ubs, lbs, nap, label=8)

    assert torch.isclose(lbs[1][0][0], torch.tensor(0.0)), "Active neuron should be pushed to >= 0"
    assert torch.isclose(ubs[1][0][1], torch.tensor(0.0)), "Inactive neuron should be clamped to <= 0"


def test_inactive_neuron_already_negative():
    lbs = [torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[0.3, -0.5]]])]
    ubs = [torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[0.5, -0.1]]])]
    nap = [[1, 0]]

    update_bounds_with_nap(ubs, lbs, nap, label=8)

    assert torch.isclose(lbs[1][0][0], torch.tensor(0.3))
    assert torch.isclose(ubs[1][0][1], torch.tensor(-0.1))


def test_inactive_neuron_positive():
    lbs = [torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[0.3, -0.5]]])]
    ubs = [torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[0.5, 0.2]]])]
    nap = [[1, 0]]

    update_bounds_with_nap(ubs, lbs, nap, label=8)

    assert torch.isclose(lbs[1][0][0], torch.tensor(0.3))
    assert torch.isclose(ubs[1][0][1], torch.tensor(0.0)), "Inactive neuron upper bound should be clamped to 0"


def test_multiple_neurons_mixed():
    lbs = [torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[-0.2, 0.1, -0.5]]])]
    ubs = [torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[0.5, 0.7, 1.0]]])]
    nap = [[1, 0, 1]]  # Active - Inactive - Active

    update_bounds_with_nap(ubs, lbs, nap, label=8)

    assert torch.isclose(lbs[1][0][0], torch.tensor(0.0)), "Active neuron should be pushed to >= 0"
    assert torch.isclose(ubs[1][0][1], torch.tensor(0.0)), "Inactive neuron should be taken to <= 0"
    assert torch.isclose(lbs[1][0][2], torch.tensor(0.0)), "Active neuron should be pushed to >= 0"
