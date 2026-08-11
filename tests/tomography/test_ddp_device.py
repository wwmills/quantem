"""Device resolution in ``DDPMixin.setup_distributed``.

The single-process branch used to read

    device = torch.device("cuda:0" if device is None else device)
    torch.cuda.set_device(device.index)

which meant, on any machine with a visible GPU:

* ``device="cuda"`` -- the default on ``Tomography.from_models``,
  ``TomographyBase`` and ``TomographyLite`` -- raised ``ValueError``, because a
  device with no index has ``index is None``;
* ``device="cpu"`` was ignored and then raised the same way, so a CPU run was
  impossible on a GPU node;
* ``device=None`` pinned ``cuda:0`` regardless of ``CUDA_VISIBLE_DEVICES``.

None of that was reachable from the CPU test suite, which is why it survived.
"""

import pytest
import torch

from quantem.core.ml.ddp import DDPMixin


class _Host(DDPMixin):
    """Bare DDPMixin user -- setup_distributed touches nothing else."""


@pytest.mark.parametrize("requested", [None, "cuda", "cpu", torch.device("cpu")])
def test_every_device_form_resolves(requested):
    """No accepted spelling of a device may raise."""
    host = _Host()
    host.setup_distributed(device=requested)
    assert isinstance(host.device, torch.device)
    # A resolved CUDA device always carries an index, which is what
    # torch.cuda.set_device and every downstream .to() need.
    if host.device.type == "cuda":
        assert host.device.index is not None


def test_explicit_cpu_is_honoured():
    """Asking for CPU must give CPU, GPU present or not.

    This is the one that made CPU smoke tests impossible on a GPU node.
    """
    host = _Host()
    host.setup_distributed(device="cpu")
    assert host.device == torch.device("cpu")


def test_bare_cuda_string_gets_an_index():
    """`device="cuda"` is the documented default and must work."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    host = _Host()
    host.setup_distributed(device="cuda")
    assert host.device.type == "cuda"
    assert host.device.index == torch.cuda.current_device()


def test_indexed_device_is_returned_unchanged():
    """Existing call sites pass an indexed device; they must be unaffected."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    host = _Host()
    host.setup_distributed(device="cuda:0")
    assert host.device == torch.device("cuda", 0)


def test_single_process_ranks_are_set():
    """The non-torchrun branch still reports a world size of one."""
    host = _Host()
    host.setup_distributed(device="cpu")
    assert (host.world_size, host.global_rank, host.local_rank) == (1, 0, 0)
