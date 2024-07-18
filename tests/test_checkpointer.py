import os
import random
import unittest

import torch
import torch.nn as nn

from ssepoptim.base.checkpointing import Checkpointer


class ExampleModel(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(in_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class TestCheckpointer(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "checkpoints/"
        self.checkpointer = Checkpointer({"path": self.path})
        self.model_class = ExampleModel
        self.model_name = self.model_class.__name__

    def tearDown(self) -> None:
        # Clear checkpoints directory
        for filename in os.listdir(self.path):
            os.remove(os.path.join(self.path, filename))

    def _get_model(self):
        return self.model_class(10, 5)

    def test_checkpointer_saving(self) -> None:
        self.checkpointer.save(self.model_name, {}, {})

    def test_checkpointer_loading_metadata(self) -> None:
        major_metadata = {"example_major_metadata_entry": "example_entry_value"}
        minor_metadata = {"example_minor_metadata_entry": [1, 2, 3, 4, 5]}
        self.checkpointer.save(self.model_name, {}, {})
        self.checkpointer.save(self.model_name, major_metadata, minor_metadata)
        self.checkpointer.save(self.model_name, {}, {})
        loaded_minor_metadata = self.checkpointer.search_minor_metadata(
            self.model_name, major_metadata
        )
        self.assertEqual(minor_metadata, loaded_minor_metadata)

    def test_checkpointer_loading_metadata_2(self) -> None:
        model = self._get_model()
        torch.manual_seed(100)
        t = torch.rand(10)
        torch.manual_seed(
            int.from_bytes(random.randbytes(4), byteorder="little", signed=False)
        )
        output_1 = model(t)
        self.checkpointer.save(
            self.model_name, {"epoch": "10"}, {"state_dict": model.state_dict()}
        )
        model = self._get_model()
        output_2 = model(t)
        minor_metadata = self.checkpointer.search_minor_metadata(
            self.model_name, {"epoch": "10"}, pick_latest=True
        )
        model.load_state_dict(minor_metadata["state_dict"])
        output_3 = model(t)
        self.assertTrue(torch.equal(output_1, output_3))
        self.assertFalse(torch.equal(output_1, output_2))
