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
        self.checkpointer = Checkpointer(self.path)
        self.model_class = ExampleModel
        self.model_name = self.model_class.__name__

    def tearDown(self) -> None:
        # Clear checkpoints directory
        for filename in os.listdir(self.path):
            os.remove(os.path.join(self.path, filename))

    def _get_model(self):
        return self.model_class(10, 5)

    def test_checkpointer_saving(self) -> None:
        self.checkpointer.save_checkpoint(self.model_name, {}, {}, {})

    def test_checkpointer_loading(self) -> None:
        visible_metadata = {"example_visible_metadata_entry": "example_value"}
        hidden_metadata = {"example_hidden_metadata_entry": "example_entry_value"}
        data = {"example_data_entry": [1, 2, 3, 4, 5]}
        self.checkpointer.save_checkpoint(self.model_name, {}, {}, {})
        self.checkpointer.save_checkpoint(self.model_name, visible_metadata, hidden_metadata, data)
        self.checkpointer.save_checkpoint(self.model_name, {}, {}, {})
        checkpoints = self.checkpointer.search_checkpoints(
            self.model_name, visible_metadata, hidden_metadata
        )
        self.assertEqual(len(checkpoints), 1)
        loaded_visible_metadata, loaded_hidden_metadata = (
            self.checkpointer.get_checkpoint_metadata(checkpoints[0])
        )
        del loaded_visible_metadata[Checkpointer.TIME_METADATA]
        self.assertEqual(visible_metadata, loaded_visible_metadata)
        self.assertEqual(hidden_metadata, loaded_hidden_metadata)
        loaded_visible_metadata, loaded_hidden_metadata, loaded_data = (
            self.checkpointer.load_checkpoint(checkpoints[0])
        )
        del loaded_visible_metadata[Checkpointer.TIME_METADATA]
        self.assertEqual(visible_metadata, loaded_visible_metadata)
        self.assertEqual(hidden_metadata, loaded_hidden_metadata)
        self.assertEqual(data, loaded_data)

    def test_checkpointer_loading_parameters(self) -> None:
        model = self._get_model()
        torch.manual_seed(100)
        t = torch.rand(10)
        torch.manual_seed(
            int.from_bytes(random.randbytes(4), byteorder="little", signed=False)
        )
        output_1 = model(t)
        self.checkpointer.save_checkpoint(
            self.model_name,
            {"epoch": "10"},
            {"loss": "23"},
            {"state_dict": model.state_dict()},
        )
        model = self._get_model()
        output_2 = model(t)
        checkpoints = self.checkpointer.search_checkpoints(
            self.model_name,
            {"epoch": "10"},
            {},
            desc_sort_by=Checkpointer.TIME_METADATA,
        )
        self.assertGreater(len(checkpoints), 0)
        _, _, data = self.checkpointer.load_checkpoint(checkpoints[0])
        model.load_state_dict(data["state_dict"])
        output_3 = model(t)
        self.assertTrue(torch.equal(output_1, output_3))
        self.assertFalse(torch.equal(output_1, output_2))
