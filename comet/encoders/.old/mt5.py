# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
mT5 Encoder
==============
    Pretrained mT5 encoder from Hugging Face.
"""
from typing import Dict, Optional

import torch
from comet.encoders.metric5 import MeTric5
from transformers import T5TokenizerFast, T5Config

from comet.encoders.base import Encoder

class MeTric5Encoder(Encoder):
    """mT5 encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
    """

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(
            pretrained_model, use_fast=True
        )
        if load_pretrained_weights:
            self.model = MeTric5.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = MeTric5(
                T5Config.from_pretrained(pretrained_model), add_pooling_layer=False
            )
        self.model.encoder.output_hidden_states = True

    @property
    def output_units(self) -> int:
        """Max number of tokens the encoder handles."""
        return self.model.config.hidden_size

    @property
    def max_positions(self) -> int:
        """Max number of tokens the encoder handles."""
        # where is this used??
        return 1024

    @property
    def num_layers(self) -> int:
        """Number of model layers available."""
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self) -> int:
        """Size of the seperator.
        E.g: For mT5 is just 1 ([SEP]) but models such as XLM-R use 2 (</s></s>).

        Returns:
            int: Number of tokens used between two segments.
        """
        return 1

    @property
    def uses_token_type_ids(self) -> bool:
        """Whether or not the model uses token type ids to differentiate sentences.

        Returns:
            bool: True for models that use token_type_ids such as mT5.
        """
        return False

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ):
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face

        Returns:
            Encoder: XLMREncoder object.
        """
        #mT5Encoder = cls(pretrained_model, load_pretrained_weights)
        return MeTric5Encoder(pretrained_model, load_pretrained_weights)

    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer."""
        for param in self.model.shared.parameters():
            param.requires_grad = False

    def layerwise_lr(self, lr: float, decay: float):
        """Calculates the learning rate for each layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            list: List of model parameters for all layers and the corresponding lr.
        """
        # Embedding Layer
        opt_parameters = [
            {
                "params": self.model.shared.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        # All layers
        opt_parameters += [
            {
                "params": self.model.encoder.block[i].parameters(),
                "lr": lr * decay**i,
            }
            for i in range(self.num_layers - 2, 0, -1)
        ]
        return opt_parameters

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """mT5 model forward

        Args:
            input_ids (torch.Tensor): tokenized batch.
            attention_mask (torch.Tensor): batch attention mask.
            token_type_ids (Optional[torch.tensor]): batch attention mask. Defaults to
                None

        Returns:
            Dict[str, torch.Tensor]: dictionary with 'sentemb', 'wordemb', 'all_layers'
                and 'attention_mask'.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {
            "sentemb": out.logits[:, 0, :],
            "wordemb": out.logits,
            "all_layers": out.encoder_hidden_states,
            "attention_mask": attention_mask,
        }
