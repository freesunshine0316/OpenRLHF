import numbers
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

from openrlhf.utils.utils import convert_token_to_id
from .utils import zero_pad_sequences


class OutcomeRewardDataset(Dataset):
    """
    Dataset for outcome reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        max_n_samples=-1,
        enable_test_memory_mode: bool=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.enable_test_memory_mode = enable_test_memory_mode

        # chat_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)

        # Store the processed data in class attributes
        inputs = dataset[self.input_key]
        outputs = dataset[self.output_key]
        labels = dataset[self.label_key]
        q2ols = defaultdict(list)
        for q, o, l in zip(inputs, outputs, labels):
            q2ols[q].append((o, l))

        if max_n_samples != -1:
            for q in q2ols.keys():
                q2ols[q] = q2ols[q][:max_n_samples]

        self.input_ids = []
        self.output_ids = []
        self.labels = []
        self.action_masks = []
        for q in tqdm(q2ols, total=len(q2ols)):
            input_ids = self.tokenizer.encode(input_template.format(q), add_special_tokens=False)

            for oup, label in q2ols[q]:
                output_ids = self.tokenizer.encode(oup, add_special_tokens=False)
                if len(input_ids) + len(output_ids) > self.max_length:
                    continue
                self.input_ids.append(input_ids)
                self.output_ids.append(output_ids)
                self.labels.append(label)
                self.action_masks.append([0] * len(input_ids) + [1] * len(output_ids))

        self.max_len = max([len(x) for x in self.input_ids])

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        seq_ids = torch.tensor(self.input_ids[idx] + self.output_ids[idx])
        label = self.labels[idx]
        action_mask = torch.tensor(self.action_masks[idx])
        
        mask = torch.ones_like(seq_ids)
        label = torch.tensor(label, dtype=torch.float).view((-1,))

        if self.enable_test_memory_mode:
            seq_ids = torch.randint(4, 100, (self.max_len,))
            mask = torch.ones_like(seq_ids)
            action_mask = torch.ones_like(seq_ids)

        return (
            seq_ids,
            mask,
            label,
            action_mask,
        )

    def collate_fn(self, item_list):
        input_ids = []
        input_masks = []
        labels = []
        action_masks = []
        for input_id, input_mask, label, action_mask in item_list:
            input_ids.append(input_id)
            input_masks.append(input_mask)
            labels.append(label)
            action_masks.append(action_mask)

        padding_side = "right"
        input_ids = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks, side=padding_side)
        labels = torch.concat(labels)
        action_masks = zero_pad_sequences(action_masks, side=padding_side)
        return input_ids, input_masks, labels, action_masks

    def packing_collate_fn(self, item_list):
        raise NotImplementedError
