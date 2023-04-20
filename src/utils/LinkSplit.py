import torch
from typing import Union, Optional
from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
import pandas as pd
from torch_geometric.seed import seed_everything
from copy import deepcopy


class LinkSplit:
    def __init__(
        self,
        num_val: float = 0.1,
        num_test: Union[int, float] = 0.2,
        edge_type: Optional[EdgeType] = None,
        source_label_key: str = 'edge_weight',
        seed=42,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.edge_type = edge_type
        self.source_label_key = source_label_key
        self.seed = seed

        assert isinstance(self.num_val, float) and 0 < num_val < 1, "num_val argument must be within [0,1]"
        assert isinstance(self.num_test, float) and 0 < num_val < 1, "num_test argument must be within [0,1]"
        assert (self.num_val + self.num_test) < 1,\
            "num_val + num_test must be lower than 1. In this case no training set is available!"

        seed_everything(self.seed)

    def __call__(self, data: HeteroData):
        # compute val and test size
        val_size = int(self.num_val * data[self.edge_type].edge_index.shape[1])
        test_size = int(self.num_test * data[self.edge_type].edge_index.shape[1])

        # create base train, val and test set
        train, val, test = deepcopy(data), deepcopy(data), deepcopy(data)

        # create source dataframe
        df = pd.DataFrame(data[self.edge_type].edge_index.cpu().numpy().T, columns=['user','artwork'])
        df['label'] = data[self.edge_type][self.source_label_key].cpu().numpy()

        # create edge_label and edge_label_index for validation set
        val_df = df.sample(n=val_size, replace=False, random_state=self.seed)
        val[self.edge_type]['edge_label_index'] = torch.from_numpy(val_df[['user', 'artwork']].values.T).type(
            torch.LongTensor)
        val[self.edge_type]['edge_label'] = torch.from_numpy(val_df['label'].values).type(torch.LongTensor)
        df.drop(val_df.index, inplace=True)

        # create edge_label and edge_label_index for test set
        test_df = df.sample(n=test_size, replace=False, random_state=self.seed)
        test[self.edge_type]['edge_label_index'] = torch.from_numpy(test_df[['user', 'artwork']].values.T).type(
            torch.LongTensor)
        test[self.edge_type]['edge_label'] = torch.from_numpy(test_df['label'].values).type(torch.LongTensor)
        df.drop(test_df.index, inplace=True)

        # create edge_label and edge_label_index for training set
        train[self.edge_type]['edge_label_index'] = torch.from_numpy(df[['user', 'artwork']].values.T).type(
            torch.LongTensor)
        train[self.edge_type]['edge_label'] = torch.from_numpy(df['label'].values).type(torch.LongTensor)

        # overwrite knowledge in training set
        train[self.edge_type]['edge_index'] = train[self.edge_type]['edge_label_index']
        train[self.edge_type]['edge_weight'] = train[self.edge_type]['edge_label']

        # overwrite knowledge in validation set
        val[self.edge_type]['edge_index'] = train[self.edge_type]['edge_index']
        val[self.edge_type]['edge_weight'] = train[self.edge_type]['edge_weight']

        # overwrite knowledge in test set
        test[self.edge_type]['edge_index'] = train[self.edge_type]['edge_index']
        test[self.edge_type]['edge_weight'] = train[self.edge_type]['edge_weight']

        return train, val, test
