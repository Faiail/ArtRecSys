from src.utils.LinkSplit import LinkSplit
from src.neo4j2raw.load_dataset.artgraph import ArtGraph
import torch
from src.utils.FineTuner import FineTuner
from torch_geometric.transforms import ToUndirected


def get_binary_graph(graph, relation, item):
    graph[relation][item] = (graph[relation][item] > 0).type(torch.LongTensor)
    return graph


def main():
    artgraph2recsys = ArtGraph(root='./src/neo4j2raw/artgraph2recsys',
                               preprocess='constant')[0]


    artgraph2recsys = get_binary_graph(artgraph2recsys, ('user', 'rates', 'artwork'), 'edge_weight')

    splitter = LinkSplit(edge_type=('user', 'rates', 'artwork'))
    train_data, val_data, test_data = splitter(artgraph2recsys)
    train_data, val_data, test_data = ToUndirected()(train_data), ToUndirected()(val_data), ToUndirected()(test_data)

    fine_tuner = FineTuner(artgraph2recsys, train_data, val_data, out_dir='test_ft/', device='cpu')
    best = fine_tuner.tune(max_evals=1)
    print(best)


if __name__ == '__main__':
    main()