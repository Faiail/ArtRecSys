from models.graph.encoder import GNNEncoder
from neo4j2raw.load_dataset.artgraph import ArtGraph
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero

if __name__ == '__main__':
    artgraph2recsys = ArtGraph(root='./neo4j2raw/artgraph2recsys',
                               preprocess='constant')[0]
    #print(artgraph2recsys)
    encoder=GNNEncoder()
    encoder = to_hetero(encoder, T.ToUndirected()(artgraph2recsys).metadata(), aggr='sum')
    #print(encoder)
    out = encoder(artgraph2recsys.x_dict, artgraph2recsys.edge_index_dict)
    print(out)