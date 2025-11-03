import torch.nn as nn
import torch

try:
    import dgl
except ImportError:
    print("Can not import dgl module. Please install it")


class GraphConvlayer(nn.Module):
    def __init__(self, in_features: int = 512, out_features: int = 512):
        super().__init__()
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(out_features)
        #self.gnn_layer = dgl.nn.GraphConv(in_features, out_features)
        self.gnn_layer = dgl.nn.GraphConv(in_features, out_features)

    def forward(self, graph, features):
        with graph.local_scope():
            features = self.gnn_layer(graph, features)
            features = self.normalization(features)
            features = self.activation(features)
        return features


class GrphConvNetwork(nn.Module):
    def __init__(self, num_layers: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([GraphConvlayer() for _ in range(num_layers)])

    def forward(self, graph, features):
        with graph.local_scope():
            for layer in self.layers:
                features = layer(graph, features)
        return features


class GraphConvEncoder(nn.Module):
    def __init__(self, in_features: int = 10, pseudo_features: int = 5, hidden_features: int = 512,
                 projection_features: int = 512, num_layers: int = 3):
        super().__init__()
        self.pseudo_encoder = nn.Linear(in_features=pseudo_features, out_features=hidden_features, bias=True)

        self.node_encoder = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.types_embed = nn.Embedding(num_embeddings=3, embedding_dim=hidden_features)

        self.graph_network = GrphConvNetwork(num_layers=num_layers)
        self.global_pooling = dgl.nn.pytorch.glob.AvgPooling()
        self.out_projection = nn.Linear(in_features=hidden_features, out_features=projection_features, bias=True)
        self.in_projector = nn.Sequential(
            nn.LayerNorm(3 * hidden_features),
            nn.GELU(),
            nn.Linear(in_features = 3 * hidden_features, out_features=hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
        )



    def add_tensor_to_nodes(self, batched_graph, tensor):
        nodes_per_graph = batched_graph.batch_num_nodes()
        node_tensor = torch.repeat_interleave(tensor, nodes_per_graph, dim=0)
        return node_tensor

    def forward(self, graph, broad_features):
        with graph.local_scope():
            types = self.types_embed(graph.ndata["node_types"]).unsqueeze(-2)
            nodes_emb = self.node_encoder(graph.ndata["nodes_emb"])
            broad_emb = self.pseudo_encoder(broad_features)
            broad_emb = self.add_tensor_to_nodes(graph, broad_emb)


            types = types.expand(-1, nodes_emb.shape[-2], -1)
            nodes_emb = torch.cat((nodes_emb, broad_emb, types), dim=-1)
            nodes_emb = self.in_projector(nodes_emb)
            nodes_emb = self.graph_network(graph, nodes_emb)
            graph_vector = self.global_pooling(graph, nodes_emb)
        return self.out_projection(graph_vector)



class GrphFormerNetwork(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        num_heads = 2
        self.spatial_encoder = dgl.nn.SpatialEncoder(max_dist=5, num_heads=num_heads)
        self.layers = torch.nn.ModuleList([
        dgl.nn.GraphormerLayer(
        feat_size=256,  # the dimension of the input node features
        hidden_size=256,  # the dimension of the hidden layer
        num_heads=num_heads,  # the number of attention heads
        dropout=0.1,  # the dropout rate
        activation=torch.nn.GELU(),  # the activation function
        norm_first=False,  # whether to put the normalization before attention and feedforward
    )
    for _ in range(num_layers)
    ])

    def forward(self, graph, nodes_emb):
        nodes_per_graph = graph.batch_num_nodes()
        attn_mask = torch.zeros((sum(nodes_per_graph), sum(nodes_per_graph)), dtype=torch.bool, device=nodes_emb.device)
        start = 0
        end = 0
        if len(nodes_per_graph) == 1:
            attn_mask = ~attn_mask
        else:
            for node_batch in nodes_per_graph:
                end += node_batch
                attn_mask[start:end, start:end] = True
                start += node_batch

        dist = dgl.shortest_dist(graph)
        bias = self.spatial_encoder(dist)
        nodes_emb = nodes_emb.transpose(-2, -3)
        b_size = nodes_emb.shape[0]

        for layer in self.layers:
            nodes_emb = layer(
                nodes_emb,
                #attn_mask=attn_mask.unsqueeze(-3).expand(b_size, -1, -1),
                attn_bias=bias.unsqueeze(-4),
            )
        return nodes_emb.transpose(-2, -3)


class GraphFormerEncoder(nn.Module):
    def __init__(self, in_features: int = 10, pseudo_features: int = 5, hidden_features: int = 256,
                 projection_features: int = 256, num_layers: int = 3):
        super().__init__()
        self.pseudo_encoder = nn.Linear(in_features=pseudo_features, out_features=hidden_features, bias=True)

        self.node_encoder = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.types_embed = nn.Embedding(num_embeddings=3, embedding_dim=hidden_features)

        self.graphormer_network = GrphFormerNetwork(num_layers=num_layers)
        self.global_pooling = dgl.nn.pytorch.glob.AvgPooling()
        self.out_projection = nn.Linear(in_features=hidden_features, out_features=projection_features, bias=True)
        self.in_projector = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(in_features = hidden_features, out_features=hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
        )

    def add_tensor_to_nodes(self, batched_graph, tensor):
        nodes_per_graph = batched_graph.batch_num_nodes()
        node_tensor = torch.repeat_interleave(tensor, nodes_per_graph, dim=0)
        return node_tensor

    def forward(self, graph, broad_features):
        with graph.local_scope():
            types = self.types_embed(graph.ndata["node_types"]).unsqueeze(-2)
            nodes_emb = self.node_encoder(graph.ndata["nodes_emb"])
            broad_emb = self.pseudo_encoder(broad_features)
            broad_emb = self.add_tensor_to_nodes(graph, broad_emb)


            types = types.expand(-1, nodes_emb.shape[-2], -1)
            nodes_emb = nodes_emb + broad_emb + types
            #nodes_emb = torch.cat((nodes_emb, broad_emb, types), dim=-1)
            nodes_emb = self.in_projector(nodes_emb)

            nodes_emb = self.graphormer_network(graph, nodes_emb)
            graph_vector = self.global_pooling(graph, nodes_emb)
        return self.out_projection(graph_vector)