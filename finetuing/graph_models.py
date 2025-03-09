'''
Author: your name
Date: 2021-09-06 10:39:43
LastEditTime: 2021-10-24 15:58:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/code/graph_models.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np

import inspect
from src.utils import Voc
from torch_geometric.utils import softmax, add_self_loops
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros, uniform

from src.build_tree import build_stage_one_edges, build_stage_two_edges, build_cominbed_edges
from src.build_tree import build_icd9_tree, build_atc_tree, build_px_tree

from torch_scatter import scatter

class OntologyEmbedding(nn.Module):
    def __init__(self, voc, build_tree_func,
                 in_channels=100, out_channels=20, heads=5):
        super(OntologyEmbedding, self).__init__()

        # 禁用原来的树边构建
        self.graph_voc = Voc()
        for word in voc.idx2word.values():
            self.graph_voc.add_sentence([word])

        # 移除GNN图构建
        self.idx_mapping = [self.graph_voc.word2idx[word]
                           for word in voc.idx2word.values()]

        # 直接使用简单的嵌入层替代GNN
        num_nodes = len(self.graph_voc.word2idx)
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))
        self.init_params()

    def get_all_graph_emb(self):
        # 直接返回嵌入
        return self.embedding

    def forward(self):
        """
        :param idxs: [N, L]
        :return:
        """
        # 直接返回相应的嵌入，绕过GNN传播
        return self.embedding[self.idx_mapping]

    def init_params(self):
        # 使用Xavier初始化
        nn.init.xavier_normal_(self.embedding)
# class OntologyEmbedding(nn.Module):
#     def __init__(self, voc, build_tree_func,
#                  in_channels=100, out_channels=20, heads=5):
#         super(OntologyEmbedding, self).__init__()
#
#         # initial tree edges
#         res, graph_voc = build_tree_func(list(voc.idx2word.values()))
#         stage_one_edges = build_stage_one_edges(res, graph_voc)
#         stage_two_edges = build_stage_two_edges(res, graph_voc)
#
#         self.edges1 = torch.tensor(stage_one_edges)
#         self.edges2 = torch.tensor(stage_two_edges)
#         self.graph_voc = graph_voc
#
#         # construct model
#         assert in_channels == heads * out_channels
#         self.g = GATConv(in_channels=in_channels,
#                          out_channels=out_channels,
#                          heads=heads)
#
#         # tree embedding
#         num_nodes = len(graph_voc.word2idx)
#         self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))
#
#         # idx mapping: FROM leaf node in graphvoc TO voc
#         self.idx_mapping = [self.graph_voc.word2idx[word]
#                             for word in voc.idx2word.values()]
#
#         self.init_params()
#
#     def get_all_graph_emb(self):
#         emb = self.embedding
#         emb = self.g(self.g(emb, self.edges1.to(emb.device)),
#                      self.edges2.to(emb.device))
#         return emb
#
#     def forward(self):
#         """
#         :param idxs: [N, L]
#         :return:
#         """
#         emb = self.embedding
#
#         # 打印一下边的形状，帮助调试
#         print("Edges1 shape:", self.edges1.shape)
#         print("Edge index:", self.edges1)
#
#         # 检查边是否为空
#         if self.edges1.numel() == 0:
#             print("Warning: edges1 is empty!")
#             # 可以添加一些基本的边以避免错误
#             self.edges1 = torch.tensor([[0, 0], [0, 0]]).t()
#
#         # 同样检查edges2
#         if self.edges2.numel() == 0:
#             print("Warning: edges2 is empty!")
#             self.edges2 = torch.tensor([[0, 0], [0, 0]]).t()
#
#         # 应用图卷积
#         try:
#             emb = self.g(self.g(emb, self.edges1.to(emb.device)),
#                          self.edges2.to(emb.device))
#         except Exception as e:
#             print(f"Error in graph computation: {e}")
#             # 如果出错，可以返回原始嵌入向量
#             return emb[self.idx_mapping]
#
#         return emb[self.idx_mapping]
#
#     def init_params(self):
#         glorot(self.embedding)


class MessagePassing(nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getfullargspec(self.message).args[1:]
        self.update_args = inspect.getfullargspec(self.update).args[2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        # out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = scatter(out, edge_index[1], reduce=aggr, dim=0)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate('add', edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)

        alpha = F.dropout(alpha, p=self.dropout)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class ConcatEmbeddings_4(nn.Module):
    """Concat rx and dx ontology embedding for easy access
    """

    def __init__(self, config, dx_voc, px_voc, item_voc, rx_voc):
        super(ConcatEmbeddings_4, self).__init__()
        # special token: "[PAD]", "[CLS]", "[MASK]"
        self.special_embedding = nn.Parameter(
            torch.Tensor(
                config.code_vocab_size - len(dx_voc.idx2word) - len(rx_voc.idx2word) - len(px_voc.idx2word) - len(
                    item_voc.idx2word), \
                config.code_hidden_size))
        self.rx_embedding = OntologyEmbedding(rx_voc, build_atc_tree,
                                              config.code_hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.dx_embedding = OntologyEmbedding(dx_voc, build_icd9_tree,
                                              config.code_hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.px_embedding = OntologyEmbedding(px_voc, build_px_tree,
                                              config.code_hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.item_embedding = nn.Parameter(
            torch.Tensor(len(item_voc.idx2word), config.code_hidden_size)
        )
        self.init_params()

    def forward(self, input_ids):
        emb = torch.cat(
            [self.special_embedding, self.dx_embedding(), self.px_embedding(), self.item_embedding,
             self.rx_embedding()], dim=0)
        return emb[input_ids]

    def init_params(self):
        glorot(self.special_embedding)


class ConcatEmbeddings(nn.Module):
    """Concat rx and dx ontology embedding for easy access
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(ConcatEmbeddings, self).__init__()
        # special token: "[PAD]", "[CLS]", "[MASK]"
        self.special_embedding = nn.Parameter(
            torch.Tensor(config.code_vocab_size - len(dx_voc.idx2word) - len(rx_voc.idx2word), config.code_hidden_size))
        self.rx_embedding = OntologyEmbedding(rx_voc, build_atc_tree,
                                              config.code_hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.dx_embedding = OntologyEmbedding(dx_voc, build_icd9_tree,
                                              config.code_hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.init_params()

    def forward(self, input_ids):
        emb = torch.cat(
            [self.special_embedding, self.rx_embedding(), self.dx_embedding()], dim=0)
        return emb[input_ids]

    def init_params(self):
        glorot(self.special_embedding)


class FuseEmbeddings(nn.Module):
    """Construct the embeddings from ontology, patient info and type embeddings.
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(FuseEmbeddings, self).__init__()
        self.ontology_embedding = ConcatEmbeddings(config, dx_voc, rx_voc)
        self.type_embedding = nn.Embedding(2, config.code_hidden_size)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids, input_types=None, input_positions=None):
        """
        :param input_ids: [B, L]
        :param input_types: [B, L]
        :param input_positions:
        :return:
        """
        # return self.ontology_embedding(input_ids)
        ontology_embedding = self.ontology_embedding(
            input_ids) + self.type_embedding(input_types)
        return ontology_embedding