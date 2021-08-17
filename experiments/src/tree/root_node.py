import os
import time

from experiments.src.tree.base_node import BaseNode
from experiments.src.tree.node_type import NodeType


class RootNode(BaseNode):
    type = NodeType.ROOT

    def __init__(self):
        super(RootNode, self).__init__()
        self.name = "Root"

    def resolve_impl(self):
        return

