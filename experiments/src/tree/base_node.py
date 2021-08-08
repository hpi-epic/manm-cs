import copy
import threading
from abc import ABC
from anytree import NodeMixin

from experiments.src.tree.node_type import NodeType


class BaseNode(ABC, NodeMixin):
    type: NodeType
    resolved = False
    resolved_data = None

    def __init__(self):
        self.lock = threading.Lock()

    def resolve(self):
        if self.parent:
            self.parent.resolve()

        self.lock.acquire()
        if not self.resolved:
            self.resolved_data = self.resolve_impl()
            self.resolved = True
        self.lock.release()

        return

    def resolve_impl(self):
        raise NotImplementedError()

    def create_children(self):
        raise NotImplementedError()

    def get_parent_with_type(self, type: NodeType):
        current_parent = self.parent

        while self.parent:
            if current_parent.type == type:
                return current_parent

            current_parent = current_parent.parent

        raise Exception("Did not found parent with type. Do you have the correct order of parent types?")
