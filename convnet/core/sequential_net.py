"""
Abstract Class of a Neural Network formed as a sequence of computation layer.
E.g. A CNN can be formalized as layers of computation.
"""

from .list import DoublyLinkedList, ListNode


class Layer(ListNode):
    """An base class for all type of layers in a neural network"""

    def __init__(self, typename: str, name='', prev=None, next=None):
        super().__init__(prev, next)
        self.type = typename
        self.name = name
        self._call_counter = 0

    def __call__(self, input_, train=False):
        """
        Call this layer will create an operation,
        which take use the layer to process the input data and return the results.
        A Layer must first be compiled to be able to be called.
        :param input: input tensor of this layer
        :param train: indicating the layer is called by training mode or not
        :param name: the name (used for ops) of this layer
        """
        assert self.is_compiled
        self._call_counter += 1
        return input_

    def compile(self):
        """
        Do all the things that should be done before running
        """
        return

    @property
    def is_compiled(self):
        """

        :return: True if the layer is compiled, False if the layer is not yet compiled.
        """
        return True

    @property
    def output_shape(self):
        """

        :return: the shape of the output
        """
        raise NotImplementedError("The base Layer class has no output_shape implementation!")

    @property
    def n_calls(self):
        return self._call_counter


class SequentialNet(DoublyLinkedList):
    """A base class for sequential Nets"""
    def __init__(self, name: str):
        super(SequentialNet, self).__init__()
        self.name = name
        self.is_compiled = False

    def push_back(self, layer: Layer):
        self.push_back_node(layer)

    def pop_back(self) -> Layer:
        self.pop_back_node()

    def compile(self):
        cur = self.front
        while cur != self.end:
            cur.compile()
            cur = cur.next
        self.is_compiled = True

    def layers(self):
        cur = self.front
        while cur != self.end:
            yield cur
            cur = cur.next

    def __call__(self, data, train):
        cur = self.front
        results = data
        while cur != self.end:
            results = cur(results, train)
            cur = cur.next
        return results
