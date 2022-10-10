import numpy as np


class Node:

    def __init__(self, pixel, num_states):
        self.kids = []
        self.parent = None
        self.pixel = pixel
        self.in_message_first = np.zeros(num_states ** 2)
        self.out_message_first = None
        self.in_map = None  # I assume there is no need with stack or sth
        self.out_map = None  # I assume there is no need with stack or sth
        self.unary = None
        self.argmax_flow = None
        self.first_msg_counter = 0

    def set_parent(self, parent):
        self.parent = parent

    def add_kid(self, kid):
        self.kids.append(kid)

    def get_parent(self):
        return self.parent

    def get_kids(self):
        return self.kids

    def get_pixel(self):
        return self.pixel

    def set_unary(self, potential):
        self.unary = potential

    def is_equal(self, other_node):
        if self.pixel[0] == other_node.pixel[0] and self.pixel[1] == other_node.pixel[1]:
            return True
        else:
            return False

    def add_first_msg(self, msg):
        self.in_message_first += msg
        self.first_msg_counter += 1
