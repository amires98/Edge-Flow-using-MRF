import numpy as np
from PGM_Node import Node


def smoothness(flow1, flow2):
    return np.linalg.norm(flow1 - flow2)


class ProbableTree:

    def __init__(self, component_img, id, stats, img1, img2, num_states, patch_size):
        self.img1 = img1
        self.height, self.width = img1.shape
        self.img2 = img2
        self.visited_pixels = set()
        self.root = None
        self.num_states = num_states
        self.pairwise = self.compute_pairwise_energy()
        self.ps = patch_size
        self.alpha = 0.6
        self.map = set()
        self.number_of_stationary=0
        self.stationary_rate=0.45
        left, top, h, w, area = stats
        self.area=area
        i = 0
        while not component_img[top + i, left] == id:
            i += 1
        stack_node = [Node([top + i, left], num_states)]

        sms_stack = []  # importnatno
        while stack_node:

            top_node = stack_node.pop()
            self.visited_pixels.add(tuple(top_node.get_pixel()))
            if not self.root:
                self.root = top_node
                self.root.set_unary(- self.init_unary_message(self.root.pixel, self.ps))

            child = 0
            # orx means original pixels
            for i in [-1, +1]:
                for j in [0, 1, 2, 3]:
                    pixel = top_node.get_pixel().copy()
                    if j == 2:
                        pixel[0] += i
                        pixel[1] += i
                    elif j == 3:
                        pixel[0] += i
                        pixel[1] -= i
                    else:
                        pixel[j] += i
                    # If the second condition holds there is a loop but we don't connect it so we still remain a tree
                    if self.if_in_component(pixel) and component_img[pixel[0], pixel[1]] == id and tuple(pixel) not in self.visited_pixels:
                        kid_node = Node(pixel, self.num_states)
                        top_node.add_kid(kid_node)
                        kid_node.set_parent(top_node)
                        stack_node.append(kid_node)
                        sms_stack.append(kid_node)
                        child += 1

            if not child:
                if top_node.is_equal(self.root):
                    print("hell neww and print area: ", area)
                sms_stack.remove(top_node)
                # patch size is important you want to check on that !!!!
                if top_node is not self.root:
                    temp_unary = - self.init_unary_message(top_node.pixel, self.ps)
                    top_node.set_unary(temp_unary)
                    msg = (temp_unary + self.alpha * self.pairwise).min(1)
                    msg -= np.mean(msg)
                    top_node.get_parent().add_first_msg(msg)

                    # !!!! !!!!! here is a problem, whether to put - NCC and minimize or this choice
                    # or maybe -log(pairwise)

                    while sms_stack and (not stack_node or not stack_node[-1].is_equal(sms_stack[-1])):
                        sms_node = sms_stack.pop()
                        temp_un = - self.init_unary_message(sms_node.pixel, self.ps)
                        sms_node.set_unary(temp_un)
                        msg = ((temp_un + sms_node.in_message_first) + self.alpha * self.pairwise).min(1)
                        msg -= np.mean(msg)
                        sms_node.get_parent().add_first_msg(msg)

    def init_unary_message(self, point, patch_size):
        potential = np.zeros((self.num_states ** 2))
        l = self.num_states // 2
        flows = np.array([i for i in range(-l, l + 1)])
        counter = 0
        for i in range(self.num_states ** 2):
            flow = np.array([flows[i // self.num_states], flows[i % self.num_states]])
            potential[counter] = self.flow_error(point, flow, patch_size)
            counter += 1
        return potential

    def flow_error(self, point, flow, patch_size):
        l = patch_size // 2

        h_low = point[0] - l
        h_high = point[0] + l + 1
        w_low = point[1] - l
        w_high = point[1] + l + 1

        h1 = max(0, h_low)
        h2 = min(self.height, h_high)
        w1 = max(0, w_low)
        w2 = min(self.width, w_high)

        # print('!!!!!!!!! new set !!!!!!!')
        # print('init center: ', point)
        # print('flow: ', flow)
        # print('patch1: ', h1, h2, w1, w2)

        dh1 = np.abs(h1 - h_low)
        dh2 = np.abs(h2 - h_high)
        dw1 = np.abs(w1 - w_low)
        dw2 = np.abs(w2 - w_high)

        h_low += flow[0]
        h_high += flow[0]
        w_low += flow[1]
        w_high += flow[1]

        if w_high <= 0 or h_high <= 0 or w_low > self.width or h_low > self.height:
            return 0

        ph1 = max(0, h_low)
        ph2 = min(self.height, h_high)
        pw1 = max(0, w_low)
        pw2 = min(self.width, w_high)

        # print('patch2', ph1, ph2, pw1, pw2)

        pdh1 = np.abs(ph1 - h_low)
        pdh2 = np.abs(ph2 - h_high)
        pdw1 = np.abs(pw1 - w_low)
        pdw2 = np.abs(pw2 - w_high)

        if pdh1 > dh1:
            diff = pdh1 - dh1
            h1 += diff
        elif dh1 > pdh1:
            diff = dh1 - pdh1
            ph1 += diff
        if pdh2 > dh2:
            diff = pdh2 - dh2
            h2 -= diff
        elif dh2 > pdh2:
            diff = dh2 - pdh2
            ph2 -= diff
        if pdw1 > dw1:
            diff = pdw1 - dw1
            w1 += diff
        elif dw1 > pdw1:
            diff = dw1 - pdw1
            pw1 += diff
        if pdw2 > dw2:
            diff = pdw2 - dw2
            w2 -= diff
        elif dw2 > pdw2:
            diff = dw2 - pdw2
            pw2 -= diff

        patch1 = self.img1[h1:h2, w1:w2]
        patch2 = self.img2[ph1: ph2, pw1: pw2]

        product = np.mean((patch1 - patch1.mean((0, 1))) * (patch2 - patch2.mean((0, 1))))
        # stds = patch1.std() * patch2.std()
        #
        # if stds == 0:
        #     return 0
        # else:
        #     product /= stds
        return product

    def compute_pairwise_energy(self):
        l = self.num_states // 2
        energy = np.zeros((self.num_states ** 2, self.num_states ** 2))
        flows = np.array([i for i in range(-l, l + 1)])
        for i in range(self.num_states ** 2):
            for j in range(self.num_states ** 2):
                # ATTENTION if you change the smoothness fucntion you might wanna omit energy
                if i == j:
                    energy[i, j] = 0
                else:
                    f1 = np.array([flows[i // self.num_states], flows[i % self.num_states]])
                    f2 = np.array([flows[j // self.num_states], flows[j % self.num_states]])
                    energy[i, j] = smoothness(f1, f2)
        return energy

    def backtrack(self):
        stack = []
        stack.append(self.root)

        while stack:
            temp_node = stack.pop()
            if temp_node is self.root:
                out_map = np.argmin(temp_node.in_message_first + temp_node.unary)
            else:
                # check the dimensions and addition and min
                in_map_msg = temp_node.in_map
                out_map = np.argmin(
                    self.alpha * self.pairwise[in_map_msg, :] + temp_node.unary + temp_node.in_message_first)
            temp_node.out_map = out_map
            if self.is_stationary(self.map_flow(out_map)):
                self.number_of_stationary+=1
            self.map.add(tuple(temp_node.pixel) + tuple(self.map_flow(out_map)))
            print(temp_node.pixel, ':', self.map_flow(out_map))

            for n in temp_node.kids:
                n.in_map = out_map
                stack.append(n)

    def map_flow(self, index):
        hl = self.num_states // 2
        flows = [i for i in range(-hl, hl + 1)]
        return np.array([flows[index // self.num_states], flows[index % self.num_states]])

    def get_map(self):
        return self.map

    def is_stationary(self, flow):
        if flow[0] ==0 and flow[1] ==  0:
            return True
        else:
            return False
    def staionary_report(self):
        if self.stationary_rate <= self.number_of_stationary/self.area:
            return True
        else:
            False

    def if_in_component(self, location):
        if location[0]<self.height and location[0]> -1 and location[1]<self.width and location[1]> -1:
            return True
        else:
            return False