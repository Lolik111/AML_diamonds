import numpy as np


class DTree:
    """
    Simple regression tree classifier for a training data with both categorical and numerical
    features
    """
    _model = None

    def __init__(self, mask, max_depth=np.inf, min_leaves=3, stdr_threshold=0.001):
        """

        :param mask: mask that shows which attributes it numerical
        :param max_depth: maximum tree depth
        :param min_leaves: minimum samples in a node
        :param stdr_threshold: minimum std reduction for splitting
        """
        self.mask = mask
        self.max_depth = max_depth
        self.min_leaves = min_leaves
        self.stdr_threshold = stdr_threshold

    def fit(self, X, Y):
        self._model = self.create_branches({'attr_id': -1,
                                            'branches': dict(),
                                            'decision': None,
                                            'info': {'d': 0}}, X, Y)



    def choose_best_attribute_regr(self, X, Y):
        """
        input:
            X: numpy array of size (n,m) containing training examples
            Y: numpy array of size (n,) containing target class

        returns: the index of the attribute that results in maximum standard deviation (std) reduction .
        Returns -1 (leaf node) in the next cases:
        1) Not enough samples (less than min_leaves)
        2) Targets are almost the same (std less than 1% of mean)
        3) Std reduction less than 5%
        """
        n = len(Y)
        before_std = np.std(Y)
        if n <= self.min_leaves or before_std < 0.01 * np.mean(Y):
            return -1, None

        res = []
        splits = []

        # different functions can be applied here
        count_score = lambda count, sum, sum_sq: \
            np.sqrt((sum_sq / count - (sum / count) ** 2)) if (sum_sq / count - (sum / count) ** 2) >= 0 else 0

        for i, column in enumerate(X.T):
            dev = 0.
            split = None
            if not self.mask[i]:
                concat_arr = np.c_[column, Y]
                for idx, counts in zip(*np.unique(column, return_counts=True)):
                    class_values = concat_arr[concat_arr[:, 0] == idx]
                    dev += np.std(class_values[:, -1], dtype=np.float64) * counts / n
            else:
                sort_idx = np.argsort(column)
                sort_col, sort_y = column[sort_idx], Y[sort_idx]
                # dev_arr = np.apply_along_axis(lambda arr, p: np.std(arr[0:p]) + np.std(arr[p:]), 0, sort_y)
                right_sum, right_sum_sq = sort_y.sum(), (sort_y ** 2).sum()
                left_sum, left_sum_sq = 0., 0.
                dev = count_score(n, right_sum, right_sum_sq)
                for j in range(0, n - 1):
                    if sort_col[j] == sort_col[-1]:  # condition where split can be invalid
                        break
                    y_i = sort_y[j]
                    left_sum += y_i
                    right_sum -= y_i
                    left_sum_sq += y_i ** 2
                    right_sum_sq -= y_i ** 2
                    curr_score = count_score(j + 1, left_sum, left_sum_sq) * (j + 1) + \
                                 count_score(n - j - 1, right_sum, right_sum_sq) * (n - j - 1)
                    curr_score /= n
                    if curr_score <= dev:
                        dev, split = curr_score, (sort_col[j] + sort_col[j + 1]) / 2

            res.append(dev)
            splits.append(split)
        q, p = np.argmin(res), splits[np.argmin(res)]
        if before_std - res[q] <= self.stdr_threshold * before_std:
            return -1, None
        return q, p

    def most_common_class(self, Y):
        """
        input: target values
        returns: Mean of the target values (since it's minimum value for squared error,
                for another loss metrics it can be different)
        """
        return np.mean(Y)

    def create_branches(self, node, X, Y):
        """
        create branches in a decision tree recursively
        input:
            node: current node represented by a dictionary of format
                    {'attr_id': -1,
                     'branches': dict(),
                     'decision': None},
                     'info': dict()
                  where attr_id: specifies the current attribute index for branching
                                -1 mean the node is leaf node
                        braches: is a dictionary of format {attr_val:node}
                        decision: contains either the best guess based on
                                most common class or an actual class label if the
                                current node is the leaf
                        info: is a dictionary of additional information using to research, such as:
                            'd' - depth of leaf
                            'Y' - all target values stopped at that leaf
            X: training examples
            Y: target class

        returns: input node with fields updated
        """
        # choose best attribute to branch
        attr_id, split = self.choose_best_attribute_regr(X, Y)
        node['attr_id'] = attr_id
        # record the most common class
        node['decision'] = self.most_common_class(Y)
        info = node['info']

        # stop when we got to max depth
        if info['d'] == self.max_depth:
            attr_id = node['attr_id'] = -1

        if attr_id != -1:
            # find the set of unique values for the current attribute
            if self.mask[attr_id]:
                sel = X[:, attr_id] <= split
                node['branches'][split] = self.create_branches(self.node_template(info['d'] + 1), X[sel, :], Y[sel])
                node['branches'][split + 1e-10] = self.create_branches(self.node_template(info['d'] + 1), X[~sel, :],
                                                                       Y[~sel])
            else:
                attr_vals = np.unique(X[:, attr_id])
                for a_val in attr_vals:
                    # compute the boolean array for slicing the data for the next
                    # branching iteration
                    sel = X[:, attr_id] == a_val
                    # perform slicing
                    X_branch = X[sel, :]
                    Y_branch = Y[sel]
                    # perform recursive call
                    node['branches'][a_val] = self.create_branches(self.node_template(info['d'] + 1), X_branch,
                                                                   Y_branch)
        # else:
        #     if 'Y' in info:
        #         info['Y'].append(Y)
        #     else:
        #         info['Y'] = Y.copy()
        return node

    def predict(self, X):
        if X.ndim == 1:
            return self.traverse(self._model, X)
        elif X.ndim == 2:
            return np.array([self.traverse(self._model, row) for row in X])
        else:
            print("Dimensions error")

    def traverse(self, model, sample):
        """
        recursively traverse decision tree
        input:
            model: trained decision tree
            sample: input sample to classify

        returns: class label
        """
        if model['attr_id'] == -1:
            decision = model['decision']
        else:
            attr_val = sample[model['attr_id']]
            if self.mask[model['attr_id']]:
                l, r = [*model['branches']]
                if l > r:
                    l, r = r, l
                if attr_val <= l:
                    decision = self.traverse(model['branches'][l], sample)
                else:
                    decision = self.traverse(model['branches'][r], sample)
            else:
                if attr_val not in model['branches']:
                    decision = model['decision']
                else:
                    decision = self.traverse(model['branches'][attr_val], sample)
        return decision

    def node_template(self, d=0):
        return {'attr_id': -1,
                'branches': dict(),
                'decision': None,
                'info': {'d': d}}