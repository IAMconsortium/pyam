import pandas as pd

from pyam.utils import _meta_idx, merge_meta


def subtract(a, b):
    return a - b


class BinaryOp(object):

    def __init__(self, a, b, ignore_meta_conflict=False):
        self.a_df = a
        self.b_df = b
        
        # Generate meta data merge if two dataframes are not the same,
        # and do so early to fail early
        # TODO: merge_meta needs tests
        self.meta = merge_meta(a.meta, b.meta, ignore_meta_conflict)

    def op_data(self, axis):
        # TODO:
        # - this currently requires only single entries in both dataframes
        #   in the computation axis.
        # - adding support for more will require a (ndarray) shape check *and*
        #   intelligent name support (e.g., share of {var_numerator} in
        #   {var_denominator}).
        # - intelligent shape checking for N vs. 1 operations is easy:
        #   - if len(a_df[axis]) % len(b_df[axis]) != 0 then raise (% is modulo
        #     operator)
        # - multi-dimensional would require looking up how, e.g., pandas or
        #   numpy does it
        too_many_vals_error = "{} operand contains more than one `{}`"
        if len(self.a_df[axis].unique()) > 1:
            raise ValueError(too_many_vals_error.format("First", axis))

        if len(self.b_df[axis].unique()) > 1:
            raise ValueError(too_many_vals_error.format("Second", axis))

        idx = list(set(self.a_df.data.columns) - set([axis, 'value']))

        return (
            self.a_df.data.set_index(idx).drop(axis, axis='columns'),
            self.b_df.data.set_index(idx).drop(axis, axis='columns')
        )

    def calc_meta(self, res):
        # final meta wrangling
        keep_meta_idx = self.meta.index.intersection(_meta_idx(res))
        return self.meta.loc[keep_meta_idx]

    def calc(self, func, axis, axis_value):
        a, b = self.op_data(axis)
        data = func(a, b).reset_index()
        data[axis] = axis_value
        meta = self.calc_meta(data)
        return data, meta
