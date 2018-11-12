import sys

from pandas import DataFrame


class StrToBytesWrapper:
    def __init__(self, bytes_buffer):
        self.bytes_buffer = bytes_buffer

    def write(self, s):
        self.bytes_buffer.write(s.encode('utf-8'))


class StackerPrinter:
    def __init__(self, header=True, out_buffer=sys.stdout.buffer):
        self.header = header
        self.out_buffer = out_buffer

    @staticmethod
    def status(msg):
        print(f'[stacker] {msg}', file=sys.stderr)

    def out(self, material: DataFrame):
        first_cols = ['timestamp', 'x', 'z', 'volume']
        col_order = first_cols.copy()
        col_order.extend(list(set(material.columns) - set(first_cols)))
        material.to_csv(StrToBytesWrapper(self.out_buffer), index=False, columns=col_order, header=self.header, sep=' ')

    def flush(self):
        self.out_buffer.flush()
