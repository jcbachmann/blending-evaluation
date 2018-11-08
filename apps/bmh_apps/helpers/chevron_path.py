from pandas import DataFrame


def chevron_path(layers: float) -> DataFrame:
    path = [[float(f), float(f % 2)] for f in range(0, int(layers) + 1)]
    if layers - int(layers) > 0:
        # incomplete layer
        path.append([layers, (layers - int(layers)) if int(layers) % 2 == 0 else (1 - (layers - int(layers)))])
    return DataFrame(data=path, columns=['part', 'path'])
