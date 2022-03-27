from typing import List


def calculate_blended_output(time_slots, in_volumes, in_qualities, positions, bed_width):
    out_volumes = [0] * bed_width
    parameter_count = len(in_qualities[0]) if len(in_qualities) > 0 else 1
    out_parameter_sums = [[0] * parameter_count for _ in range(bed_width)]

    for i in range(time_slots):
        volume = in_volumes[i]
        parameters = in_qualities[i]
        position = positions[i]

        out_volumes[position] += volume
        for j in range(parameter_count):
            out_parameter_sums[position][j] += parameters[j] * volume

    out_qualities: List[List[float]] = [[0] * parameter_count for _ in range(bed_width)]
    for position in range(bed_width):
        if out_volumes[position] > 0:
            for j in range(parameter_count):
                out_qualities[position][j] = out_parameter_sums[position][j] / out_volumes[position]

    return out_volumes, out_qualities
