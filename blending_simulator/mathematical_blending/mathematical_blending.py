def calculate_blended_output(time_slots, in_volumes, in_qualities, positions, bed_width):
    out_volumes = [0] * bed_width
    out_parameter_sums = [0] * bed_width

    for i in range(time_slots):
        volume = in_volumes[i]
        parameter = in_qualities[i]
        position = positions[i]

        out_volumes[position] += volume
        out_parameter_sums[position] += parameter * volume

    out_qualities = [0] * bed_width
    for position in range(bed_width):
        if out_volumes[position] > 0:
            out_qualities[position] = out_parameter_sums[position] / out_volumes[position]

    return out_volumes, out_qualities
