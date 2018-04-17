def calculate_blended_output(time_slots, in_volumes, in_qualities, positions, bed_width):
    out_volumes = [0] * bed_width
    out_quality_sums = [0] * bed_width

    for i in range(time_slots):
        volume = in_volumes[i]
        quality = in_qualities[i]
        position = positions[i]

        out_volumes[position] += volume
        out_quality_sums[position] += quality * volume

    out_qualities = [0] * bed_width
    for position in range(bed_width):
        if out_volumes[position] > 0:
            out_qualities[position] = out_quality_sums[position] / out_volumes[position]

    return out_volumes, out_qualities
