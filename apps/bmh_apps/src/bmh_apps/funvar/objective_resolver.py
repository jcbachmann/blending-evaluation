import re


def prettify_objective(objective: str) -> str:
    matchers = [
        (r"F1(?:/(.*))?", "F1 Homogenization Efficiency Ratio"),
        (r"F2", "F2 Standard Deviation from Ideal Stockpile Reclaim Volume"),
        (r"F3", "F3 Total Travel Distance"),
        (r"F4", "F4 Maximum Travel Speed"),
    ]

    for matcher in matchers:
        r = re.compile(matcher[0])
        match = re.match(r, objective)
        if match:
            return matcher[1] + (f" for {match.group(1)}" if r.groups > 0 and match.group(1) else "")

    return objective
