import logging
import os
import sys
import urllib.parse
import uuid

import plotly.graph_objects as go
import plotly.offline
import pyperclip

from bmh_apps.funvar.fun_var_results import FunVarResults


def export_fig(graph_type: str, fig: go.Figure, results: FunVarResults, label: str):
    output_directory = os.path.commonpath(results.df["file_path"].to_list())
    output_label = f"plot fun {graph_type} {label} {str(uuid.uuid4())[:4]}"
    html_filename = f"{output_label}.html"
    html_filepath = os.path.join(output_directory, html_filename)
    txt_filepath = os.path.join(output_directory, f"{output_label}.txt")
    logging.info(f"Writing plot to {html_filepath}")
    fig.write_html(html_filepath)
    fig.show()

    txt_content = "##### Graph\n"
    txt_content += plotly.offline.plot(fig, include_plotlyjs=False, output_type="div")
    txt_content += "\n\n"
    txt_content += f"Command: `{os.path.basename(sys.argv[0])} {' '.join(sys.argv[1:])}`"
    txt_content += "\n\n"
    txt_content += f"Local file: [{html_filename}](file://{urllib.parse.quote(html_filepath)})"

    with open(f"{txt_filepath}", "w") as f:
        f.write(txt_content)

    try:
        pyperclip.copy(txt_content)
        logging.info("Graph copied to clipboard")
    except pyperclip.PyperclipException as e:
        logging.warning(f"Could not copy graph to clipboard: {e}")
