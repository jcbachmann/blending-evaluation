import logging
import os
import sys

from PyQt5.QtWidgets import QApplication

from data_explorer.gui import MainWindow


def execute(
		path: str,
		entry_list: list,
		testlet_list: list,
		main_figures=None,
		label: str = 'Unknown',
		icon: str = os.path.join(os.path.dirname(__file__), 'icon.svg'),
		verbose: bool = False
):
	if verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	app = QApplication([])
	win = MainWindow(path, entry_list, testlet_list, label, main_figures=main_figures, icon=icon)
	win.show()

	# Execute and exit with app result code
	sys.exit(app.exec_())
