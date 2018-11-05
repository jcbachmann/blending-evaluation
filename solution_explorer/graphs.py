import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from ciglobal.mpl_interaction import PanAndZoom
from data_explorer.gui import ExpandingFigureCanvas


class ObjectivesScatterCanvas(ExpandingFigureCanvas):
    def __init__(self, objectives: DataFrame, all_variables: DataFrame, all_objectives: DataFrame,
                 selection_callback):
        self.all_variables = all_variables
        self.all_objectives = all_objectives
        self.selection_callback = selection_callback

        # Setup interactive figure
        fig, self.ax = plt.subplots()
        super().__init__(fig)
        self.figure.pan_zoom = PanAndZoom(self.figure)

        # Setup plot
        # noinspection PyUnresolvedReferences
        self.colormap = plt.cm.winter
        self.norm = plt.Normalize(0, len(all_objectives.index) - 1)
        self.sc = self.ax.scatter(
            x=all_objectives[all_objectives.columns[0]],
            y=all_objectives[all_objectives.columns[1]],
            c=all_objectives.index,
            norm=self.norm,
            marker='.',
            cmap=self.colormap,
            s=3,
            label='All Solutions',
            picker=5
        )
        self.ax.scatter(
            x=objectives[objectives.columns[0]],
            y=objectives[objectives.columns[1]],
            marker='o',
            facecolors='none',
            edgecolors='black',
            linewidths=0.5,
            s=15,
            label='Optimization Result'
        )
        self.ax.set_xlabel(objectives.columns[0])
        self.ax.set_ylabel(objectives.columns[1])
        self.ax.set_xlim(0, None)
        self.ax.set_ylim(0, None)

        # Setup annotations
        self.hover_annotation = self.ax.annotate(
            'hover',
            xy=(0, 0),
            xytext=(20, 20),
            textcoords='offset points',
            color='black',
            bbox=dict(boxstyle='round4', fc='white', ec='black', lw=0.5, alpha=0.9),
            arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.3')
        )
        self.hover_annotation.set_visible(False)

        self.permanent_annotation = self.ax.annotate(
            'permanent',
            xy=(0, 0),
            xytext=(-30, -30),
            textcoords='offset points',
            color='black',
            bbox=dict(boxstyle='round4', fc='white', ec='black', lw=0.5, alpha=0.9),
            arrowprops=dict(arrowstyle='simple')
        )
        self.permanent_annotation.set_visible(False)

        # Connect events
        self.mpl_connect('motion_notify_event', self.hover_all_objectives)
        self.mpl_connect('button_press_event', self.handle_click)
        self.mpl_connect('pick_event', self.pick_from_all_objectives)

    def __str__(self):
        return 'Objectives Scatter'

    def hover_all_objectives(self, event):
        vis = self.hover_annotation.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                index = ind['ind'][0]
                self.hover_annotation.xy = self.sc.get_offsets()[index]
                self.hover_annotation.set_text(str(index))
                self.hover_annotation.get_bbox_patch().set_edgecolor(self.colormap(self.norm(index)))
                self.hover_annotation.set_visible(True)
                self.draw_idle()
                if not self.permanent_annotation.get_visible() and self.selection_callback:
                    self.selection_callback(index, self.all_variables.iloc[index], self.all_objectives.iloc[index])
            else:
                if vis:
                    self.hover_annotation.set_visible(False)
                    self.draw_idle()

    def pick_from_all_objectives(self, event):
        if event.mouseevent.dblclick:
            index = event.ind[0]
            logging.info(f'Picked item with index {index}')
            self.permanent_annotation.xy = self.sc.get_offsets()[index]
            self.permanent_annotation.set_text(str(index))
            self.permanent_annotation.get_bbox_patch().set_edgecolor(self.colormap(self.norm(index)))
            self.permanent_annotation.set_visible(True)
            self.draw_idle()
            if self.selection_callback:
                self.selection_callback(index, self.all_variables.iloc[index], self.all_objectives.iloc[index])

    def handle_click(self, event):
        if self.permanent_annotation.get_visible() and event.dblclick:
            if event.inaxes == self.ax:
                cont, ind = self.sc.contains(event)
                if not cont:
                    self.permanent_annotation.set_visible(False)
                    self.draw_idle()
            else:
                self.permanent_annotation.set_visible(False)
                self.draw_idle()


class PathDetailCanvas(ExpandingFigureCanvas):
    def __init__(self):
        # Setup interactive figure
        fig, self.ax = plt.subplots()
        super().__init__(fig)
        self.index = None

    def __str__(self):
        return f'Path Detail {self.index}'

    def plot_selection(self, index: int, variables: pd.Series, objectives: pd.Series):
        self.index = index
        self.ax.cla()
        self.figure.suptitle(f'Path {index}')
        self.ax.plot(np.linspace(0, 1, len(variables)), variables, label=f'Solution {index}')
        self.draw_idle()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
