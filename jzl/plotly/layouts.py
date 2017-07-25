import plotly
import plotly.graph_objs as go

class DefaultLayout(object):
    def __init__(self,
            title='Default Title',
            xlabel='X Axis',
            ylabel='Y Axis'):
        self.title = title
        self.xlabel=xlabel
        self.ylabel=ylabel
        self._xaxis = None
        self._yaxis = None

    @property
    def xaxis(self):
        self._xaxis=dict(
                    title = self.xlabel,
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f',
                        ),
                    )
        return self._xaxis

    @property
    def yaxis(self):
        self._yaxis=dict(
                    title = self.ylabel,
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f',
                        ),
                    )
        return self._yaxis

    def build(self):
        return go.Layout(
                title=self.title,
                xaxis=self.xaxis,
                yaxis=self.yaxis
                )

class PowerSpectrumLayout(DefaultLayout):
    def __init__(self,
            xlabel='Time (s)',
            ylabel='Frequency (Hz)',**kwargs):
        self.xlabel=xlabel
        self.ylabel=ylabel
        super(PowerSpectrumLayout,self).__init__(xlabel=xlabel,ylabel=ylabel,**kwargs)
