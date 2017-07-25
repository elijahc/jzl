import plotly
import plotly.graph_objs as go
from .layouts import PowerSpectrumLayout

class Spectrogram(go.Figure):
    def __init__(self,title,z,y,colorscale='Jet',**kwargs):
        self.data = [dict(
                type='heatmapgl',
                y=y,
                z=z,
                colorscale=colorscale,
                colorbar=dict(
                    title='Power (dB)'
                    ),
                )]
        self.layout = PowerSpectrumLayout(title=title).build()

        super(Spectrogram,self).__init__(**kwargs)
