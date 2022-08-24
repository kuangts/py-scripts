import os, sys, csv
import numpy as np
from image import Image, seg2surf
from landmark import Landmark
from mesh import TriangleSurface
import plotly.graph_objects as go

accept_click = False

if __name__ == '__main__':

    case = sys.argv[1]
    reader = csv.reader(open(fr'\\p920-jxia-f7\Share\FL_lmk\temp\{case}.csv'))
    labels, *_, err = zip(*reader)
    lmk_sel = [l for i,l in enumerate(labels) if float(err[i])>5]

    # read this current case
    seg = Image.read(os.path.join(r'\\p920-jxia-f7\Share\FL', case, 'seg.nii.gz'))
    s1 = seg2surf(Image(seg==1))
    s2 = seg2surf(Image(seg==2))
    lmk = Landmark.read(os.path.join(r'\\p920-jxia-f7\Share\FL', case,'lmk.csv'))
    # read problematic landmarks
    lmk_sel = lmk.select(lmk_sel)

    fig = go.Figure()
    trc1 = s1.plot(fig, color='blue', opacity= 1, lighting=dict(ambient=0.2, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2))
    trc2 = s2.plot(fig, color='orange', opacity= 1, lighting=dict(ambient=0.2, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2))
    lmk_sel.plot(fig, mode="markers+text", textposition="top center", textfont=dict(color='white', size=30), marker=dict(size=15))
    fig.update_traces(lightposition=dict(x=s1.V[:,0].mean(), y= s1.V[:,1].min()*2 - s1.V[:,1].mean() , z=s1.V[:,2].mean()), selector=dict(type='mesh3d'))
    fig.update_layout(clickmode='event+select', margin=dict(l=0, r=0, b=0, t=0))

    if accept_click:
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
        import json
        app = Dash()
        app.layout = html.Div([
            dcc.Graph(
                id='click',
                figure=fig
            ),
            html.Div(className='row', children=[
                html.Div([
                    dcc.Markdown("""
                        **Click Data**

                        Click on points in the graph.
                    """),
                    html.Pre(id='click-data'),
                ], className='three columns'),
            ])
        ])
        @app.callback(
            Output('click-data', 'children'),
            Input('click', 'clickData')
        )
        def display_click_data(clickData):
            print(clickData)
            return json.dumps(clickData, indent=2)

        app.run_server()

    fig.show()
