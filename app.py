# Morris-Lecar model

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np
import pandas as pd


def ml1(gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=None,
        V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
        T=100, dt=0.05, n0=0):
    """
    Single-cell Morris-Lecar dynamics

    Parameters
    ----------
    gL : float,  optional
        leak conductance
    VL : float,  optional
        reversal potential of leak current
    gCa : float,  optional
        Ca2+ conductance
    VCa : float,  optional
        reversal potential of Ca2+ current
    gK : float,  optional
        K+ conductance
    VK : float,  optional
        reversal potential of K+ current
    C : float,  optional
        membrane capacitance
    I_ext : float, list, tuple, nd.array, optional
        external current, if list, tuple or nd.array, use first value
    V1 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V2 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V3 : float, optional
        shape parameter for K+ channel steady-state open probability function
    V4 : float, optional
        shape parameter for K+ channel steady-state open probability function
    phi : float, optional
        parameter in the K+ current ODE
    sd : float, optional
        standard deviation of the noise source
    v0 : float, optional
        initial value of the voltage variable V
    w0 : float, optional
        initial value of the K+ current variable w
    T : float, optional
        total simulation time
    dt : float, optional
        sampling time interval
    v_mn : float, optional
        plotting limit: minimum voltage
    v_mx : float, optional
        plotting limit: maximum voltage
    w_mn : float, optional
        plotting limit: minimum K+ channel open fraction
    w_mx : float, optional
        plotting limit: maximum K+ channel open fraction
    doplot : boolean, optional
        construct the plot or not

    Returns
    -------
    X : 1D array of float
        voltage values
    Y : 1D array of float
        K+ channel open fraction

    Example
    -------
    >>> ...

    """
    nt = int(T/dt)
    #C1 = 1/C
    sd_sqrt_dt = sd*np.sqrt(dt)
    try:
        # in case I_ext is provided
        I = np.hstack( (I_ext[0]*np.ones(n0), I_ext) )
    except:
        # I_ext not provided, set to zero
        I = np.zeros(n0+nt)

    # initial conditions
    v = v0
    w = w0
    X = np.zeros(nt)
    X[0] = v
    Y = np.zeros(nt)
    Y[0] = w

    # steady-state functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(n0+nt):
        #if (t%100 == 0): print(f"t={t:d}/{n0+nt:d}\r", end="")
        # Morris-Lecar equations
        dvdt = 1/C * (-gL*(v-VL) -gCa*m_inf(v)*(v-VCa) -gK*w*(v-VK) + I[t])
        dwdt = lambda_w(v) * (w_inf(v) - w)
        # integrate
        v += (dvdt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w += (dwdt*dt)
        if (t >= n0):
            X[t-n0] = v
            Y[t-n0] = w
    #print("")

    #if doplot:
    #    time = np.arange(nt)*dt
    #    fig, ax = plt.subplots(1,1,figsize=(16,4))
    #    ax.plot(time, X, '-k', lw=2)
    #    ax.set_ylim(-100,60)
    #    ax.set_xlabel("time [ms]", fontsize=fs_)
    #    ax.set_ylabel("voltage [mV]", fontsize=fs_)
    #    #ax.set_title(f"Single cell Morris-Lecar", fontsize=fs_)
    #    ax.grid(True)
    #    plt.tight_layout()
    #    plt.show()

    return X, Y


# run initial model
dt = 0.05
I_min, I_max = 0, 200
T = 1000
I_ext = I_min*np.ones(int(T/dt))
params = {
    'T': T,
    'dt': dt,
    'sd': 0.05,
    'I_ext': I_ext,
    'v0': 30,
    'w0': 0.05,
}
X, Y = ml1(**params)
df = pd.DataFrame({
    'time' : dt*np.arange(len(X)),
    'voltage' : X,
})

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        I_min,
        I_max,
        step=None,
        value=I_min,
        id='I-slider'
    )
])
#marks={str(year): str(year) for year in df['year'].unique()},


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('I-slider', 'value'))
def update_figure(I_curr):
    I_ext = I_curr*np.ones(int(T/dt))
    params = {
        'T': T,
        'dt': dt,
        'sd': 0.05,
        'I_ext': I_ext,
        'v0': 30,
        'w0': 0.05,
    }
    X, Y = ml1(**params)
    df = pd.DataFrame({
        'time' : dt*np.arange(len(X)),
        'voltage' : X,
    })

    fig = px.scatter(df, x="time", y="voltage")
    fig.update_layout(transition_duration=500)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
