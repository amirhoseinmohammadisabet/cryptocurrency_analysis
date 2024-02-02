import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Create multiple Plotly plots
plots = []

for i in range(5):
    x = [j for j in range(i+1)]
    y = [j**2 for j in range(i+1)]

    plot = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', name=f'Plot {i+1}'))
    plot.update_layout(title=f'Plot {i+1}', xaxis_title='X-axis', yaxis_title='Y-axis')

    plots.append(plot)

# Create a subplot with multiple plots
subplot = make_subplots(rows=2, cols=3, subplot_titles=[f'Plot {i+1}' for i in range(5)])

# Add traces to subplot
for i in range(2):
    for j in range(3):
        if plots:
            subplot.add_trace(plots.pop(0)['data'][0], row=i+1, col=j+1)

# Save the subplot to an HTML file
pyo.plot(subplot, filename='interactive_plots_plotly.html')
