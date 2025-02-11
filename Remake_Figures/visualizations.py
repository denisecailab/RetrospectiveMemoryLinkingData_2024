import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from numpy.polynomial.polynomial import polyfit
from plotly.subplots import make_subplots
from scipy.stats.stats import pearsonr



## Bar Graphs
## ==========

def plotMeanData(
    agg_data,
    groupby,
    plot_var,
    datapoint_var='Mouse',
    colors=["steelblue", "darkred"],
    plot_mode='bar',
    mean_line_color='black',
    marker_pattern_shape='',
    plot_datapoints=False,
    plot_datalines=False,
    y_range=(0, 100),
    y_title=None,
    x_title=None,
    text_size=20,
    font_family='Arial',
    plot_title=None,
    opacity=0.8,
    tick_angle=45,
    plot_width=350,
    plot_height=500,
    save_path=None,
    plot_scale=5
):
    """
    Parameters
    ==========
    agg_data : pandas dataframe
        aggregated pandas data frame where each mouse occupies one row
    groupby : str
        either 'ExpGroup' or 'Context' for what you'd like the data split up by
    plot_var : str
        column name to plot
    datapoint_var : str
        column name of individual datapoint variable. Default is 'Mouse'
    colors : list
        list of colors you would like used in plotting
    plot_mode : str
        one of 'bar' or 'point' where means are represented as bars or as points
    plot_datapoints, plot_datalines : boolean
        whether or not to plot individual datapoints, or individual datalines. Defaults are False.
    y_range : tuple
        tuple or min and max values of plot
    y_title, x_title : str
        y-axis and x-axis title respectively. Default is None.
    text_size : int
        size of text to use in the plot. Default is 20.
    font_family : str
        font to use for all plot labels
    plot_title : str
        master title for the graph. Default is None.
    opacity : float
        opacity value for bars. Default is 0.8.
    tick_angle : int
        angle that text on x-axis is displayed at. Default is 45.
    plot_width, plot_height : int
        width and height of the entire plot. Defaults are 350 & 500, respectively.
    save_path : boolean
        an optional file path to save the plot. If save_path=None, plot will not be saved. Default is None.
    plot_scale : int
         how high of a resolution to save the plot as. Default is 5.
    """

    if plot_title == None:
        plot_title = groupby

    means = agg_data[[groupby, plot_var]].groupby(groupby).mean()[plot_var].sort_index()
    sems = agg_data[[groupby, plot_var]].groupby(groupby).sem()[plot_var].sort_index()
    names = means.index.values

    fig = go.Figure()
    if plot_mode == 'bar':
        fig.add_trace(
            go.Bar(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True),
                marker_color=colors,
                marker=dict(line=dict(width=1, color="black"), opacity=opacity),
                marker_pattern_shape=marker_pattern_shape
            )
        )
    elif (plot_mode == 'point') & (plot_datalines):
        fig.add_trace(
            go.Scatter(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True),
                mode='lines+markers',
                marker_color=colors,
                marker=dict(size=15, line=dict(width=1, color="black"), opacity=opacity),
                line=dict(color=mean_line_color, width=3)
            )
        )
    elif plot_mode == 'point':
        fig.add_trace(
            go.Scatter(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True),
                mode='markers',
                marker_color=colors,
                marker=dict(size=15, line=dict(width=1, color=mean_line_color), opacity=opacity),
            )
        )
    else:
        raise Exception("Invalid plot_mode. Must be one of 'bar' or 'point'.")
    if plot_datapoints:
        for sub in agg_data[datapoint_var].unique():
            sub_data = agg_data[agg_data[datapoint_var] == sub]
            fig.add_trace(
                go.Scatter(
                    x=sub_data[groupby].values,
                    y=sub_data[plot_var].values,
                    mode="markers",
                    marker=dict(color="black", symbol='circle-open', size=10),
                    name=str(sub),
                )
            )
    if plot_datalines:
        for line in agg_data[datapoint_var].unique():
            line_data = agg_data[agg_data[datapoint_var] == line]
            line_data = line_data.iloc[line_data[groupby].argsort(),:]
            fig.add_trace(
                go.Scatter(
                    x=line_data[groupby].values,
                    y=line_data[plot_var].values,
                    mode="lines",
                    line=dict(width=1, color='grey'),
                    name=str(line),
                )
            )
    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        xaxis_title=x_title,
        font=dict(size=text_size, family=font_family),
        title_text=plot_title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
    )
    if tick_angle is not None:
        fig.update_xaxes(tickangle=tick_angle)
    fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


def plotAcrossGroups(
    agg_data,
    groupby,
    separateby,
    plot_var,
    colors,
    plot_title,
    datapoint_var="Mouse",
    y_range=None,
    plot_mode='bar',
    mean_line_color='black',
    marker_pattern_shape='',
    plot_datapoints=False,
    plot_datalines=False,
    y_title=None,
    x_title=None,
    add_hline=True,
    hline_y=0,
    text_size=20,
    font_family='Arial',
    opacity=0.8,
    plot_width=600,
    plot_height=600,
    tick_angle=45,
    scale_y=True,
    h_spacing=0.1,
    save_path=None,
    plot_scale=5
):
    """
    Plot multiple bar plots (one for each unique type in 'separateby') with multiple bars on each plot (one bar for each unique type in 'groupby')
    """
    # Note that the separating variable must be of type pd.Categorical(ordered=True), such that its unique values can be sorted
    # This can be done by: agg_data[separateby] = pd.Categorical(agg_data[separateby], categories=['list','of','unique','values'], ordered=True)
    subplot_titles = agg_data[separateby].unique().sort_values()
    fig = make_subplots(
        cols=len(subplot_titles), subplot_titles=subplot_titles, horizontal_spacing=h_spacing, shared_yaxes=scale_y
    )
    for i, val in enumerate(agg_data[separateby].unique().sort_values()):
        sub_data = agg_data[agg_data[separateby] == val]
        means = sub_data[[groupby, plot_var]].groupby(groupby).mean()[plot_var].sort_index()
        sems = sub_data[[groupby, plot_var]].groupby(groupby).sem()[plot_var].sort_index()
        xlabels = means.index.values
        if plot_mode == 'bar':
            fig.add_trace(
                go.Bar(
                    x=xlabels,
                    y=means[xlabels].values,
                    error_y=dict(type="data", array=sems.values, visible=True),
                    marker_color=colors,
                    marker=dict(line=dict(width=1, color="black"), opacity=opacity),
                    marker_pattern_shape=marker_pattern_shape
                ),
                row=1,
                col=i + 1,
            )
        elif (plot_mode == 'point') & (plot_datalines):
            fig.add_trace(
                go.Scatter(
                    x=xlabels,
                    y=means[xlabels].values,
                    error_y=dict(type="data", array=sems.values, visible=True),
                    mode='lines+markers',
                    marker_color=colors,
                    marker=dict(size=15, line=dict(width=1, color="black"), opacity=opacity),
                    line=dict(color=mean_line_color, width=4)
                ),
                row=1,
                col=i + 1,
            )
        elif plot_mode == 'point':
            fig.add_trace(
                go.Scatter(
                    x=xlabels,
                    y=means[xlabels].values,
                    error_y=dict(type="data", array=sems.values, visible=True),
                    mode='markers',
                    marker_color=colors,
                    marker=dict(size=15, line=dict(width=1, color=mean_line_color), opacity=opacity)
                ),
                row=1,
                col=i + 1,
            )
        else:
            raise Exception("Invalid plot_mode. Must be one of 'bar' or 'point'.")
        if plot_datapoints:
            for point in sub_data[datapoint_var].unique():
                point_data = sub_data[sub_data[datapoint_var] == point]
                fig.add_trace(
                    go.Scattergl(
                        x=point_data[groupby].values,
                        y=point_data[plot_var].values,
                        mode="markers",
                        marker=dict(color="black", symbol='circle-open', size=10),
                        name=str(point),
                    ),
                    row=1,
                    col=i + 1,
                )
        if plot_datalines:
            for line in sub_data[datapoint_var].unique():
                line_data = sub_data[sub_data[datapoint_var] == line]
                line_data = line_data.iloc[line_data[groupby].argsort(),:]
                fig.add_trace(
                    go.Scatter(
                        x=line_data[groupby].values,
                        y=line_data[plot_var].values,
                        mode="lines",
                        line=dict(width=1.5, color='grey'),
                        name=str(line),
                    ),
                    row=1,
                    col=i + 1,
                )
    if add_hline:
        fig.add_hline(y=hline_y, row=1, col='all', line_width=1, opacity=1, line_color='black')
    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        xaxis_title=x_title,
        font=dict(size=text_size, family=font_family),
        title_text=plot_title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
    )
    if tick_angle is not None:
        fig.update_xaxes(tickangle=tick_angle)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


## Scatter Plots
## =============

def plotCorrelation(
    data_x,
    data_y,
    x_title,
    y_title,
    title="",
    colors="steelblue",
    plot_fit=True,
    same_xy_scale=False,
    x_range=None,
    y_range=None,
    text_size=20,
    font_family='Arial',
    marker_size=7,
    outline_width=1,
    line_width=2,
    point_opacity=0.8,
    plot_height=600,
    plot_width=600,
    save_path=None,
    plot_scale=5
):
    """
    Given two sorted vectors ('data_x' & 'data_y'), plot the correlation between them
    Parameters:
    ==========
    data_x, data_y : numpy 1d vector or list
        data values to be plotted along x-axis and y-axis respectively
    x_title, y_title : str
        title text for x-axis and y-axis respectively
    title : str
        title for overall plot. Note that the title will also include the r and p values for
        line of best fit following the text provided. Default is ''.
    colors : str or vector/list of colors
        color to plot the datapoints. If string, all datapoints will be that color. If a list/vector
        of colors, length must be equal to length of data. Then, each index of the colors list will
        match the index of data in data_x and data_y. Default is 'steelblue'.
    plot_fit : boolean
        whether or not to plot the line of best fit. If False, identity line will be plotted instead.
        Default is True.
    same_xy_scale : boolean
        whether to set the ranges of the x- and y-axes to the same ranges. Default is False.
    x_range, y_range : None or tuple
        the range to set the x-axis and y-axis, respectively. If both are of type tuple and same_xy_scale
        is set to False, the plot will be updated with the specified axis ranges. Defaults are None.
    marker_size : float
        size of each datapoint. Default is 7.
    outline_width : float
        width of the outlines of each datapoint. Default is 1.
    line_width : float
        width of the line of best fit. Only used if plot_fits is True. Default is 2.
    point_opacity : float
        how opaque each datapoint should be, from [0,1]. Default is 0.8.
    plot_height, plot_width : int
        the height and width, respectively, of the entier plot. Defaults are 600 and 600, respectively.
    """
    corr_data = pd.DataFrame({"X": data_x, "Y": data_y})
    r, p = pearsonr(corr_data.X, corr_data.Y)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=corr_data.X,
            y=corr_data.Y,
            mode="markers",
            marker_size=marker_size,
            marker=dict(
                line=dict(color="black", width=outline_width), color=colors, opacity=point_opacity
            ),
            text=[
                str(np.round(x, 4)) + ", " + str(np.round(y, 4))
                for x, y in zip(corr_data.X, corr_data.Y)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )

    if plot_fit:
        b, m = polyfit(corr_data.X, corr_data.Y, 1)
        line_x = np.linspace(corr_data.X.min(), corr_data.X.max(), 10)
        fig.add_trace(
            go.Scattergl(
                x=line_x,
                y=(m * line_x + b),
                mode="lines",
                line=dict(color="black", width=line_width),
                name="Line of<br>best fit",
            )
        )
    else:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.add_trace(
            go.Scattergl(
                x=np.linspace(min_val, max_val),
                y=np.linspace(min_val, max_val),
                mode="lines",
                line=dict(color="black", width=line_width),
                name="Identity<br>Line",
            )
        )

    if same_xy_scale:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.update_xaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=x_title,
        )
        fig.update_yaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=y_title,
        )
    else:
        fig.update_xaxes(title_text=x_title)
        fig.update_yaxes(title_text=y_title)
        if (type(x_range) is tuple) & (type(y_range) is tuple):
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)
    fig.update_layout(
        dragmode="pan",
        font=dict(size=text_size, family=font_family),
        title_text=title
        + " r = "
        + str(np.round(r, 4))
        + ", p = "
        + str(np.round(p, 6)),
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
    )
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


def plotCorrelation_multifit(
    data_x,
    data_y,
    groups,
    x_title,
    y_title,
    title="",
    colors=None,
    color_palette=px.colors.qualitative.Safe,
    textinfo=None,
    plot_fits=True,
    plot_identity=False,
    same_xy_scale=False,
    x_range=None,
    y_range=None,
    text_size=20,
    font_family='Arial',
    marker_size=7,
    outline_width=1,
    line_width=2,
    point_opacity=0.8,
    plot_height=600,
    plot_width=600,
    save_path=None,
    plot_scale=5
):
    """
    Given two sorted vectors ('data_x' & 'data_y') and the associated 'groups' vector, plot the correlation between them, coloring each group separately
    Parameters:
    ==========
    data_x, data_y : numpy 1d vector or list
        data values to be plotted along x-axis and y-axis respectively
    groups : numpy 1d vector or list
        the group identity that each datapoint belongs to. Must be of same length as data_x and data_y.
    x_title, y_title : str
        title text for x-axis and y-axis respectively
    title : str
        title for overall plot. Note that the title will also include the r and p values for
        line of best fit following the text provided. Default is ''.
    colors : None or vector/list of colors
        colors to plot the datapoints. If None, color_palette will be discretized based on the number of
        unique values in groups. Each datapoint will be assigned a color based on the group it belongs to.
        If a list/vector of colors, length must be equal to length of data and the colors should match the
        group that each datapoint belongs to. Default is None.
    color_palette : plotly express color palette
        a color palette which is only used if colors = None (see above argument, colors). Default is px.colors.qualitative.Safe.
    textinfo : None or 1d vector or list
        the text that should appear over each datapoint during hover. If not None, must be of same length as data. Default is None.
    plot_fits : boolean
        whether or not to plot the lines of best fit for each unique group type. If False, only datapoints will be plotted.
        Default is True.
    plot_identity : boolean
        whether or not to plot the identity line as a black dotted line. Default is False.
    same_xy_scale : boolean
        whether to set the ranges of the x- and y-axes to the same ranges. Default is False.
    x_range, y_range : None or tuple
        the range to set the x-axis and y-axis, respectively. If both are of type tuple and same_xy_scale
        is set to False, the plot will be updated with the specified axis ranges. Defaults are None.
    text_size : int
        size of text to use in the plot. Default is 20.
    font_family : str
        font to use for all plot labels
    marker_size : float
        size of each datapoint. Default is 7.
    outline_width : float
        width of the outlines of each datapoint. Default is 1.
    line_width : float
        width of the line of best fit. Only used if plot_fits is True. Default is 2.
    point_opacity : float
        how opaque each datapoint should be, from [0,1]. Default is 0.8.
    plot_height, plot_width : int
        the height and width, respectively, of the entier plot. Defaults are 600 and 600, respectively.
    """

    groups = np.array(groups)
    if colors is None:
        len(np.unique(groups))
        colors = np.repeat("x", len(data_x)).astype(object)
        for i, group in enumerate(np.unique(groups)):
            colors[groups == group] = color_palette[i]

    corr_data = pd.DataFrame(
        {"X": data_x, "Y": data_y, "Groups": groups, "Colors": colors}
    )
    r, p = pearsonr(corr_data.X, corr_data.Y)

    fig = go.Figure()
    if textinfo is None:
        textinfo = np.repeat("", corr_data.shape[0])
    fig.add_trace(
        go.Scattergl(
            x=corr_data.X,
            y=corr_data.Y,
            mode="markers",
            marker_size=marker_size,
            marker=dict(
                line=dict(color="black", width=outline_width), color=colors, opacity=point_opacity
            ),
            text=[
                text + ", (" + str(np.round(x, 4)) + ", " + str(np.round(y, 4)) + ")"
                for text, (x, y) in zip(textinfo, zip(corr_data.X, corr_data.Y))
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )

    if plot_fits:
        for group in np.unique(groups):
            sub_x_data = corr_data[groups == group].X
            sub_y_data = corr_data[groups == group].Y
            color = corr_data[groups == group].Colors.values[0]
            b, m = polyfit(sub_x_data, sub_y_data, 1)
            line_x = np.linspace(sub_x_data.min(), sub_x_data.max(), 10)
            fig.add_trace(
                go.Scattergl(
                    x=line_x,
                    y=(m * line_x + b),
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    name=group,
                )
            )

    if plot_identity:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.add_trace(
            go.Scattergl(
                x=np.linspace(min_val, max_val),
                y=np.linspace(min_val, max_val),
                mode="lines",
                line=dict(color="black", width=line_width, dash='dash'),
                name="Identity<br>Line",
            )
        )

    if same_xy_scale:
        min_val = min(corr_data.X.min(), corr_data.Y.min())
        max_val = max(corr_data.X.max(), corr_data.Y.max())
        fig.update_xaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=x_title,
        )
        fig.update_yaxes(
            range=(min_val - (np.abs(min_val) / 5), max_val + (max_val / 5)),
            title_text=y_title,
        )
    else:
        fig.update_xaxes(title_text=x_title)
        fig.update_yaxes(title_text=y_title)
        if (type(x_range) is tuple) & (type(y_range) is tuple):
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)
    fig.update_layout(
        dragmode="pan",
        font=dict(size=text_size, family=font_family),
        title_text=title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
    )
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)


## Line Graphs
## ===========

def plotAcrossTime(
    agg_data,
    plot_var,
    colors,
    title,
    groupby=None,
    separateby=None,
    time_var='Time',
    datapoint_var="Mouse",
    y_range=None,
    mean_marker_size=15,
    plot_datapoints=False,
    plot_datalines=False,
    y_title=None,
    x_title=None,
    add_hline=True,
    hline_y=0,
    text_size=20,
    font_family='Arial',
    opacity=0.8,
    plot_width=600,
    plot_height=500,
    tick_angle=45,
    scale_y=True,
    h_spacing=0.2,
    save_path=None,
    plot_scale=5
):
    """
    Plot multiple line graphs (one for each unique type in 'separateby') with multiple line graphs on each plot (one bar for each unique type in 'groupby'), across time ('time_var').
    """

    if groupby is None:
        groupby = 'group_blank'
        agg_data[groupby] = ''
        agg_data[groupby] = pd.Categorical(agg_data[groupby])
    if separateby is None:
        separateby = 'sep_blank'
        agg_data[separateby] = ''
        agg_data[separateby] = pd.Categorical(agg_data[separateby])

    # Note that the separating variable must be of type pd.Categorical(ordered=True), such that its unique values can be sorted
    # This can be done by: agg_data[separateby] = pd.Categorical(agg_data[separateby], categories=['list','of','unique','values'], ordered=True)
    subplot_titles = agg_data[separateby].unique().sort_values()
    fig = make_subplots(
        cols=len(subplot_titles), subplot_titles=subplot_titles, horizontal_spacing=h_spacing, shared_yaxes=scale_y
    )
    for i, sep_val in enumerate(agg_data[separateby].unique().sort_values()):
        separateby_data = agg_data[agg_data[separateby] == sep_val]
        for j, group_val in enumerate(separateby_data[groupby].unique().sort_values()):
            sub_data = separateby_data[separateby_data[groupby] == group_val]
            means = sub_data[[time_var, plot_var]].groupby(time_var).mean()[plot_var].sort_index()
            sems = sub_data[[time_var, plot_var]].groupby(time_var).sem()[plot_var].sort_index()
            xlabels = means.index.values

            fig.add_trace(
                go.Scattergl(
                    x=xlabels,
                    y=means[xlabels].values,
                    error_y=dict(type="data", array=sems.values, visible=True),
                    mode='lines+markers',
                    marker_color=colors[j],
                    marker=dict(size=mean_marker_size, line=dict(width=1, color='black'), opacity=opacity),
                    line=dict(color=colors[j], width=4)
                ),
                row=1,
                col=i + 1,
            )

            if plot_datapoints:
                for point in sub_data[datapoint_var].unique():
                    point_data = sub_data[sub_data[datapoint_var] == point]
                    fig.add_trace(
                        go.Scattergl(
                            x=point_data[time_var].values,
                            y=point_data[plot_var].values,
                            mode="markers",
                            marker=dict(color="black", symbol='circle-open', size=10),
                            name=str(point),
                        ),
                        row=1,
                        col=i + 1,
                    )
            if plot_datalines:
                for line in sub_data[datapoint_var].unique():
                    line_data = sub_data[sub_data[datapoint_var] == line]
                    line_data = line_data.iloc[line_data[time_var].argsort(),:]
                    fig.add_trace(
                        go.Scattergl(
                            x=line_data[time_var].values,
                            y=line_data[plot_var].values,
                            mode="lines",
                            line=dict(width=1.5, color='grey'),
                            name=str(line),
                        ),
                        row=1,
                        col=i + 1,
                    )
    if add_hline:
        fig.add_hline(y=hline_y, row=1, col='all', line_width=1, opacity=1, line_color='black')
    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        font=dict(size=text_size, family=font_family),
        title_text=title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
    )
    fig.update_xaxes(title_text=x_title)
    if tick_angle is not None:
        fig.update_xaxes(tickangle=tick_angle)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)



## Misc. Graphs
## ============

def plotRasterAndTimeHistogram(
        raster,
        time,
        title=None,
        colorscale='gray_r',
        line_color='slategrey',
        x_title='Time (sec)',
        raster_y_title='Peaks',
        histogram_y_title='Mean Response',
        plot_height=600,
        plot_width=500,
        text_size=20,
        font_family='Arial',
        dtick=None,
        plot_scale=5,
        renderer='notebook',
        save_path=None):
    '''
    Assumes an input matrix (raster) where each row is a trial and each column is a timepoint, and a time vector. Plots a
    matrix of the input matrix on top with a trial-averaged histogram below it.
    '''

    mean = raster.mean(axis=0)
    sem  = raster.std(axis=0) / np.sqrt(raster.shape[0])

    fig = make_subplots(rows=2, shared_xaxes=True, x_title=x_title, vertical_spacing=0.05)
    fig.add_trace(go.Heatmap(x=time, z=raster, colorscale=colorscale, showscale=False, showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=time, y=(mean + sem),
                             mode='lines', fill=None, line_color=line_color, hoverinfo='skip', showlegend=False, name=histogram_y_title, legendgroup='mean'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=(mean - sem),
                             mode='lines', fill='tonexty', line=dict(color=line_color), hoverinfo='skip', showlegend=False, legendgroup='mean'), row=2, col=1)

    fig.update_yaxes(title_text=raster_y_title, row=1, col=1)
    fig.update_yaxes(title_text=histogram_y_title, row=2, col=1)
    if dtick is not None:
        fig.update_xaxes(dtick=dtick)
    # fig.update_yaxes(range=(0,np.ceil(mean)), row=2, col=1)
    fig.update_layout(template='simple_white', height=plot_height, width=plot_width, title_text=title,
                      font=dict(size=text_size, family=font_family))
    fig.update_annotations(font=dict(size=text_size+3))
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(renderer=renderer, config=config)