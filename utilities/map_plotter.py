import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy
import cartopy.crs as ccrs


def plot_cartopy_map(ax: plt.subplot, high_quality=True, scale='low', add_greenland=True) -> tuple[classmethod]:
    """
    Generates a plot of globe with cartopy GSHHS.
    :param ax: axis (matplotlib.plt.subplot) with projection not set to None.
        subplot axis object in which map is plotted
    :param high_quality: boolean
        Boolean which decides if both GSHSS and NaturalEarth (NE) are used for optimal resolution, or only NE.
        (default: True)
    :param scale: string
        resolution specific to cartopy GSHHS (default: 'low')
    :param add_greenland: bool (optional)
    :return: class method
        grid-line (is not needed for plotting, but can be useful to change its parameters)
    """
    if high_quality:
        # add high quality GSHHS coastlines without Antarctica
        coast_1 = cfeature.GSHHSFeature(scale=scale, levels=[1])
        ax.add_feature(coast_1)
        if add_greenland is True:
            ax.add_feature(cfeature.COASTLINE, alpha=1, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_feature(cfeature.RIVERS, alpha=1, linewidth=1, edgecolor='k', facecolor='none')
    elif high_quality is False:
        coast_1 = cfeature.NaturalEarthFeature(category='physical', scale='110m', name='coastline', facecolor='none')
        ax.add_feature(coast_1, alpha=1, linewidth=1)
        ax.add_feature(cartopy.feature.NaturalEarthFeature(
        category='physical', name='rivers_lake_centerlines',
        scale='110m', facecolor='none', edgecolor='k'), alpha=1, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_feature(cfeature.LAKES, alpha=1, linewidth=1, edgecolor='k')
    # add gridlines
    gl = ax.gridlines(
        draw_labels=True, alpha=0.3)
    # remove right labels
    gl.right_labels = False
    # specifies the size of xticks
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}

    return gl, coast_1