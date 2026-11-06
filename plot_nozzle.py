import matplotlib.pyplot as plt

from mlutil import colon


def plot_nozzle(nozzle_geometry, h, ax=None):
    """plot_nozzle(nozzle_geometry, h, ax)

    ax (optional): axes to plot into.
      - If omitted, a new figure/axes is created (original behavior).
      - If provided, plots into that axes.
    """
    g = nozzle_geometry
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title('Nozzle')

    def wall(region, r, style):
        ax.plot(region, r, style)
        ax.plot(region, -r, style)

    x1 = g[0]
    x2 = x1 + g[1]
    x3 = x2 + g[2]
    x4 = x3 + g[3]
    x5 = x4 + g[4]

    # Inlet
    region = colon(0, h, x1)
    wall(region, g[5] / 2 + 0 * region, 'r')

    # Converging
    region = colon(x1, h, x2)
    m = (g[6] / 2 - g[5] / 2) / g[1]
    wall(region, g[5] / 2 + m * (region - x1), 'r')

    # Throat
    region = colon(x2, h, x3)
    wall(region, g[6] / 2 + 0 * region, 'r')

    # Diverging
    region = colon(x3, h, x4)
    m = (g[7] / 2 - g[6] / 2) / g[3]
    wall(region, g[6] / 2 + m * (region - x3), 'r')

    # Standoff
    region = colon(x4, h, x5)
    wall(region, g[7] / 2 + 0 * region, 'k:')

    ax.set_aspect('equal')
    ax.axis('off')
    return ax
