import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm


class ColorbarTools:

    @staticmethod
    def label(label, ax=None):
        """
        Set a colorbar label. Works for horizontal/vertical.
        """
        if ax is None:
            ax = plt.gca()

        cbar = None
        for m in ax.figure.axes:
            if hasattr(m, 'colorbar') or m.get_label() == 'colorbar':
                cbar = m
                break

        # Fallback: find existing colorbars
        if cbar is None:
            try:
                cbar = ax.figure.colorbar_ax
            except Exception:
                raise RuntimeError("No colorbar found in this figure.")

        # Orientation check
        if cbar.get_orientation() in ("horizontal", "south", "north"):
            cbar.set_label(label, rotation=0)
        else:
            cbar.set_label(label)

    @staticmethod
    def zero_white(value=0, cmap="viridis", gap=0.1, vmin=None, vmax=None):
        """
        Insert white at `value` in colormap, with whitening gap.
        """
        cmap = plt.get_cmap(cmap)
        N = cmap.N
        colors = cmap(np.linspace(0, 1, N))

        if vmin is None or vmax is None:
            vmin, vmax = plt.gca().images[0].get_clim()

        vals = np.linspace(vmin, vmax, N)
        idx = np.argmin(np.abs(vals - value))
        colors[idx] = [1, 1, 1, 1]

        gap_idx = int(gap * N)
        for i in range(3):
            if idx - gap_idx >= 0:
                colors[idx-gap_idx:idx, i] = np.linspace(
                    colors[idx-gap_idx, i], 1, gap_idx
                )
            if idx + gap_idx < N:
                colors[idx:idx+gap_idx, i] = np.linspace(
                    1, colors[idx+gap_idx-1, i], gap_idx
                )

        return ListedColormap(colors)

    @staticmethod
    def nan(color="white", cmap="viridis"):
        """
        Add a distinct NaN color.
        """
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(color=color)
        return cmap

    @staticmethod
    def manual(domain, cmap="viridis"):
        """
        Rescale colormap to fit a custom domain.
        """
        cmap = plt.get_cmap(cmap)
        N = cmap.N
        domain = np.array(domain)

        # enforce sorted
        if not np.all(np.diff(domain) >= 0):
            raise ValueError("domain must be sorted ascending")

        # linear remap
        vals = np.linspace(domain.min(), domain.max(), N)
        colors = cmap(np.linspace(0, 1, N))
        return ListedColormap(colors)

    @staticmethod
    def center(value=0, cmap="seismic", gap=0.1, vmin=None, vmax=None):
        """
        Force white at center value (symmetric diverging colormap).
        """
        cmap = plt.get_cmap(cmap)
        N = cmap.N
        colors = cmap(np.linspace(0, 1, N))

        if vmin is None or vmax is None:
            vmin, vmax = plt.gca().images[0].get_clim()

        idx = int((value - vmin) / (vmax - vmin) * (N - 1))
        idx = np.clip(idx, 0, N - 1)

        gap_idx = int(gap * N)

        # Insert white band
        colors[idx, :] = [1, 1, 1, 1]
        if gap_idx > 0:
            for i in range(3):
                if idx - gap_idx >= 0:
                    colors[idx-gap_idx:idx, i] = np.linspace(
                        colors[idx-gap_idx, i], 1, gap_idx
                    )
                if idx + gap_idx < N:
                    colors[idx:idx+gap_idx, i] = np.linspace(
                        1, colors[idx+gap_idx-1, i], gap_idx
                    )
        return ListedColormap(colors)

    @staticmethod
    def opt(cmap="viridis", ax=None, data=None, invert=False):
        """
        Resample colormap according to data density.
        """
        if ax is None:
            ax = plt.gca()

        if data is None:
            im = ax.images[0]
            data = im.get_array().ravel()
            vmin, vmax = im.get_clim()
        else:
            data = np.ravel(data)
            vmin, vmax = np.nanmin(data), np.nanmax(data)

        cmap = plt.get_cmap(cmap)
        N = cmap.N
        colors = cmap(np.linspace(0, 1, N))

        # Histogram weighting
        hist, bins = np.histogram(data[np.isfinite(data)], bins=N, range=(vmin, vmax))
        weights = hist**2 / hist.sum()
        # cumulative distribution
        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]

        if invert:
            cdf = 1 - cdf[::-1]  # flip cumulative distribution
        new_vals = cdf * (vmax - vmin) + vmin

        resampled = np.zeros((N, 4))
        for i in range(4):
            resampled[:, i] = np.interp(new_vals, np.linspace(vmin, vmax, N), colors[:, i])

        return ListedColormap(resampled)
