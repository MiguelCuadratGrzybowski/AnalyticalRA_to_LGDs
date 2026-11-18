import pandas as pd
import numpy as np
from scipy import interpolate, ndimage
from scipy.signal import sosfiltfilt, savgol_filter, butter, bessel, cheby2
import pymap3d as pm
from scipy.linalg import lstsq



def retrieve_gfm_year_and_months_list(year: int or float, **months) -> tuple:
    """
    Function is specific to the retrieval of the aforementioned start of a specific year (and optionally month).
    :param year: int or float
    :param months: dict with keys
        Optional months 1 and 2 for start and end dates.
    :return: tuple
        start_date, end_date, months of that year
    """
    path_dates = (r'..\monthly_dates\months_def_CSR.csv')
    df_GFM_dates = pd.read_csv(path_dates, sep='\t')

    years = np.arange(2002, 2018, 1)
    GFM_dict = {}
    for i in range(len(years)):
        GFM_dict.__setitem__(years[i], dict())

    for i in range(len(years)):
        df_year = df_GFM_dates[df_GFM_dates.year == years[i]]
        GFM, GFM_start, GFM_end = df_year['month'].to_numpy(), df_year['start'].to_numpy(), df_year['end'].to_numpy()
        GFM_start_date, GFM_end_date = df_year['datetime_start'].to_numpy(), df_year['datetime_end'].to_numpy()
        for j in range(len(GFM)):
            GFM_dict[years[i]].__setitem__(GFM[j], {'start': GFM_start[j], 'end': GFM_end[j],
                                                    'start_date': GFM_start_date[j], 'end_date': GFM_end_date[j]})

    # all the GFMs of that year
    all_monhtly_GFMs = GFM_dict[year]
    # retrieve the months
    months_GFMs = list(all_monhtly_GFMs.keys())
    if len(list(months.keys())) == 0:
        min_month, max_month = min(months_GFMs), max(months_GFMs)
    else:
        min_month, max_month = list(months.values())
    # start, end
    date_start, date_end = all_monhtly_GFMs[min_month]['start'], all_monhtly_GFMs[max_month]['end']
    return date_start, date_end, all_monhtly_GFMs


# group months function
def group_months(number_of_months: int, list_of_months: list) -> list[list]:
    """
    Function groups the months starting and ending dates.
    :param number_of_months:
    :param list_of_months:
    :return: list[list]
    """
    grouped_months = []
    current_group = [list_of_months[0]]  # Initialize the first group

    for month in list_of_months[1:]:
        if len(current_group) < number_of_months:  # If month follows the previous month
            current_group.append(month)
        else:  # Start a new group
            grouped_months.append(current_group)
            current_group = [month]

    grouped_months.append(current_group)  # Add the last group

    return grouped_months


# Function that identifies gaps within a data-set.
def get_time_pieces(time_array: np.array, tol: float | int, divider=False) -> (
        tuple)[tuple[np.array, np.array], list | None]:
    """
    Function divides time arrays into pieces depending on a certain time tolerance.
    :param time_array: np.array
        Time array with specific time step and data gaps.
    :param tol: float or int
        Maximum allowable time period, which is used to define a gap.
    :param divider: boolean
        Default is False, if set to True, the actual pieces are returned as well.
    :return: tuple[list, list]
        Gives the indices of the data pieces. If set to true, the time pieces are returned as well.
    """
    # Calculate differences between consecutive timestamps
    diffs = np.diff(time_array)

    # Find indices where differences exceed tolerance
    gap_indices = np.where(diffs >= tol)[0]

    # Add start and end indices for each piece
    start_indices = np.concatenate(([0], gap_indices + 1))
    end_indices = np.concatenate((gap_indices, [len(time_array) - 1]))

    # Separate pieces based on start and end indices if divider is True
    if divider is True:
        pieces = [time_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
    else:
        pieces = None

    return (start_indices, end_indices), pieces


# Functions related to gaps and interpolation
def fill_in_gaps(value_0, value_1, t_0, t_1, dt=5) -> tuple[np.array, np.array]:
    """
    Function fills gaps with linear interpolation.
    :param value_0: int or float
    :param value_1: int or float
    :param t_0: int or float
    :param t_1: int or float
    :param dt: int or float
    :return: tuple[np.array, np.array]
        gap_values, gap_times
    """
    N = (t_1 - t_0)/dt
    gap_times = np.linspace(t_0, t_1, N)
    gap_values = value_0 + (gap_times-t_0)*(value_1-value_0)/(t_1-t_0)
    return gap_values, gap_times


def interpolate_array(y_array: np.array, x_array: np.array, dx: float or int) -> tuple[np.array, np.array]:
    """
    Function interpolates the array with a given step size dx. This function conveniently provides the interpolated data
    using numpy.interp.
    :param y_array: np.array
        Time-series to be linearly interpolated.
    :param x_array: np.array
        Associated x-values of the original data time-series (y_array).
    :param dx: float or int
        Step size of the wished interpolated data array.
    :return: tuple[np.array, np.array]
        Interpolated data values and x-values with step size dx.
    """
    # check if it is a pandas.Dataframe, transforms them into numpy.arrays
    if not isinstance(x_array, (np.ndarray, np.generic)) and not isinstance(x_array, list):
        x_array = x_array.to_numpy()
    if not isinstance(y_array, (np.ndarray, np.generic)) and not isinstance(x_array, list):
        y_array = y_array.to_numpy()
    # check size of array (piece) to be higher than 1
    if len(x_array) > 1:
        # for now this is essential: flip the order, sometimes this is necessary!
        if x_array[-1] < x_array[0]:
            x_array = np.flip(x_array)
        x_int_array = np.arange(x_array[0], x_array[-1]+dx, dx)
        y_int_array = np.interp(x_int_array, x_array, y_array)
    if len(x_array) <= 1:
        x_int_array, y_int_array = x_array, y_array

    return y_int_array, x_int_array


def interpolate_array_to_size(y_array: np.array, x_array: np.array, x_int_array: np.array) -> np.array:
    """
    Function interpolates the array with a given size. This function conveniently provides the interpolated data
    using numpy.interp.
    :param y_array: np.array
        Time-series to be linearly interpolated.
    :param x_array: np.array
        Associated x-values of the original data time-series (y_array).
    :param x_int_array: np.array
        Associated already interpolated x-values of the original data time-series.
    :return: np.array
        Interpolated data values from original and interpolated x-values.
    """
    # check if it is a pandas.Dataframe, transforms them into numpy.arrays
    if isinstance(x_array, pd.DataFrame):
        x_array = x_array.to_numpy()
    if isinstance(y_array, pd.DataFrame):
        y_array = y_array.to_numpy()
    y_int_array = np.interp(x_int_array, x_array, y_array)
    return y_int_array


def interpolate_to_match_size(array1, array2, label1='array1', label2='array2'):
    # Determine which array is smaller and which is larger
    if len(array1) < len(array2):
        small_array, large_array = array1, array2
        small_label, large_label = label1, label2
    else:
        small_array, large_array = array2, array1
        small_label, large_label = label2, label1

    # Interpolation points (indices for the larger array size)
    interp_points = np.linspace(0, len(small_array) - 1, num=len(large_array))

    # Interpolation function
    interp_func = interpolate.interp1d(np.arange(len(small_array)), small_array, kind='linear')

    # Interpolated values
    interpolated_array = interp_func(interp_points)
    return {small_label: interpolated_array, large_label: large_array}


# Three functions (4 counting the deprecated one) for differentiation for own method
def bdtaylor(n: int, f=0.2) -> np.ndarray:
    """
    bdtaylor: int, float -> ndarray
    Function calculates the coefficients for the n-th order backward difference approximation.
    :param n: int
        Order of approximation.
    :param f: float
        Sampling frequency in Hz (default = 0.2).
    :return: ndarray
        Coefficients of n-th order approximation.

    Original author (matlab version):
        Weigelt, GI Stuttgart (14.04.11)
    """
    # Recursive Calculation
    coeff = np.zeros(n + 1)
    coeff[0] = -np.sum(1 / np.arange(1, n + 1))
    coeff[1] = n
    for k in range(2, n + 1):
        coeff[k] = -coeff[k - 1] * (k - 1) * (n - k + 1) / (k ** 2)

    # Applying the sampling rate
    coeff = -np.flip(coeff) * f

    return coeff


def cdtaylor(n: int, f=0.2) -> np.ndarray:
    """
    cdtaylor: int, float -> ndarray
    Function calculates the coefficients for the n-th order central difference approximation.
    :param n: int
        Order of approximation.
    :param f: float
        Sampling frequency in Hz (default = 1).
    :return: ndarray
        Coefficients of n-th order approximation.

    Original author (matlab version):
        Matthias Weigelt (21.06.04)
    """
    # Recursive Calculation
    rhcoeff = np.zeros(n)
    rhcoeff[0] = n / (n + 1)
    for k in range(2, n + 1):
        rhcoeff[k - 1] = -rhcoeff[k - 2] * (k - 1) * (n - k + 1) / (k * (n + k))

    # Set up coefficients
    coeff = np.concatenate((np.flip(-rhcoeff), [0], rhcoeff))

    # Applying the sampling rate
    coeff = coeff * f

    return coeff


def fdtaylor(n, f=0.2):
    """
    Function calculates the coefficients for the n-th order accurate forward
    difference approximation of the first derivative.
    :param n: int
        Order of approximation.
    :param f: float
        Sampling frequency in Hz (default = 0.2).
    :return: ndarray
        Coefficients of n-th order approximation.

    Original author (matlab version):
        Matthias Weigelt (21.06.04)
    Recursive Calculation:
    # from "Closed-form expression for the finite difference approximations of
    # first and higher derivatives based on Taylor series", Khan and Ohba,
    # Journal of Computational and Applied Mathematics 107 (1999) 179-193
    """

    # Right-handed coefficients
    coeff = np.zeros(n + 1)
    coeff[0] = -np.sum(1 / np.arange(1, n + 1))
    coeff[1] = n
    for k in range(2, n + 1):
        coeff[k] = -coeff[k - 1] * (k - 1) * (n - k + 1) / ((k) ** 2)

    # Applying the sampling rate
    coeff *= f

    return coeff.flatten()


def low_pass_diff_coeff(K, L):
    """
    Maximally flat FIR lowpass differentiator (Type-III and IV)
    :param K: int
        Number of zeros at z = -1.
    :param L: int
        Number of zeros away from z = -1 and z = 1.
    :return: numpy.ndarray
        Coefficients of the lowpass differentiator filter.

    Example of use:
        K = 7
        L = 5
        h = low_pass_diff_coeff(K, L)
        w, H = scipy.signal.freqz(h)
        plt.plot(w / np.pi, np.abs(H))
        plt.show()
    Reference and original matlab code:
    - Ivan Selesnick, Polytechnic University, Brooklyn, NY
    - Email: selesi@taco.poly.edu
    - January 2001
    """

    c = np.zeros(L + 1, dtype=float)
    c[0] = 2
    c[1] = K + 1 / 3

    for n in range(2, L + 1):
        c[n] = ((-K + 4 * K * n - 10 * n + 3 + 8 * n ** 2) * c[n - 1] - (2 * n + K - 3) ** 2 * c[n - 2]) / (
                    2 * n * (2 * n + 1))

    q = 1
    h = c[0]

    for n in range(1, L + 1):
        q = np.convolve(q, np.array([-1, 2, -1]) / 4)
        h = np.convolve(h, np.array([0, 1, 0])) + c[n] * q

    for k in range(K):
        h = np.convolve(h, np.array([1, 1]) / 2)

    h = np.convolve(h, np.array([1, -1]) / 2)

    return h

def _robust_dt_from_x(x: np.ndarray) -> float:
    """Robust estimate of sampling step from a (mostly) increasing time array."""
    x = np.asarray(x)
    if x.size < 2:
        return np.nan
    dx = np.diff(x.astype(np.float64))
    dx = dx[np.isfinite(dx) & (dx > 0)]
    if dx.size == 0:
        return np.nan
    return float(np.median(dx))

def differentiate_central_scheme(
    y_array,
    x_array,
    n,
    fs=None,
    id_gaps=True,
    tol=20,
    int_flag=True,
    dx_interp=None,   # NEW: if None, uses 1/fs or median dt
):
    """
    Central difference 1st derivative with edge forward/backward schemes.
    - Splits by gaps >= tol (seconds) when id_gaps=True
    - Optionally interpolates small gaps onto a regular grid
    - Uses dt inferred from interpolated time grid (or from fs if provided)
    """

    # ---- input checks ----
    if not isinstance(n, (int, float)):
        raise ValueError("n must be int/float (float will be rounded).")
    n = int(round(n))
    if n < 1:
        raise ValueError("n must be >= 1")

    y_array = np.asarray(y_array)
    if not isinstance(x_array, (np.ndarray, np.generic)):
        x_array = x_array.to_numpy()
    x_array = np.asarray(x_array)

    # ---- split into pieces (large gaps) ----
    if id_gaps:
        pieces_indices, _ = get_time_pieces(x_array, tol=tol)
        start_indices, end_indices = pieces_indices
        y_pieces = [y_array[s:e + 1] for s, e in zip(start_indices, end_indices)]
        x_pieces = [x_array[s:e + 1] for s, e in zip(start_indices, end_indices)]
    else:
        y_pieces = [y_array]
        x_pieces = [x_array]

    coeff_cd_1 = cdtaylor(n, 1)
    coeff_fd_1 = fdtaylor(n, 1)
    coeff_bd_1 = bdtaylor(n, 1)

    dy_dx_pieces = []
    x_out_pieces = []

    for y_piece, x_piece in zip(y_pieces, x_pieces):
        y_piece = np.asarray(y_piece)
        x_piece = np.asarray(x_piece)

        # handle tiny pieces safely
        if y_piece.size == 0:
            continue
        if y_piece.size == 1:
            dy_dx_pieces.append(np.zeros_like(y_piece, dtype=np.float64))
            x_out_pieces.append(x_piece.astype(np.float64))
            continue

        # ensure float
        if y_piece.dtype not in (np.float64, np.float32):
            y_piece = y_piece.astype(np.float64)

        # ---- optional interpolation onto regular grid ----
        if int_flag:
            # choose dx for interpolation
            if dx_interp is None:
                if fs is not None and np.isfinite(fs) and fs > 0:
                    dx_use = 1.0 / float(fs)
                else:
                    # fall back to median dt of this piece
                    dx_use = _robust_dt_from_x(x_piece)
                    if not np.isfinite(dx_use) or dx_use <= 0:
                        dx_use = 1.0  # last-resort fallback
            else:
                dx_use = float(dx_interp)

            y_int, x_int = interpolate_array(y_piece, x_piece, dx=dx_use)
        else:
            y_int, x_int = y_piece, x_piece.astype(np.float64)

        size = y_int.size

        # determine dt/fs from the (possibly interpolated) time axis
        dt = _robust_dt_from_x(x_int)
        if not np.isfinite(dt) or dt <= 0:
            # fall back to provided fs if sane, else 1
            if fs is not None and np.isfinite(fs) and fs > 0:
                dt = 1.0 / float(fs)
            else:
                dt = 1.0
        fs_eff = 1.0 / dt

        # ---- if not enough points for 2n central stencil ----
        if size <= 2 * n:
            if size < 2:
                dy = np.zeros(size, dtype=np.float64)
            else:
                # simple first difference then resize to match x_int
                dy_raw = fs_eff * np.diff(y_int)
                dy = interpolate_array_to_size(dy_raw, x_int[:-1], x_int)
            dy_dx_pieces.append(dy)
            x_out_pieces.append(x_int)
            continue

        # ---- main differentiation ----
        dy = np.zeros(size, dtype=np.float64)

        # forward edge (first n points)
        dy_front = np.zeros(n, dtype=np.float64)
        for i in range(n):
            slice_y = y_int[i:i + len(coeff_fd_1)]
            dy_front[i] = fs_eff * np.dot(coeff_fd_1, slice_y)

        # backward edge (last n points)
        for i in range(size - n, size):
            slice_y = y_int[i - len(coeff_bd_1) + 1:i + 1]
            dy[i] = fs_eff * np.dot(coeff_bd_1, slice_y)
        dy_back = dy[size - n:size].copy()

        # central convolution (discard edges)
        dy_central = fs_eff * ndimage.convolve1d(
            y_int, -coeff_cd_1, axis=-1, mode="nearest"
        )

        dy = np.concatenate((dy_front, dy_central[n:size - n], dy_back))

        dy_dx_pieces.append(dy)
        x_out_pieces.append(x_int)

    if len(dy_dx_pieces) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.concatenate(dy_dx_pieces), np.concatenate(x_out_pieces)

# def differentiate_central_scheme(y_array: np.ndarray or np.array, x_array: np.ndarray or np.array,
#                                  n: int, fs: float, id_gaps=True, tol=20, int_flag=True) -> tuple[np.ndarray, np.array]:
#     """
#     Calculates the central difference 1st order derivative with an accuracy order 2n.
#     At the edges of each section, a n^th accuracy forward and backward difference scheme is applied.
#     Uses scipy.ndimage.convolve1d() for central finite difference.
#     Large gaps (>= tol) allow the section distinction. Small gaps are linearly interpolated.
#     :param y_array: ndarray
#         Array to be differentiated.
#     :param x_array: ndarray
#         X-coordinates values associated to the y_array.
#     :param n: int
#         Accuracy order for the edges, whereas the central difference exhibits a 2n accuracy order.
#     :param fs: float
#         Sampling frequency f_s = 1/dt, where dt is sampling step size.
#     :param id_gaps: boolean
#         If set to False, the differentiator does not distinguish if there are gaps within the data-set. (default: True)
#     :param tol: float or int
#         Tolerance for identifying a (large) data gap in [s]. Default: 20 s.
#     :param int_flag: boolean
#         If set to False, the interpolation process is skipped. (default: True)
#     :return: tuple[ndarray, np.array]
#         First derivative of the y_array, with n^th order accuracy. Interpolated small gaps x-axis!
#     """
#
#     # input check
#     if not isinstance(n, int) and not isinstance(n, float):
#         raise ValueError('The accuracy order must be an integer '
#                          '(floats are transformed to nearest integer)!')
#     if not isinstance(n, int):
#         # in the case of a float, change it to the nearest integer. Make sure to use integer.
#         n = int(n)
#     if not isinstance(x_array, (np.ndarray, np.generic)):
#         x_array = x_array.to_numpy()
#
#     # checks if gaps need to be identified.
#     if id_gaps:
#         # gets the indices of the gaps
#         pieces_indices, _ = get_time_pieces(x_array, tol=tol)
#         start_indices, end_indices = pieces_indices
#         # separates the y_array into pieces, defined by tolerance requirement
#         y_array_pieces = [y_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
#         x_array_pieces_old = [x_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
#     if not id_gaps:
#         y_array_pieces = [y_array]
#         x_array_pieces_old = [x_array]
#
#     # initialise entire list for dy_dx
#     dy_dx_array_pieces, x_array_pieces = [], []
#
#     # compute the central difference scheme coefficients (no sampling applied yet)
#     coeff_cd_1 = cdtaylor(n, 1)
#     coeff_fd_1 = fdtaylor(n, 1)
#     coeff_bd_1 = bdtaylor(n, 1)
#
#     # loop within the pieces
#     ind_piece = 0
#     for y_array in y_array_pieces:
#         # retrieve x_array of same size as the piece y_array!
#         x_array = x_array_pieces_old[ind_piece]
#
#         # avoid numerical issues due to how floats are written in big numbers!
#         y_array = np.asarray(y_array)
#         # Ensure that x is either single or double precision floating point.
#         if y_array.dtype != np.float64 and y_array.dtype != np.float32:
#             y_array = y_array.astype(np.float64)
#
#         # interpolate linearly small gaps (< 20 s) if int_flag is True!
#         if int_flag is True:
#             y_array, x_int_array = interpolate_array(y_array, x_array, dx=5)
#         if int_flag is False:
#             x_int_array = x_array_pieces_old[ind_piece]
#
#         # size of the updated/interpolated data array piece.
#         size = len(y_array)
#         # if not sufficient points for nth order
#         if size <= 2*n:
#             # remove if less than 2 points!
#             if size < 2: dy_dx_array = [0]*size  # array full of 0s (dampens the mean signal)
#             # use simple numpy.diff function
#             else:
#                 dx_array = x_int_array[:-1]
#                 dy_dx_array = interpolate_array_to_size(fs*np.diff(y_array), dx_array, x_int_array)
#
#         # else perform differentiation algorithm
#         if size > 2*n:
#             # create empty derivative array.
#             dy_dx_array = np.zeros(size)
#
#             # Compute forward and backward differences only for the edges
#             dy_dx_array_front = np.zeros(n)
#
#             # Applying forward difference coefficients
#             for i in range(n):
#                 # dx = x_array[i+1] - x_array[i]
#                 slice_y = y_array[i:i + len(coeff_fd_1)]
#                 dy_dx_array_front[i] = fs * np.dot(coeff_fd_1, slice_y)
#
#             # Applying backward difference coefficients
#             for i in range(size - n, size):
#                 # dx = x_array[i] - x_array[i - 1]
#                 slice_y = y_array[i - len(coeff_bd_1) + 1:i + 1]
#                 dy_dx_array[i] = fs * np.dot(coeff_bd_1, slice_y)
#             dy_dx_array_back = dy_dx_array[size - n:size]
#
#             # Compute central differences for the main part of the array (beware order of coefficients is reversed here)
#             dy_dx_array_central = fs * ndimage.convolve1d(y_array, -coeff_cd_1, axis=-1)
#
#             # Concatenate edge and central parts (by getting rid of incorrect edge implementations!)
#             dy_dx_array = np.concatenate((dy_dx_array_front, dy_dx_array_central[n:size-n], dy_dx_array_back))
#
#         # Append to mega list of dy_dx_array_pieces:
#         dy_dx_array_pieces.append(dy_dx_array), x_array_pieces.append(x_int_array)
#         ind_piece += 1  # update the index of the piece
#
#     # Concatenate the total result made out of the pieces
#     dy_dx_array = np.concatenate(dy_dx_array_pieces)
#     x_array_total = np.concatenate(x_array_pieces)
#     return dy_dx_array, x_array_total


def filter_array_with_noise_model(array_to_be_filtered: np.ndarray,
                                  t_array: np.ndarray,
                                  h=450e3,
                                  mu=3.986004418e14,
                                  Re=6371e3):
    """
    Apply a low-frequency noise model per orbital revolution (T_rev) to the input array.

    Noise model:
    n(t) = A + Bt + Ct^2 + D sin(2π t / T_rev) + E cos(2π t / T_rev) + F t sin(2π t / T_rev) + G t cos(2π t / T_rev)

    Parameters
    ----------
    array_to_be_filtered : np.ndarray
        Input signal array.
    t_array : np.ndarray
        Corresponding time array in seconds.
    h : float
        Satellite altitude in meters (default 450 km).
    mu : float
        Earth's gravitational parameter in m^3/s^2 (default 3.986e14).
    Re : float
        Earth's radius in meters (default 6371 km).

    Returns
    -------
    array_corrected : np.ndarray
        Signal with low-frequency noise removed per orbital revolution.
    """
    # Compute orbital period
    a = Re + h
    T_rev = 2 * np.pi * np.sqrt(a**3 / mu)

    array_corrected = []
    t_arrays_corrected = []

    # Define orbit start times: every T_rev starting from first timestamp
    if len(t_array)>1:
        t_start = t_array[0]
        t_end = t_array[-1]
    else:
        return array_to_be_filtered, t_array

    orbit_edges = np.arange(t_start, t_end + T_rev, T_rev)

    # Loop over each orbit segment
    for i in range(len(orbit_edges)-1):
        mask = (t_array >= orbit_edges[i]) & (t_array < orbit_edges[i+1])
        t_seg = t_array[mask]
        y_seg = array_to_be_filtered[mask]

        min_points = 1 * 7  # 1× number of parameters
        if len(y_seg) < min_points:
            if len(y_seg) != 0:
                print(f"not filtered - {len(y_seg)}")
            continue

        # Design matrix for low-frequency noise
        omega = 2 * np.pi / T_rev
        X = np.column_stack([
            np.ones_like(t_seg),
            t_seg,
            t_seg**2,
            np.sin(omega*t_seg),
            np.cos(omega*t_seg),
            t_seg*np.sin(omega*t_seg),
            t_seg*np.cos(omega*t_seg)
        ])

        # Least-squares fit
        coeffs, _, _, _ = lstsq(X, y_seg)
        n_fit = X @ coeffs

        # Subtract noise
        array_corrected.append(y_seg - n_fit)
        t_arrays_corrected.append(t_seg)

    # Concatenate all corrected segments
    array_final_filtered = np.concatenate(array_corrected)
    t_final_array = np.concatenate(t_arrays_corrected)

    return array_final_filtered, t_final_array


def _padlen_for_sos(sos, n: int) -> int:
    """
    Conservative padlen choice that won't exceed n-1.
    Default sosfiltfilt padlen is often 3*(2*sections) style; we cap safely.
    """
    if n <= 1:
        return 0
    # a common safe heuristic:
    default = 3 * (sos.shape[0])  # sos.shape[0] = number of biquad sections
    return int(min(n - 1, max(0, default)))

def filter_array(
    array,
    t_array=None,
    method="butw",
    N=1,
    btype="low",
    cutoff=1e-2,
    **optional
):
    """
    Butterworth/Bessel/Cheby2 SOS zero-phase filtering with gap splitting.
    If t_array is None: filters whole array and returns (filtered, None).
    If int_flag=True: interpolates onto regular grid before filtering.
    """
    sos_methods_dic = {"butw": butter, "bessel": bessel, "cheby2": cheby2}

    array = np.asarray(array)

    # ----- no time axis: just filter the full array -----
    if t_array is None:
        fs = float(optional.get("fs", 0.2))
        sos = sos_methods_dic[method](N, cutoff, btype, fs=fs, output="sos")

        padlen = _padlen_for_sos(sos, array.size)
        filtered = sosfiltfilt(sos, array, padlen=padlen, padtype="even") if array.size > 1 else array.copy()
        return filtered, None

    # ----- time axis present: split pieces by gaps -----
    t_array = np.asarray(t_array)
    tol = float(optional.get("tol", 20))
    pieces_indices, _ = get_time_pieces(t_array, tol=tol)
    start_indices, end_indices = pieces_indices

    array_pieces = [array[s:e + 1] for s, e in zip(start_indices, end_indices)]
    t_pieces = [t_array[s:e + 1] for s, e in zip(start_indices, end_indices)]

    int_flag = bool(optional.get("int_flag", False))
    dx_interp = optional.get("dx_interp", None)  # NEW: can force grid step

    out_arr, out_t = [], []

    for a_piece, t_piece in zip(array_pieces, t_pieces):
        if a_piece.size == 0:
            continue
        if a_piece.size == 1:
            out_arr.append(a_piece.astype(np.float64))
            out_t.append(t_piece.astype(np.float64))
            continue

        # optional interpolation
        if int_flag:
            if dx_interp is None:
                dt = _robust_dt_from_x(t_piece)
                if not np.isfinite(dt) or dt <= 0:
                    dt = 5.0
                dx_use = dt
            else:
                dx_use = float(dx_interp)

            a_int, t_int = interpolate_array(a_piece, t_piece, dx=dx_use)
        else:
            a_int, t_int = a_piece, t_piece.astype(np.float64)

        # infer fs from actual time grid unless provided
        if "fs" in optional:
            fs = float(optional["fs"])
        else:
            dt_eff = _robust_dt_from_x(t_int)
            fs = (1.0 / dt_eff) if (np.isfinite(dt_eff) and dt_eff > 0) else 0.2

        sos = sos_methods_dic[method](N, cutoff, btype, fs=fs, output="sos")

        padlen = _padlen_for_sos(sos, a_int.size)
        a_filt = sosfiltfilt(sos, a_int, padlen=padlen, padtype="even") if a_int.size > 1 else a_int.copy()

        out_arr.append(a_filt)
        out_t.append(t_int)

    if len(out_arr) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.concatenate(out_arr), np.concatenate(out_t)


# def filter_array(array: np.ndarray or pd.DataFrame, t_array=None, method='butw',
#                  N=1, btype='low', cutoff=1e-2, **optional) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Function that conveniently groups all relevant filtering options.
#     Savgol-Golay filtering using scipy.signal.savgol_golay.
#     Locally Weighted Scatterplot Smoothing (LOWESS), reference of source:
#         Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.”
#         Proceedings of the 9th Python in Science Conference. 2010.
#         Algorithm of reference:
#         Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”.
#         Journal of the American Statistical Association 74 (368): 829-836.
#     Low- or high-pass filtering are implemented using butterworth algorithm (scipy.signal)
#     Simple linear detrending (scipy.signal.detrend)
#     :param array: np.ndarray
#         Data array to be filtered.
#     :param t_array: np.ndarray
#         Time or x-array associated to data array. Default is None.
#     :param method: string
#         Method to be used in filtering (default: Butterworth 'butw' from scipy.signal).
#         For other filters: Savgol-G. ('sg'), LOWESS (lowess), linear or mean detrending ('det') or cyclical ('cycl').
#     :param w_p: tupe (2 integer entries)
#         Window length and order of fitting polynomial for filtering for Savgol-Golay filter
#     :param frac: float between 0 and 1
#         Necessary input for LOWESS, fraction of data array to use in filtering process.
#     :param N: int
#         Order of filter (default: 1)
#     :param btype: string
#         Type of filter (default: 'low'), for high-pass 'hp'
#     :param cutoff: float
#         Break frequency from which or up to which the filters act.
#     :param optional: dict
#         optional inputs (such as sampling frequency, 'fs', set to 1/5 s as default).
#     :return: tuple[ndarray, ndarray]
#         Filtered array, time array.
#     """
#     # define generic dictionary with names of various filters
#     sos_methods_dic = {'butw': butter,
#                        'bessel': bessel,
#                        'cheby2': cheby2}
#
#     # initialise list that will contain the filtered arrays
#     list_of_pieces, list_of_tpieces = [], []
#     t_array_pieces_old = t_array
#     if t_array is not None:
#         if 'tol' in optional.keys(): tol = optional['tol']
#         else: tol = 20
#         # gets the indices of the gaps
#         pieces_indices, _ = get_time_pieces(t_array, tol=tol)
#         start_indices, end_indices = pieces_indices
#         # separates the y_array into pieces, defined by tolerance requirement
#         array_pieces = [array[start:end + 1] for start, end in zip(start_indices, end_indices)]
#         t_array_pieces = [t_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
#     else:
#         array_pieces = [array]
#         t_array_pieces = [t_array]
#
#     # array_pieces = [array]
#     # t_array_pieces = [t_array]
#
#     ind_piece = 0
#     for array, t_array in zip(array_pieces, t_array_pieces):
#         if len(t_array) != 0 and len(array) != 0:
#             # initialise interpolation to False
#             int_flag = False
#             # if asked otherwise do it
#             if 'int_flag' in list(optional.keys()): int_flag = optional['int_flag']
#
#             # interpolate linearly small gaps (< 20 s) if int_flag is True!
#             if int_flag is True:
#                 # print('filter array', len(array), len(t_array))
#                 array_int, t_int_array = interpolate_array(array, t_array, dx=5)
#                 # print('filter array', len(array_int), len(t_int_array))
#             if int_flag is False:
#                 array_int, t_int_array = array_pieces[ind_piece], t_array_pieces[ind_piece]
#
#             # example: sos = butter(N, cutoff, btype, fs=fs, output='sos')
#             sos = sos_methods_dic[method](N, cutoff, btype, fs=0.2, output='sos')
#
#             # in the most simple implementation sosfiltfilt implements twice sosfilt, with padding and edge fidelity!
#             # >>> array_filtered = sosfilt(sos, array)
#             # >>> array_filtered = sosfilt(sos, array_filtered[::-1])[::-1]
#             # add only padding when the array size allows it
#             if len(array) > 3*N + 3:
#                 padlen = None
#             else:
#                 padlen = len(array)-1
#
#             array_filtered = sosfiltfilt(sos, array_int, padlen=padlen, padtype='even')
#
#
#             # add piece of array to the to-be concanated list
#             list_of_pieces.append(array_filtered), list_of_tpieces.append(t_int_array)
#             ind_piece += 1
#
#     # concatenate final list with all array pieces
#     array_filtered = np.concatenate(list_of_pieces)
#     t_array_final = np.concatenate(list_of_tpieces)
#
#     return array_filtered, t_array_final

def transfer_function(frequency_array: np.ndarray) -> np.ndarray:
    """
    Analytical transfer function Z(f) from range-acceleration residuals to LGD,
    following the piecewise form used in Ghobadi-Far et al. (2018), with a smooth
    log-linear bridge between 0.3 and 0.7 mHz.

    Ghobadi-Far, K., Han, S.-C., Weller, S., Loomis, B. D., Luthcke, S. B., Mayer-Gürr, T., & Behzadpour, S. (2018).
    A Transfer Function Between Line-of-Sight Gravity Difference and GRACE Intersatellite Ranging Data and
    an Application to Hydrological Surface Mass Variation.
    Journal of Geophysical Research: Solid Earth, 123(10), 9186–9201. https://doi.org/10.1029/2018JB016088

    Piecewise:
      - f < f1              : Z(f) = 0
      - f1 <= f < f2        : Z(f) = a*log10(f) + b, with continuity at f2
      - f >= f2             : Z(f) = 1 + c * f^(-p)

    Parameters
    ----------
    frequency_array : ndarray
        Frequencies in Hz (can include 0).

    Returns
    -------
    ndarray
        Z(f) evaluated at frequency_array (same shape as input).
    """
    f = np.asarray(frequency_array, dtype=np.float64)
    Z = np.zeros_like(f, dtype=np.float64)

    # thresholds (Hz)
    f1 = 0.3e-3
    f2 = 0.7e-3

    # high-frequency branch parameters (as in your code)
    c = 3.45e-4
    p = 1.04

    # Value at f2 from HF branch (continuity target)
    Z_f2 = 1.0 + c * (f2 ** (-p))

    # log-linear bridge coefficients a,b such that:
    #   Z(f1)=0  and  Z(f2)=Z_f2
    # a*(log10(f1)-log10(f2)) = -Z_f2  ->  a = Z_f2/(log10(f2)-log10(f1))
    denom = (np.log10(f2) - np.log10(f1))
    a = Z_f2 / denom
    b = -a * np.log10(f1)

    # Masks
    m_hf = f >= f2
    m_mid = (f >= f1) & (f < f2)
    # m_lf = f < f1 already zeros

    # Mid band: avoid log10(0) safely by masking (0 not included here)
    Z[m_mid] = a * np.log10(f[m_mid]) + b

    # High band
    Z[m_hf] = 1.0 + c * (f[m_hf] ** (-p))

    # Anything non-finite -> 0 (defensive)
    Z[~np.isfinite(Z)] = 0.0
    return Z

# import matplotlib.pyplot as plt
# # Frequency range (log-spaced for clarity)
# f = np.logspace(-5, -1, 2000)  # Hz
# Z = transfer_function(f)
#
# plt.figure()
# plt.plot(f, Z)
# plt.xscale("log")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Transfer Function Z(f)")
# plt.grid()
# plt.title("Analytical Transfer Function (Range Acc → LGD)")
# plt.show()

# def compute_lgd(drho_dotdot_array: np.ndarray, time_array=None) -> np.ndarray:
#     """
#     Simplified estimation of residual LGD (Line-of-Sight Gravity Difference) or dg_AB^LOS.
#     This is a frequency domain estimate, meaning that the LGD residual is approximated via the frequency domain:
#     dg_AB^LOS(t) ~ F^-1{Z(f)*F{drho_dotdot(t)}},
#     where Z(f) is the transfer function and F{} is the Fourier Transform.
#     :param drho_dotdot_array: ndarray
#         derivative of reduced range-rate observable array.
#     :return: ndarray
#         array of LGD residuals (unit of drho_dotdot)
#     """
#     if isinstance(time_array, (np.ndarray, np.generic, pd.DataFrame)):
#         tol = 20  # s
#         list_of_pieces, list_of_tpieces = [], []
#         # gets the indices of the gaps
#         pieces_indices, _ = get_time_pieces(time_array, tol=tol)
#         start_indices, end_indices = pieces_indices
#         # separates the y_array into pieces, defined by tolerance requirement
#         y_array_pieces = [drho_dotdot_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
#         t_array_pieces = [time_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
#
#         ind_piece = 0
#         for array, t_array in zip(y_array_pieces, t_array_pieces):
#             # real Fourier transform
#             fourier_drho_dot_dot = np.fft.rfft(array, len(array))
#             n = fourier_drho_dot_dot.size
#             # sampling time
#             dt = 5.0  # s
#             # frequency array
#             freq_array = np.fft.rfftfreq(len(array), dt)
#             # Transfer function
#             Z_of_f = transfer_function(freq_array)
#             # estimate LGD residual
#             dg_AB_LOS_piece = np.fft.irfft(Z_of_f * fourier_drho_dot_dot, len(array))
#             # in the case that np.fft.irfft removes an element, simply append the last one of the original array.
#             if len(array) != len(dg_AB_LOS_piece):
#                 dg_AB_LOS_c_piece = np.append(dg_AB_LOS_piece, array[-1])
#             else:
#                 dg_AB_LOS_c_piece = dg_AB_LOS_piece
#             # add piece of array to the to-be concanated list
#             list_of_pieces.append(dg_AB_LOS_c_piece), list_of_tpieces.append(t_array)
#             ind_piece += 1
#
#         # concatenate final list with all array pieces
#         dg_AB_LOS_c = np.concatenate(list_of_pieces)
#         t_array_final = np.concatenate(list_of_tpieces)
#
#     if time_array is None:
#         # real Fourier transform
#         fourier_drho_dot_dot = np.fft.rfft(drho_dotdot_array, len(drho_dotdot_array))
#         n = fourier_drho_dot_dot.size
#         # sampling time
#         dt = 5.0  # s
#         # frequency array
#         freq_array = np.fft.rfftfreq(len(drho_dotdot_array), dt)
#         # Transfer function
#         Z_of_f = transfer_function(freq_array)
#         # estimate LGD residual
#         dg_AB_LOS = np.fft.irfft(Z_of_f*fourier_drho_dot_dot, len(drho_dotdot_array))
#         # in the case that np.fft.irfft removes an element, simply append the last one of the original array.
#         if len(drho_dotdot_array) != len(dg_AB_LOS): dg_AB_LOS_c = np.append(dg_AB_LOS, drho_dotdot_array[-1])
#         else: dg_AB_LOS_c = dg_AB_LOS
#
#     return dg_AB_LOS_c


def compute_lgd(
    drho_dotdot_array: np.ndarray,
    time_array=None,
    *,
    tol: float = 20.0,
    dt: float | None = None,
    detrend_mean: bool = True,
) -> np.ndarray:
    """
    Frequency-domain estimate of residual LGD (dg_AB^LOS) from range-acceleration residuals.

        dg_AB^LOS(t) ≈ F^{-1}{ Z(f) * F{drho_ddot(t)} }

    Key improvements over your original:
      - Vectorized / no repeated code paths.
      - dt is NOT hard-coded to 5 s:
          * if dt is provided -> use it
          * else if time_array provided -> infer dt per piece (median)
          * else -> fall back to 5 s (for backward compatibility)
      - Handles DC explicitly (Z(0)=0 already), optional mean removal to reduce edge ringing.
      - Uses gap splitting when time_array is provided.

    Parameters
    ----------
    drho_dotdot_array : ndarray
        Range-acceleration residual time series.
    time_array : ndarray or None
        Timestamps in seconds. If provided, splits at gaps >= tol and can infer dt.
    tol : float
        Gap threshold in seconds for piece splitting.
    dt : float or None
        Sampling interval in seconds. If given, overrides inferred dt.
    detrend_mean : bool
        If True, subtract mean per piece before FFT and add it back after IFFT (reduces leakage).

    Returns
    -------
    ndarray
        LGD estimate on the (possibly piecewise) original time support.
        If time_array is provided, output length matches input length.
    """
    y = np.asarray(drho_dotdot_array, dtype=np.float64)

    # Split into pieces if we have time_array
    if time_array is not None and isinstance(time_array, (np.ndarray, np.generic, pd.Series, pd.DataFrame)):
        t = np.asarray(time_array).squeeze()
        if isinstance(time_array, pd.DataFrame):
            # If someone passed a single-column DataFrame
            if t.ndim != 1:
                t = np.asarray(time_array.iloc[:, 0]).squeeze()

        pieces_indices, _ = get_time_pieces(t, tol=tol)
        start_indices, end_indices = pieces_indices

        y_pieces = [y[s:e + 1] for s, e in zip(start_indices, end_indices)]
        t_pieces = [t[s:e + 1] for s, e in zip(start_indices, end_indices)]

        out_pieces = []
        for y_piece, t_piece in zip(y_pieces, t_pieces):
            if y_piece.size == 0:
                continue
            if y_piece.size == 1:
                out_pieces.append(np.zeros_like(y_piece, dtype=np.float64))
                continue

            # Determine dt for this piece
            if dt is not None:
                dt_use = float(dt)
            else:
                dt_use = _robust_dt_from_x(t_piece)
                if not np.isfinite(dt_use) or dt_use <= 0:
                    dt_use = 5.0  # backward-compatible fallback

            # Mean removal (optional, helps reduce ringing)
            mean_piece = float(np.mean(y_piece)) if detrend_mean else 0.0
            y0 = y_piece - mean_piece

            # FFT pipeline
            F = np.fft.rfft(y0, n=y0.size)
            f = np.fft.rfftfreq(y0.size, d=dt_use)
            Z = transfer_function(f)
            x = np.fft.irfft(Z * F, n=y0.size)

            if detrend_mean:
                x = x + mean_piece  # keep DC behavior consistent with original signal support

            out_pieces.append(x)

        if len(out_pieces) == 0:
            return np.array([], dtype=np.float64)

        return np.concatenate(out_pieces)

    # No time_array: single piece
    if y.size == 0:
        return np.array([], dtype=np.float64)
    if y.size == 1:
        return np.zeros_like(y, dtype=np.float64)

    dt_use = float(dt) if dt is not None else 5.0  # keep your legacy default
    mean_all = float(np.mean(y)) if detrend_mean else 0.0
    y0 = y - mean_all

    F = np.fft.rfft(y0, n=y0.size)
    f = np.fft.rfftfreq(y0.size, d=dt_use)
    Z = transfer_function(f)
    x = np.fft.irfft(Z * F, n=y0.size)

    if detrend_mean:
        x = x + mean_all

    return x


def perform_filtering_sequence(
    array_to_be_filtered: np.ndarray,
    t_array=None,
    methods=None,
    Ns=None,
    btypes=None,
    cutoffs=None,
    fs=0.2,
    n=4,
    int_flag=True,
    id_flag=True,
    tol=20,
    dx_interp=5,          # NEW: make the grid step explicit if you want interpolation
    lfn_h=450e3,
    lfn_mu=3.986004418e14,
    lfn_Re=6371e3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a filtering sequence in the order given by `methods`.

    Supported methods:
      - 'butw'         : Butterworth SOS zero-phase filter
      - 'diff'         : central finite-difference derivative
      - 'lfn_per_orbit': low-frequency noise removal per orbit

    Notes:
      - `t_array` is always updated and passed forward.
      - If interpolation is enabled, keep it consistent across steps.
    """
    if methods is None:
        methods = ['butw']
    if btypes is None:
        btypes = ['low']
    if Ns is None:
        Ns = [1]
    if cutoffs is None:
        cutoffs = [1e-2]

    # Current working variables
    y_curr = np.asarray(array_to_be_filtered)
    t_curr = None if t_array is None else np.asarray(t_array)

    # Validate Butterworth parameter lengths
    count_butw = methods.count('butw')
    if count_butw:
        if len(Ns) != count_butw:
            raise ValueError(f"Expected {count_butw} entries in Ns, got {len(Ns)}.")
        if len(btypes) != count_butw:
            raise ValueError(f"Expected {count_butw} entries in btypes, got {len(btypes)}.")
        if len(cutoffs) != count_butw:
            raise ValueError(f"Expected {count_butw} entries in cutoffs, got {len(cutoffs)}.")

    i_butw = 0

    for method in methods:
        if method == 'butw':
            # IMPORTANT: filter_array must honor optional['fs'], optional['int_flag'], optional['tol'], optional['dx']
            y_curr, t_curr = filter_array(
                y_curr,
                t_array=t_curr,
                method='butw',
                N=Ns[i_butw],
                btype=btypes[i_butw],
                cutoff=cutoffs[i_butw],
                fs=fs,
                int_flag=int_flag,
                tol=tol,
                dx_interp=dx_interp
            )
            i_butw += 1

        elif method == 'diff':
            y_curr, t_curr = differentiate_central_scheme(
                y_curr,
                x_array=t_curr,
                n=n,
                fs=fs,
                id_gaps=id_flag,
                tol=tol,
                int_flag=int_flag,
                dx_interp=dx_interp
            )

        elif method == 'lfn_per_orbit':
            if t_curr is None:
                raise ValueError("lfn_per_orbit requires t_array (time axis).")
            y_curr, t_curr = filter_array_with_noise_model(
                y_curr,
                t_curr,
                h=lfn_h,
                mu=lfn_mu,
                Re=lfn_Re
            )

        else:
            raise ValueError(f"Unknown filtering method '{method}'")

    return y_curr, t_curr


# def perform_filtering_sequence(array_to_be_filtered: np.ndarray, t_array=None, methods=None,
#                                Ns=None, btypes=None, cutoffs=None, fs=0.2, n=4,
#                                int_flag=True, id_flag=True, tol=20,
#                                lfn_h=450e3, lfn_mu=3.986004418e14, lfn_Re=6371e3) -> tuple[np.ndarray, np.array]:
#     """
#     Perform a filtering sequence (order is set by the 'methods' list input) of own choosing.
#
#     New: Supports low-frequency noise removal per orbit ('lfn_per_orbit').
#
#     :param array_to_be_filtered: ndarray
#         Data array to be filtered.
#     :param t_array: ndarray
#         Time or x-array associated to data array.
#     :param methods: list
#         Methods in sequential order to be used in filtering.
#         Supported: 'butw', 'diff', 'lfn_per_orbit'.
#     :param Ns: list
#         Orders of butterworth filter (for 'butw').
#     :param btypes: list
#         Types of butterworth filters.
#     :param cutoffs: list
#         Cutoff frequencies for butterworth filters.
#     :param lfn_h: float
#         Satellite altitude for low-frequency noise model (m).
#     :param lfn_mu: float
#         Gravitational parameter for low-frequency noise model.
#     :param lfn_Re: float
#         Earth radius for low-frequency noise model.
#     :return:
#         Tuple (filtered_array, t_array)
#     """
#     if btypes is None: btypes = ['low']
#     if Ns is None: Ns = [1]
#     if methods is None: methods = ['butw']
#     if cutoffs is None: cutoffs = [1e-2]
#
#     x_array_filtered = t_array
#     count_int = 0
#
#     count_butw = methods.count('butw')
#     if count_butw != 0 and (count_butw != len(Ns) or count_butw != len(btypes)):
#         raise ValueError("Butterworth filters require matching Ns and btypes lengths.")
#
#     i_butw = 0
#     for method in methods:
#         if method == 'butw':
#             array_to_be_filtered, x_array_filtered = filter_array(
#                 array_to_be_filtered, t_array=t_array, method='butw',
#                 N=Ns[i_butw], btype=btypes[i_butw], cutoff=cutoffs[i_butw], fs=fs
#             )
#             i_butw += 1
#
#         elif method == 'diff':
#             array_to_be_filtered, x_array_filtered = differentiate_central_scheme(
#                 array_to_be_filtered, x_array=t_array, fs=fs, n=n,
#                 int_flag=int_flag, id_gaps=id_flag, tol=tol
#             )
#
#         elif method == 'lfn_per_orbit':
#             # Call the dedicated per-orbit low-frequency noise removal
#             array_to_be_filtered, x_array_filtered = filter_array_with_noise_model(
#                 array_to_be_filtered, x_array_filtered,
#                 h=lfn_h, mu=lfn_mu, Re=lfn_Re
#             )
#
#         else:
#             raise ValueError(f"Unknown filtering method '{method}'")
#
#         # Update time array for next step
#         t_array = x_array_filtered
#         count_int += 1
#
#     return array_to_be_filtered, x_array_filtered

########################################################################################################################
# ORDER, CUT-OFF and FILTERING SEQUENCE
N_order, fc, filt_seq = 4, 11*1e-3, ['butw', 'diff']  # [-], [Hz], [first filter, second filter]
fc_low = 0.9e-3  # low frequency removal
freq_cutoffs, filter_btypes, Norders = [fc], ['low'], [N_order]
freq_cutoffs, filter_btypes, Norders, filt_seq = ([fc_low, fc], ['high', 'low'],
                                                  [N_order, N_order], ['butw', 'butw', 'diff'])
filter_options = [filt_seq, freq_cutoffs, Norders, filter_btypes]
# DIFFERENTIATION METHOD
# differentiation function (see dict_of_diff)
method_str = 'central_diff'  # here central_diff is chosen
# if central_diff is chosen, provide accuracy order n for edges, 2*n for central
n = 4  # here chosen to be 4 for edges, 8 for central

# DICTIONARY CONTAINING ALL DIFFERENTIATION METHODS
dict_of_diff = {'central_diff': {"function": differentiate_central_scheme,
                                 'inputs': {'n': n, 'fs': 0.2, 'int_flag': True, 'tol': 20}}}
# method to be used for differentiation
method, inputs = dict_of_diff[method_str]['function'], dict_of_diff[method_str]['inputs']

def perform_dataframe_filt_diff_comb(
    df: pd.DataFrame,
    filter_options,
    filter_options_pref_geo=None,
    inputs=None,
    add_columns=None,
    bands_bool=True,
    *,
    hp_new: float = 0.7e-3,   # v2 high-pass cutoff in Hz
    hp_old: float = 0.9e-3,   # kept for readability / back-compat semantics
) -> pd.DataFrame:
    """
    Run filtering+differentiation pipelines on a dataframe containing (at least) columns:
      - timestamp (seconds)
      - postfit
      - (optional) up_global
      - latitude_A, longitude_A, latitude_B, longitude_B
      - altitude_A, altitude_B   (km)

    Notes:
      - Uses `perform_filtering_sequence(...)` for the pipeline.
      - Ensures v2 HP cutoff replacement is correctly aligned to BUTTERWORTH STEPS ONLY.
      - Aligns outputs to a reference time grid by comparing timestamps, not just lengths.
      - Diff-only runs do not pass irrelevant butterworth btypes/cutoffs.

    Guaranteed output columns (when available):
      - postfit_acc, postfit_acc_unf, postfit_acc_v2
      - up_global_acc, up_global_acc_unf, up_global_acc_v2
      - up_global_filt
      - latitude_*, longitude_*, altitude_*, latitude_mid, longitude_mid, altitude_mid
    """

    # -------------------------
    # Defensive: sort by timestamp (important for interpolation)
    # -------------------------
    if "timestamp" not in df.columns:
        raise ValueError("Input DataFrame must contain column 'timestamp'.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    if filter_options_pref_geo is None:
        filter_options_pref_geo = filter_options

    # -------------------------
    # Default inputs (robust; no reliance on external 'n')
    # -------------------------
    if inputs is None:
        inputs = {}

    # Normalize inputs with safe defaults
    inputs = {
        "n": int(inputs.get("n", 4)),
        "fs": float(inputs.get("fs", 0.2)),
        "int_flag": bool(inputs.get("int_flag", True)),
        "tol": float(inputs.get("tol", 20)),
        "id_flag": bool(inputs.get("id_flag", True)),
        # optionally let user pass dx_interp downstream if your perform_filtering_sequence supports it
        **({} if "dx_interp" not in inputs else {"dx_interp": inputs["dx_interp"]}),
    }

    filt_seq, freq_cutoffs, Norders, filter_btypes = filter_options
    filt_seq_pg, freq_cutoffs_pg, Norders_pg, filter_btypes_pg = filter_options_pref_geo

    # -------------------------
    # Helpers
    # -------------------------
    def _as_array(x):
        return np.asarray(x)

    def _run_seq(series, ts, methods, Ns, btypes, cutoffs):
        return perform_filtering_sequence(
            _as_array(series),
            _as_array(ts) if ts is not None else None,
            methods=methods,
            Ns=Ns,
            btypes=btypes,
            cutoffs=cutoffs,
            **inputs,
        )

    def _align(y, t_y, t_ref):
        """
        Align y(t_y) onto t_ref. Don't trust length-only checks.
        If timestamps match exactly, pass through.
        Otherwise interpolate to size.
        """
        y = _as_array(y)
        t_y = _as_array(t_y)
        t_ref = _as_array(t_ref)

        if (y.size == t_ref.size) and (t_y.size == t_ref.size) and np.array_equal(t_y, t_ref):
            return y
        return _as_array(interpolate_array_to_size(y, t_y, t_ref))

    def _interp_from_df(col, t_ref):
        return _as_array(interpolate_array_to_size(_as_array(df[col]), _as_array(df["timestamp"]), _as_array(t_ref)))

    def _make_v2_cutoffs(methods, btypes, cutoffs, hp_new, hp_old=hp_old):
        """
        Replace Butterworth high-pass cutoff with hp_new for steps where:
          method == 'butw' and btype startswith 'high'

        IMPORTANT:
          - btypes and cutoffs are indexed by BUTTERWORTH STEP (not method index).
          - Therefore, iterate through methods and increment a separate butter index.
        """
        out = list(cutoffs)
        j = 0  # butterworth-step index
        for m in methods:
            if m == "butw":
                bt = str(btypes[j]).lower()
                if bt.startswith("high"):
                    out[j] = hp_new
                j += 1
        return out

    # -------------------------
    # 1) POSTFIT pipelines (defines reference time grid)
    # -------------------------
    if "postfit" not in df.columns:
        raise ValueError("Input DataFrame must contain column 'postfit'.")

    postfit_acc_raw, time_array = _run_seq(
        df["postfit"], df["timestamp"],
        methods=filt_seq, Ns=Norders, btypes=filter_btypes, cutoffs=freq_cutoffs
    )
    time_array = _as_array(time_array)
    postfit_acc = _as_array(postfit_acc_raw)

    # diff-only postfit (no irrelevant btypes/cutoffs)
    postfit_acc_unf_raw, t_pf_unf = _run_seq(
        df["postfit"], df["timestamp"],
        methods=["diff"], Ns=[], btypes=[], cutoffs=[]
    )
    postfit_acc_unf = _align(postfit_acc_unf_raw, t_pf_unf, time_array)

    # v2 cutoffs (only HP changes; correctly aligned to butterworth steps)
    freq_cutoffs_v2 = _make_v2_cutoffs(
        filt_seq, filter_btypes, freq_cutoffs, hp_new=hp_new, hp_old=hp_old
    )

    postfit_acc_v2_raw, t_pf_v2 = _run_seq(
        df["postfit"], df["timestamp"],
        methods=filt_seq, Ns=Norders, btypes=filter_btypes, cutoffs=freq_cutoffs_v2
    )
    postfit_acc_v2 = _align(postfit_acc_v2_raw, t_pf_v2, time_array)

    # -------------------------
    # 2) UP_GLOBAL pipelines (optional)
    # -------------------------
    up_global_bool = "up_global" in df.columns
    nan_vec = np.full(len(time_array), np.nan, dtype=float)

    up_global_acc = nan_vec.copy()
    up_global_acc_unf = nan_vec.copy()
    up_global_filt = nan_vec.copy()
    up_global_acc_v2 = nan_vec.copy()

    wavelet_results = {}

    if up_global_bool:
        # v1 geo filtered
        ug_acc_raw, t_ug = _run_seq(
            df["up_global"], df["timestamp"],
            methods=filt_seq_pg, Ns=Norders_pg, btypes=filter_btypes_pg, cutoffs=freq_cutoffs_pg
        )
        up_global_acc = _align(ug_acc_raw, t_ug, time_array)

        # v1 geo diff-only
        ug_acc_unf_raw, t_ug_unf = _run_seq(
            df["up_global"], df["timestamp"],
            methods=["diff"], Ns=[], btypes=[], cutoffs=[]
        )
        up_global_acc_unf = _align(ug_acc_unf_raw, t_ug_unf, time_array)

        # v1 undiff/low-pass rate
        ug_filt_raw, t_ug_filt = _run_seq(
            df["up_global"], df["timestamp"],
            methods=["butw", "lfn_per_orbit"],
            Ns=[4],
            btypes=["low"],
            cutoffs=[11.25e-3],
        )
        up_global_filt = _align(ug_filt_raw, t_ug_filt, time_array)

        # v2: only HP changes (aligned to butterworth steps)
        freq_cutoffs_pg_v2 = _make_v2_cutoffs(
            filt_seq_pg, filter_btypes_pg, freq_cutoffs_pg, hp_new=hp_new, hp_old=hp_old
        )
        ug_acc_v2_raw, t_ug_v2 = _run_seq(
            df["up_global"], df["timestamp"],
            methods=filt_seq_pg, Ns=Norders_pg, btypes=filter_btypes_pg, cutoffs=freq_cutoffs_pg_v2
        )
        up_global_acc_v2 = _align(ug_acc_v2_raw, t_ug_v2, time_array)

        # Optional wavelet-like bands
        if bands_bool:
            bands = [
                ("long",   0.25e-3, 0.49e-3),
                ("049098", 0.49e-3, 0.98e-3),
                ("098195", 0.98e-3, 1.95e-3),
                ("19539",  1.95e-3, 3.9e-3),
                ("3978",   3.9e-3,  7.8e-3),
                ("7812",   7.8e-3, 12e-3),
                ("78156",  7.8e-3, 15.6e-3),
                ("3125",  15.6e-3, 31.25e-3),
                ("625",   31.25e-3, 62.5e-3),
                ("old",    0.9e-3, 11e-3),
                ("",       0.6e-3, 12e-3),
            ]

            if isinstance(Norders_pg, (list, tuple, np.ndarray)) and len(Norders_pg) > 0:
                N_order_local = int(Norders_pg[0])
            else:
                N_order_local = 4

            # Reuse ug_acc_unf_raw and its timebase for bands
            for label, low_cut, high_cut in bands:
                y_raw, t_mid = _run_seq(
                    ug_acc_unf_raw, t_ug_unf,
                    methods=["butw", "butw"],
                    Ns=[N_order_local, N_order_local],
                    btypes=["high", "low"],
                    cutoffs=[low_cut, high_cut],
                )
                wavelet_results[f"up_global_acc_{label}"] = _align(y_raw, t_mid, time_array)

    # -------------------------
    # 3) Coordinates + midpoint geometry
    # -------------------------
    required_coords = [
        "latitude_A", "longitude_A", "latitude_B", "longitude_B",
        "altitude_A", "altitude_B"
    ]
    missing = [c for c in required_coords if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")

    lat_A = _interp_from_df("latitude_A", time_array)
    lon_A = _interp_from_df("longitude_A", time_array)
    lat_B = _interp_from_df("latitude_B", time_array)
    lon_B = _interp_from_df("longitude_B", time_array)

    alt_A = _interp_from_df("altitude_A", time_array)
    alt_B = _interp_from_df("altitude_B", time_array)

    # NOTE: expects 'pm' (pymap3d) in scope, as in your original code.
    rA = np.array(pm.geodetic2ecef(lat_A, lon_A, alt_A * 1e3))
    rB = np.array(pm.geodetic2ecef(lat_B, lon_B, alt_B * 1e3))
    rA = np.vstack(rA).T
    rB = np.vstack(rB).T

    r_mean = (rA + rB) / 2.0
    scale = (np.linalg.norm(rA, axis=1) + np.linalg.norm(rB, axis=1)) / (2.0 * np.linalg.norm(r_mean, axis=1))
    r_corrected = (r_mean.T * scale).T

    lat_mid, lon_mid, alt_mid = pm.ecef2geodetic(r_corrected[:, 0], r_corrected[:, 1], r_corrected[:, 2])
    alt_mid = alt_mid / 1e3

    # -------------------------
    # 4) Build output dataframe
    # -------------------------
    out = {
        "timestamp": _as_array(time_array),

        "latitude_A": lat_A, "longitude_A": lon_A,
        "latitude_B": lat_B, "longitude_B": lon_B,
        "altitude_A": alt_A, "altitude_B": alt_B,

        "latitude_mid": _as_array(lat_mid),
        "longitude_mid": _as_array(lon_mid),
        "altitude_mid": _as_array(alt_mid),

        "postfit_acc": postfit_acc,
        "postfit_acc_unf": postfit_acc_unf,
        "postfit_acc_v2": postfit_acc_v2,

        "up_global_filt": up_global_filt,
        "up_global_acc": up_global_acc,
        "up_global_acc_unf": up_global_acc_unf,
        "up_global_acc_v2": up_global_acc_v2,
    }

    if bands_bool and wavelet_results:
        out.update(wavelet_results)

    df_updated = pd.DataFrame(out)

    # -------------------------
    # 5) Optional extra columns
    #     Robust: only interpolate original df columns.
    #     If user requests a computed column (e.g. "postfit_acc"), skip silently.
    # -------------------------
    if add_columns is not None:
        for col in add_columns:
            if col in df_updated.columns:
                continue
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame")
            arr = _interp_from_df(col, time_array)
            if col == "adtrack_A":
                arr = arr.astype(int)
            df_updated[col] = arr

    return df_updated


# def perform_dataframe_filt_diff_comb(
#     df: pd.DataFrame,
#     filter_options,
#     filter_options_pref_geo=None,
#     inputs=None,
#     add_columns=None,
#     bands_bool=True,
#     *,
#     hp_new: float = 0.7e-3,   # <-- NEW: v2 high-pass cutoff in Hz
#     hp_old: float = 0.9e-3,   # <-- optional: what you consider "old" (for clarity/logical grouping)
# ) -> pd.DataFrame:
#     """
#     ... (docstring omitted for brevity; keep your existing one) ...
#
#     Guaranteed output columns:
#       - postfit_acc, postfit_acc_unf, postfit_acc_v2, postfit_acc_unf_v2
#       - up_global_acc, up_global_acc_unf, up_global_acc_v2, up_global_acc_unf_v2
#       - up_global_filt
#     """
#
#     if filter_options_pref_geo is None:
#         filter_options_pref_geo = filter_options
#
#     if inputs is None:
#         inputs = {"n": n, "fs": 0.2, "int_flag": True, "tol": 20}
#
#     filt_seq, freq_cutoffs, Norders, filter_btypes = filter_options
#     filt_seq_pg, freq_cutoffs_pg, Norders_pg, filter_btypes_pg = filter_options_pref_geo
#
#     # -------------------------
#     # Helpers
#     # -------------------------
#     def _as_array(x):
#         return np.asarray(x)
#
#     def _run_seq(series, ts, methods, Ns, btypes, cutoffs):
#         return perform_filtering_sequence(
#             series, ts,
#             methods=methods, Ns=Ns, btypes=btypes, cutoffs=cutoffs,
#             **inputs,
#         )
#
#     def _align(y, t_y, t_ref):
#         if len(t_y) == len(t_ref):
#             return _as_array(y)
#         return _as_array(interpolate_array_to_size(y, t_y, t_ref))
#
#     def _interp_from_df(col, t_ref):
#         return _as_array(interpolate_array_to_size(df[col], df["timestamp"], t_ref))
#
#     def _make_v2_cutoffs(methods, btypes, cutoffs, hp_new, hp_old=hp_old):
#         """
#         Replace Butterworth high-pass cutoff with hp_new for steps where:
#           method == 'butw' and btype startswith 'high'
#         We don't need to inspect hp_old; it's kept for readability/back-compat.
#         """
#         out = list(cutoffs)
#         for i, (m, bt) in enumerate(zip(methods, btypes)):
#             if m == "butw" and str(bt).lower().startswith("high"):
#                 out[i] = hp_new
#         return out
#
#     # -------------------------
#     # 1) POSTFIT pipelines (time_array defined here)
#     # -------------------------
#     postfit_acc_raw, time_array = _run_seq(
#         df["postfit"], df["timestamp"],
#         methods=filt_seq, Ns=Norders, btypes=filter_btypes, cutoffs=freq_cutoffs
#     )
#     postfit_acc = _as_array(postfit_acc_raw)
#
#     # diff-only postfit
#     postfit_acc_unf_raw, t_pf_unf = _run_seq(
#         df["postfit"], df["timestamp"],
#         methods=["diff"], Ns=None, btypes=filter_btypes, cutoffs=freq_cutoffs
#     )
#     postfit_acc_unf = _align(postfit_acc_unf_raw, t_pf_unf, time_array)
#
#     # v2 cutoffs (only HP changes)
#     freq_cutoffs_v2 = _make_v2_cutoffs(filt_seq, filter_btypes, freq_cutoffs, hp_new=hp_new, hp_old=hp_old)
#
#     postfit_acc_v2_raw, t_pf_v2 = _run_seq(
#         df["postfit"], df["timestamp"],
#         methods=filt_seq, Ns=Norders, btypes=filter_btypes, cutoffs=freq_cutoffs_v2
#     )
#     postfit_acc_v2 = _align(postfit_acc_v2_raw, t_pf_v2, time_array)
#
#     # -------------------------
#     # 2) UP_GLOBAL pipelines
#     # -------------------------
#     up_global_bool = "up_global" in df.columns
#     nan_vec = np.full(len(time_array), np.nan, dtype=float)
#
#     up_global_acc = nan_vec.copy()
#     up_global_acc_unf = nan_vec.copy()
#     up_global_filt = nan_vec.copy()
#     up_global_acc_v2 = nan_vec.copy()
#     up_global_acc_unf_v2 = nan_vec.copy()
#
#     wavelet_results = {}
#
#     if up_global_bool:
#         # v1 geo filtered
#         ug_acc_raw, t_ug = _run_seq(
#             df["up_global"], df["timestamp"],
#             methods=filt_seq_pg, Ns=Norders_pg, btypes=filter_btypes_pg, cutoffs=freq_cutoffs_pg
#         )
#         up_global_acc = _align(ug_acc_raw, t_ug, time_array)
#
#         # v1 geo diff-only
#         ug_acc_unf_raw, t_ug_unf = _run_seq(
#             df["up_global"], df["timestamp"],
#             methods=["diff"], Ns=None, btypes=filter_btypes_pg, cutoffs=freq_cutoffs_pg
#         )
#         up_global_acc_unf = _align(ug_acc_unf_raw, t_ug_unf, time_array)
#
#         # v1 undiff/low-pass rate
#         ug_filt_raw, t_ug_filt = _run_seq(
#             df["up_global"], df["timestamp"],
#             methods=["butw", "lfn_per_orbit"], Ns=[4], btypes=["low"], cutoffs=[11.25e-3]
#         )
#         up_global_filt = _align(ug_filt_raw, t_ug_filt, time_array)
#
#         # v2: only HP changes
#         freq_cutoffs_pg_v2 = _make_v2_cutoffs(
#             filt_seq_pg, filter_btypes_pg, freq_cutoffs_pg, hp_new=hp_new, hp_old=hp_old
#         )
#
#         ug_acc_v2_raw, t_ug_v2 = _run_seq(
#             df["up_global"], df["timestamp"],
#             methods=filt_seq_pg, Ns=Norders_pg, btypes=filter_btypes_pg, cutoffs=freq_cutoffs_pg_v2
#         )
#         up_global_acc_v2 = _align(ug_acc_v2_raw, t_ug_v2, time_array)
#
#         # ug_acc_unf_v2_raw, t_ug_unf_v2 = _run_seq(
#         #     df["up_global"], df["timestamp"],
#         #     methods=["diff"], Ns=None, btypes=filter_btypes_pg, cutoffs=freq_cutoffs_pg_v2
#         # )
#         # up_global_acc_unf_v2 = _align(ug_acc_unf_v2_raw, t_ug_unf_v2, time_array)
#
#         # Optional wavelet-like bands (heavy)
#         if bands_bool:
#             bands = [
#                 ("long",   0.25e-3, 0.49e-3),
#                 ("049098", 0.49e-3, 0.98e-3),
#                 ("098195", 0.98e-3, 1.95e-3),
#                 ("19539",  1.95e-3, 3.9e-3),
#                 ("3978",   3.9e-3,  7.8e-3),
#                 ("7812",   7.8e-3, 12e-3),
#                 ("78156",  7.8e-3, 15.6e-3),
#                 ("3125",  15.6e-3, 31.25e-3),
#                 ("625",   31.25e-3, 62.5e-3),
#                 ("old",    0.9e-3, 11e-3),
#                 ("",       0.6e-3, 12e-3),
#             ]
#
#             if isinstance(Norders_pg, (list, tuple, np.ndarray)) and len(Norders_pg) > 0:
#                 N_order_local = int(Norders_pg[0])
#             else:
#                 N_order_local = 4
#
#             for label, low_cut, high_cut in bands:
#                 y_raw, t_mid = _run_seq(
#                     ug_acc_unf_raw, t_ug_unf,
#                     methods=["butw", "butw"],
#                     Ns=[N_order_local, N_order_local],
#                     btypes=["high", "low"],
#                     cutoffs=[low_cut, high_cut],
#                 )
#                 wavelet_results[f"up_global_acc_{label}"] = _align(y_raw, t_mid, time_array)
#
#     # -------------------------
#     # 3) Coordinates + midpoint geometry (unchanged)
#     # -------------------------
#     lat_A = _interp_from_df("latitude_A", time_array)
#     lon_A = _interp_from_df("longitude_A", time_array)
#     lat_B = _interp_from_df("latitude_B", time_array)
#     lon_B = _interp_from_df("longitude_B", time_array)
#
#     alt_A = _interp_from_df("altitude_A", time_array)
#     alt_B = _interp_from_df("altitude_B", time_array)
#
#     rA = np.array(pm.geodetic2ecef(lat_A, lon_A, alt_A * 1e3))
#     rB = np.array(pm.geodetic2ecef(lat_B, lon_B, alt_B * 1e3))
#     rA = np.vstack(rA).T
#     rB = np.vstack(rB).T
#
#     r_mean = (rA + rB) / 2.0
#     scale = (np.linalg.norm(rA, axis=1) + np.linalg.norm(rB, axis=1)) / (2.0 * np.linalg.norm(r_mean, axis=1))
#     r_corrected = (r_mean.T * scale).T
#
#     lat_mid, lon_mid, alt_mid = pm.ecef2geodetic(r_corrected[:, 0], r_corrected[:, 1], r_corrected[:, 2])
#     alt_mid = alt_mid / 1e3
#
#     # -------------------------
#     # 4) Build output dataframe
#     # -------------------------
#     out = {
#         "timestamp": _as_array(time_array),
#
#         "latitude_A": lat_A, "longitude_A": lon_A,
#         "latitude_B": lat_B, "longitude_B": lon_B,
#         "altitude_A": alt_A, "altitude_B": alt_B,
#
#         "latitude_mid": _as_array(lat_mid),
#         "longitude_mid": _as_array(lon_mid),
#         "altitude_mid": _as_array(alt_mid),
#
#         "postfit_acc": postfit_acc,
#         "postfit_acc_unf": postfit_acc_unf,
#         "postfit_acc_v2": postfit_acc_v2,
#         # "postfit_acc_unf_v2": postfit_acc_unf_v2,
#
#         "up_global_filt": up_global_filt,
#         "up_global_acc": up_global_acc,
#         "up_global_acc_unf": up_global_acc_unf,
#         "up_global_acc_v2": up_global_acc_v2,
#         # "up_global_acc_unf_v2": up_global_acc_unf_v2,
#     }
#
#     if bands_bool and wavelet_results:
#         out.update(wavelet_results)
#
#     df_updated = pd.DataFrame(out)
#
#     # -------------------------
#     # 5) Optional extra columns
#     #     Robust: only interpolate original df columns.
#     #     If user requests a computed column (e.g. "postfit_acc"), skip silently.
#     # -------------------------
#     if add_columns is not None:
#         for col in add_columns:
#             if col in df_updated.columns:
#                 continue
#             if col not in df.columns:
#                 raise ValueError(f"Column '{col}' not found in input DataFrame")
#             arr = _interp_from_df(col, time_array)
#             if col == "adtrack_A":
#                 arr = arr.astype(int)
#             df_updated[col] = arr
#
#     return df_updated