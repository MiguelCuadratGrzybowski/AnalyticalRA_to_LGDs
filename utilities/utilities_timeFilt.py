import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, savgol_filter

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


def transfer_function(frequency_array):
    """
    Function provides the analytical transfer function from range-acceleration residuals to LGD, corrected by using
    source code associated to the following article:
    Ghobadi-Far, K., Han, S.-C., Weller, S., Loomis, B. D., Luthcke, S. B., Mayer-Gürr, T., & Behzadpour, S. (2018).
    A Transfer Function Between Line-of-Sight Gravity Difference and GRACE Intersatellite Ranging Data and
    an Application to Hydrological Surface Mass Variation.
    Journal of Geophysical Research: Solid Earth, 123(10), 9186–9201. https://doi.org/10.1029/2018JB016088
    :param frequency_array: float or ndarray
        Frequency (array) in Hz.
    :return: float or ndarray
        Piecewise transfer function: Z(f) = 1 + 3.45e-4 f^-1.04 for f >= 0.7 mHz;
        Z(f) ~ a*log10(f)+b for 0.3 mHz <= f < 0.7 mHz; Z(f) = 0 for f < 0.3 mHz.
        Returns a float for a scalar input, an ndarray otherwise.
    """
    f = np.asarray(frequency_array, dtype=float)
    scalar_input = f.ndim == 0
    f = np.atleast_1d(f)

    f_trunc, f_low = 0.3e-3, 0.7e-3
    Z_low = 1 + 3.45e-4 * f_low ** (-1.04)
    a_slope = (0 - Z_low) / (np.log10(f_trunc) - np.log10(f_low))
    b_val = 0 - a_slope * np.log10(f_trunc)

    Z_array = np.zeros_like(f)
    m_ramp = (f >= f_trunc) & (f < f_low)
    m_high = f >= f_low
    Z_array[m_ramp] = a_slope * np.log10(f[m_ramp]) + b_val
    Z_array[m_high] = 1 + 3.45e-4 * f[m_high] ** (-1.04)

    return float(Z_array[0]) if scalar_input else Z_array


def compute_lgd(drho_dotdot_array: np.ndarray, time_array=None) -> np.ndarray:
    """
    Simplified estimation of residual LGD (Line-of-Sight Gravity Difference) or dg_AB^LOS.
    This is a frequency domain estimate, meaning that the LGD residual is approximated via the frequency domain:
    dg_AB^LOS(t) ~ F^-1{Z(f)*F{drho_dotdot(t)}},
    where Z(f) is the transfer function and F{} is the Fourier Transform.
    :param drho_dotdot_array: ndarray
        derivative of reduced range-rate observable array.
    :return: ndarray
        array of LGD residuals (unit of drho_dotdot)
    """
    if isinstance(time_array, (np.ndarray, np.generic, pd.DataFrame)):
        tol = 20  # s
        list_of_pieces, list_of_tpieces = [], []
        # gets the indices of the gaps
        pieces_indices, _ = get_time_pieces(time_array, tol=tol)
        start_indices, end_indices = pieces_indices
        # separates the y_array into pieces, defined by tolerance requirement
        y_array_pieces = [drho_dotdot_array[start:end + 1] for start, end in zip(start_indices, end_indices)]
        t_array_pieces = [time_array[start:end + 1] for start, end in zip(start_indices, end_indices)]

        ind_piece = 0
        for array, t_array in zip(y_array_pieces, t_array_pieces):
            # real Fourier transform
            fourier_drho_dot_dot = np.fft.rfft(array, len(array))
            n = fourier_drho_dot_dot.size
            # sampling time
            dt = 5.0  # s
            # frequency array
            freq_array = np.fft.rfftfreq(len(array), dt)
            # Transfer function
            Z_of_f = transfer_function(freq_array)
            # estimate LGD residual
            dg_AB_LOS_piece = np.fft.irfft(Z_of_f * fourier_drho_dot_dot, len(array))
            # in the case that np.fft.irfft removes an element, simply append the last one of the original array.
            if len(array) != len(dg_AB_LOS_piece):
                dg_AB_LOS_c_piece = np.append(dg_AB_LOS_piece, array[-1])
            else:
                dg_AB_LOS_c_piece = dg_AB_LOS_piece
            # add piece of array to the to-be concanated list
            list_of_pieces.append(dg_AB_LOS_c_piece), list_of_tpieces.append(t_array)
            ind_piece += 1

        # concatenate final list with all array pieces
        dg_AB_LOS_c = np.concatenate(list_of_pieces)
        t_array_final = np.concatenate(list_of_tpieces)

    if time_array is None:
        # real Fourier transform
        fourier_drho_dot_dot = np.fft.rfft(drho_dotdot_array, len(drho_dotdot_array))
        n = fourier_drho_dot_dot.size
        # sampling time
        dt = 5.0  # s
        # frequency array
        freq_array = np.fft.rfftfreq(len(drho_dotdot_array), dt)
        # Transfer function
        Z_of_f = transfer_function(freq_array)
        # estimate LGD residual
        dg_AB_LOS = np.fft.irfft(Z_of_f*fourier_drho_dot_dot, len(drho_dotdot_array))
        # in the case that np.fft.irfft removes an element, simply append the last one of the original array.
        if len(drho_dotdot_array) != len(dg_AB_LOS): dg_AB_LOS_c = np.append(dg_AB_LOS, drho_dotdot_array[-1])
        else: dg_AB_LOS_c = dg_AB_LOS

    return dg_AB_LOS_c