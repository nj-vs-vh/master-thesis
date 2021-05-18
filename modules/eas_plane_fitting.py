"""
Fitting signal arrival times with plane EAS front (bonus: adaptive point exclusion)
"""

import numpy as np

from scipy.optimize import curve_fit


def eas_plane(xdata, theta, phi, z00):
    x = xdata[:, 0]
    y = xdata[:, 1]
    return z00 - np.tan(theta) * np.cos(phi) * x - np.tan(theta) * np.sin(phi) * y


def angle_between(theta1, phi1, theta2, phi2):
    """Angle between unit vectors with corresponding theta and phi angles"""
    chord = np.sqrt(
        (np.cos(theta1) - np.cos(theta2)) ** 2  # delta Z
        + (np.sin(theta1)*np.sin(phi1) - np.sin(theta2)*np.sin(phi2)) ** 2  # delta Y
        + (np.sin(theta1)*np.cos(phi1) - np.sin(theta2)*np.cos(phi2)) ** 2  # delta X
    )
    return 2 * np.arcsin(chord / 2)


def adaptive_excluding_fit(x_fov, y_fov, t_means, t_stds):
    popt = [np.pi/3, 0, 0]
    x_y = np.concatenate((np.expand_dims(x_fov, 1), np.expand_dims(y_fov, 1)), axis=1)
    in_fit_mask = np.ones_like(t_means, dtype=bool)

    angle_between_acceptable = 0.2  # degrees!
    angle_between_acceptable *= np.pi / 180

    while np.sum(in_fit_mask) > 4:
        new_popt, new_pcov = curve_fit(
            f=eas_plane,
            xdata=x_y[in_fit_mask],
            ydata=t_means[in_fit_mask],
            p0=popt,
            sigma=t_stds[in_fit_mask],
            absolute_sigma=True,
            bounds=([0, -np.pi, -np.inf], [np.pi/2, np.pi, np.inf])
        )
        if angle_between(*popt[:2], *new_popt[:2]) < angle_between_acceptable:
            break
        popt = new_popt
        pcov = new_pcov

        abs_residuals = np.ma.masked_array(np.abs(eas_plane(x_y, *popt) - t_means), mask=np.logical_not(in_fit_mask))
        excluded_point_i = abs_residuals.argmax(fill_value=0)
        in_fit_mask[excluded_point_i] = False

    perr = np.sqrt(np.diag(pcov))

    return popt, perr, in_fit_mask
