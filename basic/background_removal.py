# -*- coding: utf-8 -*-

"""
Created on Tue Aug 20 13:43:32 2019

@author: David

Module for background subtraction of images

"""

__author__ = "David Broadway"

from scipy import optimize
import numpy as np
import skimage
import basic.worker_plotting as WrkPlt

# ============================================================================
# Remove background fitting
# ============================================================================


def remove_background(
    data,
    model=None,
    poly_order=1,
    options=None,
    results="basic",
    plot_fit=False,
    title="",
    cbar_title="",
    colour_map="bwr",
    use_upper_threshold=False,
    upper_threshold=None,
    use_lower_threshold=False,
    lower_threshold=None,
    paper_figure=False,
    **kwargs,
):
    # if no model is given use a polynomial fit.
    if model is None:
        model = "poly"

    # make a copy of the data set so that the masking and raw data can be plotted seperately.
    old_data = data.copy()

    # Remove pixels in the array that are above or below the given thresholds.
    if use_upper_threshold:
        data[data > upper_threshold] = np.nan
    if use_lower_threshold:
        data[data < lower_threshold] = np.nan

    # If the model is a mean subtract the mean of the masked data set
    if model == "mean":
        bg_removed = old_data - np.nanmean(data)
        best_fit = np.nanmean(data)
        results = "basic"
    else:
        # Perform a fit on the masked data set
        if model == "poly":
            p_results, best_fit = fitpoly(data, poly_order)
        elif model == "gaussian":
            p_results, best_fit = fitgaussian(data)
        # background removed data set
        bg_removed = old_data - best_fit

    # plot the background subtraction
    if plot_fit:
        back_sub_images = np.zeros((4, data.shape[0], data.shape[1]))
        back_sub_images[0, ::] = old_data
        back_sub_images[1, ::] = data
        back_sub_images[2, ::] = best_fit
        back_sub_images[3, ::] = bg_removed

        plt_image = WrkPlt.Image(options=options, title="Background subtraction " + title)
        sub_titles = [title, title + " masked", title + " background fit", title + " sub bg"]
        plt_image.multiple_images(
            back_sub_images,
            title=sub_titles,
            cbar_title=cbar_title,
            colour_map=colour_map,
            paper_figure=paper_figure,
            **{**kwargs, **{"sub_back": False}},
        )
        # note: sub_back needs to be set to false to stop more background subtractions in the
        # plotting itself.

    if results == "basic":
        return bg_removed
    elif results == "full":
        return p_results, best_fit, bg_removed


# < TODO > include other filtering options


def image_filtering(
    data,
    flt_type=None,
    sigma=1,
    plot_fit=True,
    options=None,
    use_upper_threshold=False,
    upper_threshold=None,
    use_lower_threshold=False,
    lower_threshold=None,
    title="filter",
    cbar_title="(a.u.)",
    colour_map="bwr",
    return_filter=False,
    paper_figure=False,
    **kwargs,
):
    if flt_type is None:
        flt_type = "gaussian"
    old_data = data.copy()
    # Remove pixels in the array that are above or below the given thresholds.
    if use_upper_threshold:
        # Find all values above threshold and set to nan
        data[data > upper_threshold] = np.nan
        # the filter can't deal with nans so replace with the mean after threshold
        # data[np.isnan(data)] = np.nanmean(data)
    if use_lower_threshold:
        # same for lower threshold
        data[data < lower_threshold] = np.nan

    data[np.isnan(data)] = np.nanmean(data)

    if flt_type == "gaussian":
        filter = skimage.filters.gaussian(data, sigma=sigma)
        filtered_image = old_data - filter
        # plot the background subtraction

    if plot_fit:
        back_sub_images = np.zeros((4, data.shape[0], data.shape[1]))
        back_sub_images[0, ::] = old_data
        back_sub_images[1, ::] = data
        back_sub_images[2, ::] = filter
        back_sub_images[3, ::] = filtered_image

        plt_image = WrkPlt.Image(options=options, title="filtered " + title)
        sub_titles = [title, title + " masked", "filter", title + " filtered"]
        plt_image.multiple_images(
            back_sub_images,
            title=sub_titles,
            cbar_title=cbar_title,
            colour_map=colour_map,
            paper_figure=paper_figure,
            **{**kwargs, **{"sub_back": False}},
        )
    if return_filter:
        return filter
    else:
        return filtered_image


# ============================================================================
# Guassian background fitting
# ============================================================================


def gaussian(p, x, y):
    height, center_x, center_y, width_x, width_y = p
    return height * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    total = np.nansum(data)
    X, Y = np.indices(data.shape)
    center_x = np.nansum(X * data) / total
    center_y = np.nansum(Y * data) / total
    if center_x > np.max(X) or center_x < 0:
        center_x = np.max(X) / 2
    if center_y > np.max(X) or center_y < 0:
        center_y = np.max(Y) / 2
    row = data[int(center_x), :]
    col = data[:, int(center_y)]
    width_x = np.nansum(np.sqrt(abs((np.arange(col.size) - center_y) ** 2 * col)) / np.nansum(col))
    width_y = np.nansum(np.sqrt(abs((np.arange(row.size) - center_x) ** 2 * row)) / np.nansum(row))
    height = np.nanmax(data)
    return height, center_x, center_y, width_x, width_y


def errorfunction(p, x, y, data):
    return gaussian(p, x, y) - data


def fitgaussian(data):
    params = moments(data)
    X, Y = np.indices(data.shape)
    mask = ~np.isnan(data)
    x = X[mask]
    y = Y[mask]
    data = data[mask]
    p, success = optimize.leastsq(errorfunction, params, args=(x, y, data))
    guassian_fit = gaussian(p, X, Y)
    return p, guassian_fit


# ============================================================================
# Polynomial background fitting
# ============================================================================


def poly(p, x, y, n):
    poly = poly_order_1(p, x, y)
    if n >= 2:
        poly = poly + poly_order_2(p, x, y)
    if n >= 3:
        poly = poly + poly_order_3(p, x, y)
    if n >= 4:
        poly = poly + poly_order_4(p, x, y)
    if n >= 5:
        poly = poly + poly_order_5(p, x, y)
    if n >= 6:
        poly = poly + poly_order_6(p, x, y)
    if n >= 7:
        poly = poly + poly_order_7(p, x, y)
    if n >= 8:
        poly = poly + poly_order_8(p, x, y)
    return poly


def poly_errorfunction(p, x, y, n, data):
    return poly(p, x, y, n) - data


def fitpoly(data, n):
    """ Fit a polynomial to an image.
    Takes the data and the polynomial order (n).
    Returns the fitted values (p) and an image of the background fit (poly_fit)
    """
    num_params = [3, 6, 10, 15, 21, 28, 36, 45]
    params = np.zeros(num_params[n - 1])
    params[0] = np.nanmean(data)
    X, Y = np.indices(data.shape)
    mask = ~np.isnan(data)
    x = X[mask]
    y = Y[mask]
    data = data[mask]
    p, success = optimize.leastsq(poly_errorfunction, params, args=(x, y, n, data))
    poly_fit = poly(p, X, Y, n)
    return p, poly_fit


def poly_order_1(p, x, y):
    ab0, a1, b1 = p[0:3]
    return a1 * x + b1 * y


def poly_order_2(p, x, y):
    a2, b2, ab2 = p[3:6]
    return ab2 * x * y + a2 * x ** 2 + b2 * y ** 2


def poly_order_3(p, x, y):
    a3, b3, aab3, abb3 = p[6:10]
    return aab3 * x ** 2 * y + abb3 * x * y ** 2 + a3 * x ** 3 + b3 * y ** 3


def poly_order_4(p, x, y):
    a4, b4, aaab4, abbb4, aabb4 = p[10:15]
    return (
        aaab4 * x ** 3 * y
        + abbb4 * x * y ** 3
        + aabb4 * x ** 2 * y ** 2
        + a4 * x ** 4
        + b4 * y ** 4
    )


def poly_order_5(p, x, y):
    a5, b5, aaaab5, abbbb5, aaabb5, aabbb5, = p[15:21]
    return (
        aaaab5 * x ** 4 * y
        + aaabb5 * x ** 3 * y ** 2
        + aabbb5 * x ** 2 * y ** 3
        + abbbb5 * x * y ** 4
        + a5 * x ** 5
        + b5 * y ** 5
    )


def poly_order_6(p, x, y):
    a6, b6, a5b1, a4b2, a3b3, a2b4, a1b5, = p[21:28]
    return (
        a5b1 * x ** 5 * y
        + a4b2 * x ** 4 * y ** 2
        + a3b3 * x ** 3 * y ** 3
        + a2b4 * x ** 2 * y ** 4
        + a1b5 * x ** 1 * y ** 5
        + a6 * x ** 6
        + b6 * y ** 6
    )


def poly_order_7(p, x, y):
    a7, b7, a6b1, a5b2, a4b3, a3b4, a2b5, a1b6 = p[28:36]
    return (
        a6b1 * x ** 6 * y
        + a5b2 * x ** 5 * y ** 2
        + a4b3 * x ** 4 * y ** 3
        + a3b4 * x ** 3 * y ** 4
        + a2b5 * x ** 2 * y ** 5
        + a1b6 * x ** 1 * y ** 6
        + a7 * x ** 7
        + b7 * y ** 7
    )


def poly_order_8(p, x, y):
    a7, b7, a7b1, a6b2, a5b3, a4b4, a3b5, a2b6, a1b7 = p[36:45]
    return (
        a7b1 * x ** 7 * y
        + a6b2 * x ** 6 * y ** 2
        + a5b3 * x ** 5 * y ** 3
        + a4b4 * x ** 4 * y ** 4
        + a3b5 * x ** 3 * y ** 5
        + a2b6 * x ** 2 * y ** 6
        + a1b7 * x ** 1 * y ** 7
        + a7 * x ** 6
        + b7 * y ** 6
    )
