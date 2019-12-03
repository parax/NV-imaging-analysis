# -*- coding: utf-8 -*-

"""

The main script to process widefield data. This script will save the processed
data in a new folder along with the option.json parameters used to run it.
Note post-processing processing or non-basic fitting techniques are currently
held elsewhere. This will eventually be updated to provide the gui interface
which will select which version of the fitting/etc. script to run.

The __spec__=None is a weird python issue at the moment. Best to leave it in
it doesn't harm anything to set it to None. If not set, there are currently
(2019/09/06) issues with the parallel processing slowing down.

"""

__author__ = "David Broadway"

# ============================================================================

import basic.processor as WP
import basic.misc as misc

# ============================================================================

import matplotlib.pyplot as plt

# ============================================================================


def main(__spec__=None):
    # root = tk.Tk()
    # root.withdraw()

    # add a default path with initialdir=''
    # (perhaps add an initial directory in options we don't need to change?)
    #   file_path = filedialog.askopenfilename(title="Choose a file.")

    options = misc.json_to_dict("options/processor.json")

    #    if Path(file_path).is_file():
    #        options['filepath'] = file_path.resolve()
    #    else:
    #       raise RuntimeError("No file chosen")

    wp = WP.Processor(options)

    wp.process_file()

    wp.plot_images()

    wp.plot_area_spectra()

    wp.fit_data()

    misc.dict_to_json(wp.options, "saved_options.json", path_to_dir=wp.options["output_dir"])

    if options["show_plots"]:
        plt.show()


if __name__ == "__main__":
    main()
