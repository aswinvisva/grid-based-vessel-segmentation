class ThresholdConfig:
    """ Configuration Class for Thresholding"""

    max_std = 0.1
    min_grid_size = 256
    sigma = 4
    max_area_coverage_threshold = 0.995
    n_bins = 26
    noise_scaler = 1

    results_dir = "results"
    debug = False

    def dump(self):
        """Display Configurations."""

        print("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("\t{:30} = {}".format(a, getattr(self, a)))
        print()

    def dump_to_file(self, f):
        """Display Configurations."""

        f.write("Configuration:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                f.write("\n\t{:30} = {}".format(a, getattr(self, a)))

        f.write("\n\n")
