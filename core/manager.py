from core.libs import *
from core.functions import make_paths


class Manager:
    def __init__(self, **kwargs):
        self.args = kwargs["args"]
        self.multidir_name = kwargs.get("multidir_name", None)
        self.global_root_dir = self.args.global_root_dir
        self.global_results_dir_name = self.args.global_results_dir_name
        self.prefix = self.args.prefix

        self.global_results_dir, self.results_dir, _ = make_paths(self.global_root_dir, self.global_results_dir_name,
                                                                  self.prefix)

        if self.prefix == "vortices":
            self.global_results_dir, self.results_dir, _ = make_paths(self.global_root_dir,
                                                                      self.global_results_dir_name + "/" + self.multidir_name,
                                                                      prefix=None)

        self.track_dir = self.results_dir + "/track"
        self.beam_dir = self.results_dir + "/beam"

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def create_global_results_dir(self):
        self.create_dir(self.global_results_dir)

    def create_results_dir(self):
        self.create_dir(self.results_dir)

    def create_track_dir(self):
        self.create_dir(self.track_dir)

    def create_beam_dir(self):
        self.create_dir(self.beam_dir)