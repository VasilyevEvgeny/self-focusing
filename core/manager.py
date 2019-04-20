from core.libs import *


class Manager:
    def __init__(self, **kwargs):
        self.global_root_dir = kwargs.get("global_root_dir", "/".join(os.path.abspath(__file__).split("\\")[:-2]))
        self.global_results_dir_name = "Self-focusing_3D_results"
        self.global_results_dir = self.global_root_dir + "/" + self.global_results_dir_name
        datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = self.global_results_dir + "/" + datetime_string
        self.track_dir = self.results_dir + "/track"
        self.beam_dir = self.results_dir + "/beam"

    @staticmethod
    def create_dir(path):
        if os.path.exists(path):
            raise Exception("Directory %s already exists" % path)
        else:
            os.makedirs(path)

    def create_global_results_dir(self):
        try:
            self.create_dir(self.global_results_dir)
        except:
            pass

    def create_results_dir(self):
        self.create_dir(self.results_dir)

    def create_track_dir(self):
        self.create_dir(self.track_dir)

    def create_beam_dir(self):
        self.create_dir(self.beam_dir)