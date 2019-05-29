import os

from .functions import make_paths


class Manager:
    def __init__(self, **kwargs):
        self.__args = kwargs['args']
        self.__multidir_name = kwargs.get('multidir_name', None)
        self.__global_root_dir = self.__args.global_root_dir
        self.__global_results_dir_name = self.__args.global_results_dir_name
        self.__prefix = self.__args.prefix

        self.__global_results_dir, self.__results_dir, _ = make_paths(self.__global_root_dir, self.__global_results_dir_name,
                                                                  self.__prefix)

        if self.__multidir_name:
            self.__global_results_dir, self.__results_dir, _ = make_paths(self.__global_root_dir,
                                                                      self.__global_results_dir_name + '/' + self.__multidir_name,
                                                                      prefix=None)

        self.__track_dir = self.results_dir + '/track'
        self.__beam_dir = self.results_dir + '/beam'

    @property
    def results_dir(self):
        return self.__results_dir

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def create_global_results_dir(self):
        self.create_dir(self.__global_results_dir)

    def create_results_dir(self):
        self.create_dir(self.__results_dir)

    def create_track_dir(self):
        self.create_dir(self.__track_dir)

    def create_beam_dir(self):
        self.create_dir(self.__beam_dir)

    @property
    def beam_dir(self):
        return self.__beam_dir

    @property
    def track_dir(self):
        return self.__track_dir
