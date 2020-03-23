import os

from .functions import make_paths


class Manager:
    """
    Сlass for generating the necessary folder structure for calculations
    """

    def __init__(self, **kwargs):

        # initialization from command line arguments
        self.__args = kwargs['args']  # command line arguments
        self.__global_root_dir = self.__args.global_root_dir  # global root directory (//: '/home/<user>/projects/sf')
        self.__global_results_dir_name = self.__args.global_results_dir_name # name of results directory (//: 'results')
        self.__prefix = self.__args.prefix  # string characterizing calculation mode (//: 'xy')
        self.__insert_datetime = self.__args.insert_datetime  # flag to insert datetime to created directories names

        # multidir is a mode in which one common directory is created for a series of similar calculations
        self.__multidir_name = kwargs.get('multidir_name', None)  # name of multidirectory if used (//: 'multidir_sf')

        # global_results_dir = global_root_dir/global_results_dir_name
        # results_dir = global_results_dir/<something with prefix and may be datetime>
        self.__global_results_dir, self.__results_dir, _ = make_paths(self.__global_root_dir,
                                                                      self.__global_results_dir_name,
                                                                      self.__prefix,
                                                                      insert_datetime=self.__insert_datetime)

        # multidir mode
        if self.__multidir_name:
            self.__global_results_dir, self.__results_dir, _ = make_paths(self.__global_root_dir,
                                                                          self.__global_results_dir_name +
                                                                          '/' + self.__multidir_name,
                                                                          prefix=None,
                                                                          insert_datetime=self.__insert_datetime)

        # other directories paths
        self.__track_dir = self.results_dir + '/track'
        self.__beam_dir = self.results_dir + '/beam'

        if kwargs.get('save_field', False):
            self.__field_dir = self.__beam_dir + '/field'
            self.create_dir(self.__field_dir)

    @property
    def field_dir(self):
        return self.__field_dir

    @property
    def results_dir(self):
        return self.__results_dir

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def __create_global_results_dir(self):
        self.create_dir(self.__global_results_dir)

    def __create_results_dir(self):
        self.create_dir(self.__results_dir)

    def __create_track_dir(self):
        self.create_dir(self.__track_dir)

    def __create_beam_dir(self):
        self.create_dir(self.__beam_dir)

    def create_dirs(self):
        """Creates all nessesary directories"""

        self.__create_global_results_dir()
        self.__create_results_dir()
        self.__create_track_dir()
        self.__create_beam_dir()

    @property
    def beam_dir(self):
        return self.__beam_dir

    @property
    def track_dir(self):
        return self.__track_dir
