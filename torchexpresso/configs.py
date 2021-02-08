"""
Created on 21.01.2021

@author: Philipp
"""
import json
import os


def merge_params(experiment_config: dict, params: dict, sub_config: str = None):
    if sub_config:
        experiment_config[sub_config]["params"] = {**experiment_config[sub_config]["params"], **params}
    else:
        experiment_config["params"] = {**experiment_config["params"], **params}


def replace_placeholders(experiment_config: dict, placeholders: dict):
    for holder, value in placeholders.items():
        replace_placeholder_in_dict(holder, value, experiment_config)


def replace_placeholder_in_dict(placeholder: str, value: str, parameters: dict):
    for name, parameter in parameters.items():
        if isinstance(parameter, dict):
            replace_placeholder_in_dict(placeholder, value, parameter)
        if isinstance(parameter, list):
            replace_placeholder_in_list(placeholder, value, parameter)
        if isinstance(parameter, str):
            parameters[name] = parameter.replace("$" + placeholder, value)


def replace_placeholder_in_list(placeholder: str, value: str, parameters: list):
    for idx, parameter in enumerate(parameters):
        if isinstance(parameter, dict):
            replace_placeholder_in_dict(placeholder, value, parameter)
        if isinstance(parameter, list):
            replace_placeholder_in_list(placeholder, value, parameter)
        if isinstance(parameter, str):
            parameters[idx] = parameter.replace("$" + placeholder, value)


class ExperimentConfigLoader:

    def __init__(self, config_top_dir: str, ref_words: list = None):
        self.config_top_dir = config_top_dir
        self.ref_words = ref_words or ["model", "dataset", "task", "env", "callbacks", "savers"]
        """ Optionals """
        self.experiment_params = dict()
        self.dataset_params = dict()
        self.placeholders = dict()

    def with_experiment_params(self, **params):
        """ Dynamically inject the given params into the config["params"] on load.
        Usually we want to set if to 'resume' or the 'checkpoint_dir' given as a command line argument.

        :param params: to be injected
        """
        self.experiment_params = params
        return self

    def with_dataset_params(self, **params):
        """ Dynamically inject the given params into the config["dataset"]["params"] on load.
        Usually we want to set the 'dataset_directory' given as a command line argument.

        :param params: to be injected
        """
        self.dataset_params = params
        return self

    def with_placeholders(self, **placeholders):
        """ Replace the given placeholder within the config after load.

        :param placeholders: to be replaced with actual values.
        """
        self.placeholders = placeholders
        return self

    def load(self, experiment_name, comet_user=None):
        experiment_config = self.__load_config(experiment_name)
        cometml_config = self.__load_comet_config(comet_user)
        self.__inject_and_replace(experiment_config, cometml_config)
        return experiment_config

    def __load_config(self, experiment_name):
        experiments_dir = os.path.join(self.config_top_dir, "experiments")
        experiment_configs = [file for file in os.listdir(experiments_dir) if file.endswith(".json")]
        json_name = experiment_name + ".json"
        if json_name not in experiment_configs:
            available_configs = "\n".join(sorted([n.replace(".json", "") for n in experiment_configs]))
            err_msg = "ExperimentConfigurations %s was not found. " \
                      "Available experiment configurations:\n%s" % (json_name, available_configs)
            raise FileNotFoundError(err_msg)
        relative_experiment_config_path = os.path.join("experiments", json_name)
        experiment_config = self.__load_json_config_as_dict(self.config_top_dir, relative_experiment_config_path)
        experiment_config["name"] = experiment_name
        return experiment_config

    def __inject_and_replace(self, experiment_config, cometml_config):
        if "series" in experiment_config:
            self.__inject_and_replace(dict([(c["name"], c) for c in experiment_config["series"]]), cometml_config)
        else:
            if cometml_config:
                # We directly set the cometml params here
                experiment_config["cometml"] = cometml_config
            if len(self.experiment_params) > 0:
                merge_params(experiment_config, self.experiment_params)
            if len(self.dataset_params) > 0:
                merge_params(experiment_config, self.dataset_params, "dataset")
            # Add default placeholders
            self.placeholders["experiment_name"] = experiment_config["name"]
            if self.placeholders:
                replace_placeholders(experiment_config, self.placeholders)

    def __load_comet_config(self, comet_user):
        rel_path = "cometml/offline.json"
        if comet_user:
            rel_path = "cometml/%s.json" % comet_user
        return self.__load_json_config_as_dict(self.config_top_dir, rel_path)

    def __load_json_config_as_dict(self, config_top_directory_or_file, relative_config_file_path=None):
        """
            :param config_top_directory_or_file:
            the top directory in which other config directories are located
            or an absolute path to a config file
            :param relative_config_file_path: the path to a config file relative to the config_top_directory_or_file.
            Can be None, when the other parameter is already pointing to a config file.
        """
        config_path = config_top_directory_or_file
        if os.path.isdir(config_top_directory_or_file):
            config_path = os.path.join(config_top_directory_or_file, relative_config_file_path)
        with open(config_path, "r", encoding="utf8", newline='') as json_file:
            loaded_config = json.load(json_file)
            expanded_config = self.__expand_config_values(config_top_directory_or_file, loaded_config)
        return expanded_config

    def __expand_dict_values(self, config_top_directory_or_file, loaded_value):
        if not isinstance(loaded_value, dict):
            return loaded_value
        for key in loaded_value.keys():
            if key in self.ref_words and loaded_value[key].endswith(".json"):
                file_name = os.path.basename(loaded_value[key])[:-len(".json")]
                loaded_value[key] = self.__load_json_config_as_dict(config_top_directory_or_file, loaded_value[key])
                loaded_value[key]["name"] = file_name
            else:  # go deeper if necessary
                loaded_value[key] = self.__expand_dict_values(config_top_directory_or_file, loaded_value[key])
        return loaded_value

    def __expand_config_values(self, config_top_directory_or_file, loaded_config):
        config = dict()
        for key in loaded_config.keys():
            # Note: These are special pointer keys to dict or file configs (which could also be inlined)
            if key in self.ref_words:
                key_value = loaded_config[key]
                if isinstance(key_value, dict):
                    # if the value is a dict with values that refer to configs
                    config[key] = self.__expand_dict_values(config_top_directory_or_file, loaded_config[key])
                elif isinstance(key_value, list):
                    config[key] = key_value  # simply copy
                elif key_value.endswith(".json"):
                    file_name = os.path.basename(loaded_config[key])[:-len(".json")]
                    config[key] = self.__load_json_config_as_dict(config_top_directory_or_file, loaded_config[key])
                    config[key]["name"] = file_name
                else:
                    raise Exception("Cannot handle key_value for ref_word %s: %s" % (key, key_value))
            else:
                # Note: The key value is a potentially a dict anyway (like 'params')
                config[key] = self.__expand_dict_values(config_top_directory_or_file, loaded_config[key])
        if "series" in loaded_config.keys():  # special case that should only occur once in top level
            series = loaded_config["series"]
            config["series"] = [self.__expand_config_values(config_top_directory_or_file, entry) for entry in series]
            # copy all the values from the "series" config to each actual series entry
            for entry in config["series"]:
                for series_key in loaded_config.keys():
                    if series_key not in ["name", "series", "params"]:  # do now overwrite name or copy the series
                        entry[series_key] = loaded_config[series_key]
                    if series_key == "params":  # merge params if possible
                        if series_key in entry:
                            if isinstance(entry[series_key], dict) and isinstance(loaded_config[series_key], dict):
                                entry[series_key] = {**entry[series_key], **loaded_config[series_key]}
                        else:
                            entry[series_key] = loaded_config[series_key]
        return config
