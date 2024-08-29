##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import io
import os
import pathlib
import zipfile
import tempfile
import warnings
from typing import Any, Dict, Optional, Union, Tuple

import stable_baselines3 as sb3
import torch as th
from stable_baselines3.common.save_util import data_to_json, json_to_data, open_path
from stable_baselines3.common.utils import get_device, get_system_info
from stable_baselines3.common.type_aliases import TensorDict
from gbrl import ActorCritic, ParametricActor


def save_to_zip_file(
    save_path: Union[str, pathlib.Path, io.BufferedIOBase],
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    pytorch_variables: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
) -> None:
    """
    Save model data to a zip archive. stable_baselines3 code modified to include GBRL

    :param save_path: Where to store the model.
        if save_path is a str or pathlib.Path ensures that the path actually exists.
    :param data: Class parameters being stored (non-PyTorch variables)
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param pytorch_variables: Other PyTorch variables expected to contain name and value of the variable.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    file = open_path(save_path, "w", verbose=0, suffix="zip")
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)
    clean_path = str(save_path).rstrip('.zip')
    # Create a zip-archive and write our objects there.
    with zipfile.ZipFile(file, mode="w") as archive:
        # Do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth", mode="w", force_zip64=True) as pytorch_variables_file:
                th.save(pytorch_variables, pytorch_variables_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w", force_zip64=True) as param_file:
                    th.save(dict_, param_file)
                  
        if isinstance(save_path, (str, pathlib.Path)) and data.get('gbrl', False):
            if not data['shared_tree_struct'] and not data['nn_critic']:
                gbrl_files = [(clean_path + '_policy.gbrl_model', 'gbrl_policy.gbrl_model'), (clean_path + '_value.gbrl_model', 'gbrl_value.gbrl_model')]
                for gbrl_file, gbrl_model in gbrl_files:
                    with open(gbrl_file, "rb") as gbrl_model_file:
                        archive.writestr(gbrl_model, gbrl_model_file.read())
                    os.remove(gbrl_file)
            else:
                gbrl_file = clean_path + ".gbrl_model"
                with open(gbrl_file, "rb") as gbrl_model_file:
                    archive.writestr('actor_critic.gbrl_model', gbrl_model_file.read())
                os.remove(gbrl_file)
        
        # Save metadata: library version when file was saved
        archive.writestr("_stable_baselines3_version", sb3.__version__)
        # Save system info about the current python env
        archive.writestr("system_info.txt", get_system_info(print_info=False)[1])

    if isinstance(save_path, (str, pathlib.Path)):
        file.close()




def load_from_zip_file(
    load_path: Union[str, pathlib.Path, io.BufferedIOBase],
    load_data: bool = True,
    custom_objects: Optional[Dict[str, Any]] = None,
    device: Union[th.device, str] = "auto",
    verbose: int = 0,
    print_system_info: bool = False,
) -> Tuple[Optional[Dict[str, Any]], TensorDict, Optional[TensorDict]]:
    """
    Load model data from a .zip archive

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        ``keras.models.load_model``. Useful when you have an object in
        file that can not be deserialized.
    :param device: Device on which the code should run.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param print_system_info: Whether to print or not the system info
        about the saved model.
    :return: Class parameters, model state_dicts (aka "params", dict of state_dict)
        and dict of pytorch variables
    """
    file = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device(device=device)

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(file) as archive:
            namelist = archive.namelist()
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file_zip allows this).
            data = None
            pytorch_variables = None
            params = {}

            # Debug system info first
            if print_system_info:
                if "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())
                else:
                    warnings.warn(
                        "The model was saved with SB3 <= 1.2.0 and thus cannot print system information.",
                        UserWarning,
                    )

            if "data" in namelist and load_data:
                # Load class parameters that are stored
                # with either JSON or pickle (not PyTorch variables).
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            # Check for all .pth files and load them using th.load.
            # "pytorch_variables.pth" stores PyTorch variables, and any other .pth
            # files store state_dicts of variables with custom names (e.g. policy, policy.optimizer)
            pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".pth"]
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    # File has to be seekable, but param_file is not, so load in BytesIO first
                    # fixed in python >= 3.7
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # Load the parameters with the right ``map_location``.
                    # Remove ".pth" ending with splitext
                    # Note(antonin): we cannot use weights_only=True, as it breaks with PyTorch 1.13, see GH#1911
                    th_object = th.load(file_content, map_location=device, weights_only=False)
                    # "tensors.pth" was renamed "pytorch_variables.pth" in v0.9.0, see PR #138
                    if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                        # PyTorch variables (not state_dicts)
                        pytorch_variables = th_object
                    else:
                        # State dicts. Store into params dictionary
                        # with same name as in .zip file (without .pth)
                        params[os.path.splitext(file_path)[0]] = th_object
            gbrl_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".gbrl_model"]
            gbrl_model = None
            with tempfile.TemporaryDirectory() as temp_dir:
                for gbrl_file in gbrl_files:
                    temp_file_path = os.path.join(temp_dir, gbrl_file)
                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(archive.read(gbrl_file))

                if gbrl_files and data['shared_tree_struct']:
                    gbrl_model = ActorCritic.load_model(os.path.join(temp_dir, gbrl_files[0]))
                elif gbrl_files and not data['shared_tree_struct'] and not data['nn_critic']:
                    gbrl_model = ActorCritic.load_model(os.path.join(temp_dir, gbrl_files[0].replace('_policy.gbrl_model', '').replace('_value.gbrl_model', '')))
                elif gbrl_files and data['nn_critic']:
                    gbrl_model = ParametricActor.load_model(os.path.join(temp_dir, gbrl_files[0]))

    except zipfile.BadZipFile as e:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from e
    finally:
        if isinstance(load_path, (str, pathlib.Path)):
            file.close()
    return data, params, pytorch_variables, gbrl_model