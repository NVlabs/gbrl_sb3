##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import torch as th
from PIL import Image
from pathlib import Path
import cv2
from gym.core import ObsType
from typing import Any, Union
from stable_baselines3.common.policies import ActorCriticPolicy

from shap.explainers._deep import DeepExplainer
from shap._explanation import Explanation
from shap.explainers._explainer import Explainer
from packaging import version
import warnings
from stable_baselines3.common.vec_env import VecVideoRecorder
from typing import Callable

from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from shap.explainers._deep.deep_pytorch import PyTorchDeep, add_interim_values, deeplift_grad, _check_additivity

def get_logits(ac_policy, obs):
    features = ac_policy.extract_features(obs)
    if ac_policy.share_features_extractor:
        latent_pi, _ = ac_policy.mlp_extractor(features)
    else:
        pi_features, _ = features
        latent_pi = ac_policy.mlp_extractor.forward_actor(pi_features)
    logits = ac_policy.action_net(latent_pi)
    return logits


class ShapVecVideoRecorder(VecVideoRecorder):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):
        super().__init__(venv, video_folder, record_video_trigger, video_length, name_prefix)

    def step_wait(self) -> VecEnvStepReturn:
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1

        return obs, rews, dones, infos


class PolicyPyTorchDeep(PyTorchDeep):
    def __init__(self, model, data):
        import torch
        if version.parse(torch.__version__) < version.parse("0.4"):
            warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]
        self.data = data
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = get_logits(self.model, *data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
        self.model = model.eval()

        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            outputs = get_logits(model, *data)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(-1)
            # also get the device everything is running on
            self.device = outputs.device
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu().numpy()

    def gradient(self, idx, inputs):
        import torch
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = get_logits(self.model, *X)
        selected = [val for val in outputs[:, idx]]
        grads = []
        if self.interim:
            interim_inputs = self.layer.target_input
            for idx, input in enumerate(interim_inputs):
                grad = torch.autograd.grad(selected, input,
                                           retain_graph=True if idx + 1 < len(interim_inputs) else None,
                                           allow_unused=True)[0]
                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = torch.zeros_like(X[idx]).cpu().numpy()
                grads.append(grad)
            del self.layer.target_input
            return grads, [i.detach().cpu().numpy() for i in interim_inputs]
        else:
            for idx, x in enumerate(X):
                grad = torch.autograd.grad(selected, x,
                                           retain_graph=True if idx + 1 < len(X) else None,
                                           allow_unused=True)[0]
                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = torch.zeros_like(X[idx]).cpu().numpy()
                grads.append(grad)
            return grads
        
    def shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=True):
        import torch
        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert not isinstance(X, list), "Expected a single tensor model input!"
            X = [X]
        else:
            assert isinstance(X, list), "Expected a list of model inputs!"

        X = [x.detach().to(self.device) for x in X]

        model_output_values = None

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = get_logits(self.model, *X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                emsg = "output_rank_order must be max, min, or max_abs!"
                raise ValueError(emsg)
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (torch.ones((X[0].shape[0], self.num_outputs)).int() *
                                  torch.arange(0, self.num_outputs).int())

        # add the gradient handles
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(np.zeros((X[0].shape[0], ) + self.interim_inputs_shape[k][1: ]))
            else:
                for k in range(len(X)):
                    phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                tiled_X = [X[t][j:j + 1].repeat(
                                   (self.data[t].shape[0],) + tuple([1 for k in range(len(X[t].shape) - 1)])) for t
                           in range(len(X))]
                joint_x = [torch.cat((tiled_X[t], self.data[t]), dim=0) for t in range(len(X))]
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)
                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for k in range(len(output)):
                        x_temp, data_temp = np.split(output[k], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    for t in range(len(self.interim_inputs_shape)):
                        phis[t][j] = (sample_phis[t][self.data[t].shape[0]:] * (x[t] - data[t])).mean(0)
                else:
                    for t in range(len(X)):
                        phis[t][j] = (torch.from_numpy(sample_phis[t][self.data[t].shape[0]:]).to(self.device) * (X[t][j: j + 1] - self.data[t])).cpu().detach().numpy().mean(0)
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        # check that the SHAP values sum up to the model output
        if check_additivity:
            if model_output_values is None:
                with torch.no_grad():
                    model_output_values = get_logits(self.model, *X)

            _check_additivity(self, model_output_values.cpu(), output_phis)

        if isinstance(output_phis, list):
            # in this case we have multiple inputs and potentially multiple outputs
            if isinstance(output_phis[0], list):
                output_phis = [np.stack([phi[i] for phi in output_phis], axis=-1)
                               for i in range(len(output_phis[0]))]
            # multiple outputs case
            else:
                output_phis = np.stack(output_phis, axis=-1)
        if ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis


class PolicyDeepExplainer(Explainer):
    def __init__(self, model, data, session=None, learning_phase_flags=None):
        # first, we need to find the framework
        framework = 'pytorch'
        masker = data
        super().__init__(model, masker)
        self.explainer = PolicyPyTorchDeep(model, data)

        self.expected_value = self.explainer.expected_value
        self.explainer.framework = framework

    def __call__(self, X: Union[list, 'np.ndarray', 'pd.DataFrame', 'torch.tensor']) -> Explanation:  # noqa: F821
        """Return an explanation object for the model applied to X.

        Parameters
        ----------
        X : list,
            if framework == 'tensorflow': numpy.array, or pandas.DataFrame
            if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        Returns
        -------
        shap.Explanation:
        """
        shap_values = self.shap_values(X)
        return Explanation(values=shap_values, data=X)

    def shap_values(self, X, ranked_outputs=None, output_rank_order='max', check_additivity=True):
        """Return approximate SHAP values for the model applied to the data given by X.

        Parameters
        ----------
        X : list,
            if framework == 'tensorflow': np.array, or pandas.DataFrame
            if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where shap_values is a list of numpy
            arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
            which output indexes were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        Returns
        -------
        np.array or list
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            The shape of the returned array depends on the number of model outputs:

            * one input, one output: matrix of shape ``(#num_samples, *X.shape[1:])``.
            * one input, multiple outputs: matrix of shape ``(#num_samples, *X.shape[1:], #num_outputs)``
            * multiple inputs, one or more outputs: list of matrices, with shapes of one of the above.

            If ranked_outputs is ``None`` then this list of tensors matches
            the number of model outputs. If ranked_outputs is a positive integer a pair is returned
            (shap_values, indexes), where shap_values is a list of tensors with a length of
            ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
            were chosen as "top".

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs and one input changed from list to np.ndarray.

        """
        return self.explainer.shap_values(X, ranked_outputs, output_rank_order, check_additivity=check_additivity)
    

class MiniGridShapVisualizationWrapper(gym.Wrapper):
    def __init__(self, env, feature_labels=['Agent Direction', 'Mission'], plot_path='', algo_type=''):
        super().__init__(env)
        self.feature_labels = feature_labels
        self.height = 224
        # # self.width = int(224*1.5)
        self.width = int(224)
        # self.height = 7
        # # self.width = int(224*1.5)
        # self.width = 7
        self.mission = None
        self.extra_shap = None
        self.image_shape = np.zeros((7, 7))
        self.algo_type=algo_type
        self.save_idx = 0
        self.go_forward_idx = 0
        self.plot_path = Path(plot_path)
        self.additional_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # To store the SHAP visualization

    def set_shap_values(self, shap_values, actions):
        """
        Generate the SHAP visualization and store it.
        """
        self.additional_image = self.generate_shap_image(shap_values.squeeze(), actions)

    def generate_shap_image(self, shap_values, actions):
        """
        Generates a combined heatmap and bar plot image from SHAP values.
        """
        # max_shap = np.max(shap_values)
        # shap_values = shap_values / max_shap
        self.max_shap = np.max(shap_values)
        self.min_shap = np.min(shap_values)
        # Ensure shap_values has 51 elements (49 for heatmap, 2 for bar plot)
        if shap_values.shape[0] != 51 and len(shap_values) != 2836:
            raise ValueError("Expected 51 SHAP values or 2836 or flat vector.")
        if len(shap_values) == 2836:
            image_shap = shap_values[:147].reshape(7, 7, 3).sum(axis=-1)
            extra_shap = np.zeros(2)
            extra_shap[0] = shap_values[-1]
            extra_shap[1] = shap_values[147:-1].sum()

        else:
            # Split the SHAP values
            image_shap = shap_values[:49].reshape(7, 7)
            extra_shap = shap_values[49:]
        image_shap = image_shap.T 
        self.image_shape = image_shap
        dpi = 100
        target_width = self.width
        target_height = self.height 
        figsize = (target_width / dpi, target_height / dpi)
        # figsize = (target_width, target_height)
        width_ratios = [1, 1] 
        # Create the figure with two subplots
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        
        # Plot the heatmap
        norm = plt.Normalize(vmin=-8, vmax=8)  # Replace with your heatmap's vmin and vmax
        self.mission = self.env.envs[0].mission
        self.extra_shap = extra_shap

        # # Map SHAP values to colors using the same colormap
        colors = plt.cm.coolwarm(norm(extra_shap))
        # Plot the bar chart
        ax.bar(self.feature_labels, extra_shap, color=colors)
        ax.set_title("Additional Features", fontsize=6)
        ax.set_ylabel("SHAP Value", fontsize=6)
        ax.set_xticks(range(len(self.feature_labels)))
        ax.set_xticklabels(self.feature_labels, rotation=0, ha='right', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylim((-6, 6))
        fig.suptitle(f"Mission: {self.env.envs[0].mission} \n Action: {actions}", fontsize=7)
        # Adjust layout
        plt.tight_layout()
        
        # Render the figure to a NumPy array
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        plot_img = image[:, :, :3]  # Discard alpha channel
        plt.close(fig)

        if actions == 'pickup object' or actions == 'go forward':
            self.save_shap_image(actions)
        
        return plot_img
    
    def save_shap_image(self, action):
        # from pathlib import Path

        frame = self.env.render(mode='rgb_array')
        from matplotlib import rc

        # Enable LaTeX rendering
        rc('text', usetex=True)

        # Set the font family to serif
        rc('font', family='serif')
        fig, ax = plt.subplots(nrows=1, ncols=2)
        
        # Plot the heatmap
        norm = plt.Normalize(vmin=self.min_shap, vmax=self.max_shap)  # Replace with your heatmap's vmin and vmax
        # # Map SHAP values to colors using the same colormap
        colors = plt.cm.coolwarm(norm(self.extra_shap))
        # Plot the bar chart
        ax[1].bar(self.feature_labels, self.extra_shap, color=colors)
#         axes.labelsize: 12.0
# axes.titlesize: 14.0
# xtick.labelsize: 10.0
# ytick.labelsize: 10.0
# legend.fontsize: 10.0
# figure.titlesize: 16.0
# font.size: 10.0
        ax[1].set_title("Additional Features", fontsize=18)
        ax[1].set_ylabel("SHAP Value", fontsize=16)
        ax[1].set_xticks(range(len(self.feature_labels)))
        ax[1].set_xticklabels(self.feature_labels, rotation=0, ha='right', fontsize=14)
        ax[1].tick_params(axis='y')
        # ax[1].set_ylim((-7, 7) if 'nn' in self.algo_type else ('-36, 36'))
        # ax[1].set_ylim((-36, 36))
        fig.suptitle(f"Mission: {self.mission} \n Action: {action}", fontsize=18)

        shap_values = self.image_shape  # SHAP values already passed
        tile_size = 32  # Size of each grid cell
        grid_size = 7  # 7x7 grid for SHAP values
        start_x = (224 - (grid_size * tile_size)) // 2  # Center horizontally
        start_y = 224 - (grid_size * tile_size)  # Start from the bottom
        
        for i in range(grid_size):  # Iterate over rows
            for j in range(grid_size):  # Iterate over columns
                # Calculate the top-left corner of each grid cell
                x = start_x + (j * tile_size)
                y = start_y + (i * tile_size)
                
                # Calculate the center for text placement
                text_x = x + tile_size // 2
                text_y = y + tile_size // 2
                
                # Get the SHAP value for this cell
                shap_value = shap_values[i, j] if i < shap_values.shape[0] and j < shap_values.shape[1] else 0.0
                
                # Choose text color for better visibility
                text_color = (0, 0, 0) if abs(shap_value) < 3 else (255, 255, 255)

                text = f"{int(shap_value)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Calculate the precise text position for centering
                text_x_centered = text_x - (text_width // 2)
                # text_x_centered = text_x
                text_y_centered = text_y + (text_height // 2)
                
                # Add SHAP value as text
                cv2.putText(
                    frame,
                    text,
                    (text_x_centered, text_y_centered),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA
                )
        ax[0].imshow(frame)
        ax[0].set_xticklabels([])  # Hide x-axis tick labels
        ax[0].set_yticklabels([])  # Hide y-axis tick labels
        # Adjust layout
        idx = self.go_forward_idx if action == 'go forward' else self.save_idx
        action = action.replace(' ', '_')
        plt.tight_layout()
        fig.savefig(self.plot_path / f'shap_{self.algo_type}_{idx}_{action}.png')
        fig.savefig(self.plot_path / f'shap_{self.algo_type}_{idx}_{action}.pdf')
        
        if action == 'go_forward':
            self.go_forward_idx += 1
        else:
            self.save_idx += 1
        plt.close(fig)
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # self.additional_image = None
        self.additional_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.image_shape = np.zeros((7, 7))
        return self.env.reset()


    def render(self, mode='rgb_array'):
        # Get the original frame
        frame = self.env.render(mode=mode)
        
        if self.additional_image is not None:
            # grid, vis_mask = self.env.envs[0].gen_obs_grid()
            # Optionally, resize additional_image to match frame height or other desired size
            # frame_height, frame_width, _ = frame.shape
            # plot_height, plot_width, _ = self.additional_image.shape
            
            # # Decide where to place the plot (e.g., side-by-side)
            # # Resize plot to have the same height
            # if plot_height != frame_height:
            #     scaling_factor = frame_height / plot_height
            #     new_width = int(plot_width * scaling_factor)
            #     resized_plot = cv2.resize(self.additional_image, (new_width, frame_height), interpolation=cv2.INTER_LINEAR)
            # else:
            #     resized_plot = self.additional_image
            
            # # Concatenate horizontally
            # combined_frame = np.concatenate((frame, resized_plot), axis=1)
            # return combined_frame
            frame_height, frame_width, _ = frame.shape
            
            # Assuming a 7x7 grid corresponds to the frame view
            grid_size_x = frame_width // 7
            grid_size_y = frame_height // 7
            
            shap_values = self.image_shape  # SHAP values already passed
            tile_size = 32  # Size of each grid cell
            grid_size = 7  # 7x7 grid for SHAP values
            start_x = (224 - (grid_size * tile_size)) // 2  # Center horizontally
            start_y = 224 - (grid_size * tile_size)  # Start from the bottom
            
            for i in range(grid_size):  # Iterate over rows
                for j in range(grid_size):  # Iterate over columns
                    # Calculate the top-left corner of each grid cell
                    x = start_x + (j * tile_size)
                    y = start_y + (i * tile_size)
                    
                    # Calculate the center for text placement
                    text_x = x + tile_size // 2
                    text_y = y + tile_size // 2
                    
                    # Get the SHAP value for this cell
                    shap_value = shap_values[i, j] if i < shap_values.shape[0] and j < shap_values.shape[1] else 0.0
                    
                    # Choose text color for better visibility
                    text_color = (0, 0, 0) if abs(shap_value) < 3 else (255, 255, 255)

                    text = f"{int(shap_value)}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Calculate the precise text position for centering
                    text_x_centered = text_x - (text_width // 2)
                    # text_x_centered = text_x
                    text_y_centered = text_y + (text_height // 2)
                    
                    # Add SHAP value as text
                    cv2.putText(
                        frame,
                        text,
                        (text_x_centered, text_y_centered),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                        cv2.LINE_AA
                    )
            # Iterate over the 7x7 grid
            plot_height, plot_width, _ = self.additional_image.shape
            
            # Decide where to place the plot (e.g., side-by-side)
            # Resize plot to have the same height
            if plot_height != frame_height:
                scaling_factor = frame_height / plot_height
                new_width = int(plot_width * scaling_factor)
                resized_plot = cv2.resize(self.additional_image, (new_width, frame_height), interpolation=cv2.INTER_LINEAR)
            else:
                resized_plot = self.additional_image
            
            # Concatenate horizontally
            # return frame
            combined_frame = np.concatenate((frame, resized_plot), axis=1)
            combined_frame = cv2.resize(combined_frame, (1024, 512), interpolation=cv2.INTER_LINEAR)
            return combined_frame
            # return frame
        else:
            return frame
