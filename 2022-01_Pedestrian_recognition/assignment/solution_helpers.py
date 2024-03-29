import json
import os
from copy import deepcopy
import time

import IPython.display
import numpy as np
from common.visualization import showimage


class DurationAggregator:
    """
    Aggregate durations while iterating.

    Usage example:
    duration_aggregator = DurationAggregator(is_print_durations=False)
    for i in duration_aggregator.aggregate_durations(range(100)):
        pass

    duration_aggregator.get_mean_duration_s()

    """

    def __init__(self, is_print_durations=True):
        self.durations_s = []
        self.is_print_durations = is_print_durations

    def aggregate_durations(self, iterable):
        for i, item in enumerate(iterable):
            start = time.time()
            yield item
            duration_s = time.time() - start
            self.durations_s.append(duration_s)
            if self.is_print_durations:
                print(f"{i:03d}: {duration_s:.3f} s")

    def __len__(self):
        return len(self.durations_s)

    def get_mean_duration_s(self):
        return np.mean(self.durations_s)


def save_frame_pedestrian_dicts(frame_pedestrian_dicts, is_dry_run=False):
    """
    Save frame pedestrian dicts to json file.
    """
    assert type(frame_pedestrian_dicts) == dict, type(frame_pedestrian_dicts)
    clean_frame_pedestrian_dicts = deepcopy(frame_pedestrian_dicts)
    for frame_index, pedestrian_dicts in clean_frame_pedestrian_dicts.items():
        # make json serializable types (np.array -> list, tensorflow float32 -> float)
        for pedestrian_dict in pedestrian_dicts:
            pedestrian_dict["extent_object"] = pedestrian_dict["extent_object"].tolist()
            pedestrian_dict["T_cam_object"] = pedestrian_dict["T_cam_object"].tolist()
            pedestrian_dict["score"] = float(pedestrian_dict["score"])

    if is_dry_run:
        json.dumps(clean_frame_pedestrian_dicts)
    else:
        with open(os.path.join(os.environ["SOURCE_DIR"], "assignment", "frame-pedestrian-dicts.json"), "w") as fp:
            json.dump(clean_frame_pedestrian_dicts, fp, sort_keys=True, indent=4)


def load_frame_pedestrian_dicts():
    """
    Load frame pedestrian dicts from file.
    """
    with open(os.path.join(os.environ["SOURCE_DIR"], "assignment", "frame-pedestrian-dicts.json")) as fp:
        frame_pedestrian_dicts_raw = json.load(fp)
    # make keys int
    frame_pedestrian_dicts = {int(k): v for k, v in frame_pedestrian_dicts_raw.items()}
    # make arrays np.arrays
    for frame_index, pedestrian_dicts in frame_pedestrian_dicts.items():
        for pedestrian_dict in pedestrian_dicts:
            pedestrian_dict["extent_object"] = np.asarray(pedestrian_dict["extent_object"], dtype=np.float32)
            pedestrian_dict["T_cam_object"] = np.asarray(pedestrian_dict["T_cam_object"], dtype=np.float32)

    return frame_pedestrian_dicts


def save_pedestrian_states(pedestrian_states, vehicle_positions, frame_indices):
    tracked_pedestrian_states_file = os.path.join(
        os.environ["SOURCE_DIR"], "assignment", "tracked-pedestrian-states-generated.npz"
    )

    np.savez(
        tracked_pedestrian_states_file,
        pedestrian_states=pedestrian_states,
        vehicle_positions=vehicle_positions,
        frame_indices=frame_indices,
    )


def load_pedestrian_states():
    tracked_pedestrian_states_file = os.path.join(
        os.environ["SOURCE_DIR"], "assignment", "tracked-pedestrian-states.npz"
    )

    npz = np.load(tracked_pedestrian_states_file)
    # list(npz.keys())
    pedestrian_states = npz["pedestrian_states"]
    vehicle_positions = npz["vehicle_positions"]
    frame_indices = npz["frame_indices"]

    return pedestrian_states, vehicle_positions, frame_indices


class DebugOutputAggregator:
    """
    Aggregates debug outputs (k3d plots, matplotlib figures, images, strings)
    and provides an iterator to show all in a jupyter notebook cell.

    To be filled by PedestrianDetector objects and checked afterwards for grading.
    """

    def __init__(self):
        self.debug_outputs = []

        self.debug_show_functions = {
            "k3d plot": lambda p: p.display(),
            "mpl figure": lambda f: IPython.display.display(f),
            "image": lambda i: showimage(i),
            "string": lambda s: print(s),
        }

    def add_k3d_plot(self, name, description, plot):
        assert len(name) >= 3
        assert len(description) >= 20, "Description needs at least 20 charaters"
        debug_type = "k3d plot"
        self.debug_outputs.append((debug_type, name, description, plot))

    def add_matplotlib_figure(self, name, description, figure):
        """
        Make sure you close the figure beforehand by plt.close(figure).
        """
        assert len(name) >= 3
        assert len(description) >= 20, "Description needs at least 20 charaters"
        debug_type = "mpl figure"
        self.debug_outputs.append((debug_type, name, description, figure))

    def add_image(self, name, description, image):
        assert len(name) >= 3
        assert len(description) >= 20, "Description needs at least 20 charaters"
        debug_type = "image"
        self.debug_outputs.append((debug_type, name, description, image))

    def add_string(self, name, description, string):
        assert len(name) >= 3
        assert len(description) >= 20, "Description needs at least 20 charaters"
        debug_type = "string"
        self.debug_outputs.append((debug_type, name, description, string))

    def __iter__(self):
        """
        Iterator for all the debug outputs, printing type, name and description,
        and calling the respective function to show the content in jupyter notebook cell.
        """
        for debug_type, name, description, debug_object in self.debug_outputs:
            debug_show_function = self.debug_show_functions[debug_type]
            print(f"\n({debug_type}) {name}:\n{description}")
            yield debug_show_function(debug_object)

    def __repr__(self):
        """
        Short representation of this object and its aggregated debug objects.
        """
        output_string = ""
        for debug_type, name, description, debug_object in self.debug_outputs:
            output_string += f"({debug_type}): {name}:\n  {description}\n"
        return output_string
