"""
Assess covariance models on a dataset, tracking metrics like mean error,
area of the 95% confidence set for each prediction
and whether the true value was in that set, etc.
"""
import argparse
import logging

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from gerbilizer.architectures.base import GerbilizerArchitecture

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from gerbilizer.architectures.ensemble import GerbilizerEnsemble
from gerbilizer.calibration import CalibrationAccumulator
from gerbilizer.training.dataloaders import GerbilVocalizationDataset
from gerbilizer.training.configs import build_config
from gerbilizer.util import make_xy_grids, subplots
from gerbilizer.training.models import build_model, unscale_output
from gerbilizer.outputs.base import ModelOutput, Unit, ProbabilisticOutput

logging.getLogger("matplotlib").setLevel(logging.WARNING)

FIRST_N_VOX_TO_PLOT = 16


def plot_results(f: h5py.File):
    """
    Create and save plots of calibration curve, error distributions, etc.
    """
    point_predictions = f["point_predictions"][:]

    errs = np.linalg.norm(point_predictions - f["scaled_locations"][:], axis=-1)
    fig, axs = subplots(5, sharex=False, sharey=False)
    (err_ax, calib_ax, cset_area_ax, cset_radius_ax, dist_ax) = axs

    err_ax.hist(errs)
    err_ax.set_xlabel("errors (mm)")
    err_ax.set_ylabel("counts")
    err_ax.set_title("error distribution")

    calib_ax.plot(np.linspace(0, 1, 11), f.attrs["calibration_curve"][:], "bo")
    calib_ax.set_xlabel("probability assigned to region")
    calib_ax.set_ylabel("proportion of locations in the region")
    calib_ax.set_title("calibration curve")

    cset_area_ax.hist(f["confidence_set_areas"][:])
    cset_area_ax.set_xlabel("confidence set area (mm^2)")
    cset_area_ax.set_ylabel("counts")
    cset_area_ax.set_title("confidence set area distribution")

    cset_radius_ax.plot(np.sqrt(f["confidence_set_areas"][:]), errs, "bo")
    cset_radius_ax.set_xlabel("square root confidence set area (mm)")
    cset_radius_ax.set_ylabel("error (mm)")
    cset_radius_ax.set_title("sqrt confidence set area vs error")

    dist_ax.plot(f["distances_to_furthest_point"][:], errs, "bo")
    dist_ax.set_xlabel("distance to furthest point in confidence set (mm)")
    dist_ax.set_ylabel("error (mm)")
    dist_ax.set_title("distance to furthest point vs error")

    return fig, axs


def assess_model(
    model: GerbilizerArchitecture,
    dataloader: DataLoader,
    outfile: Union[Path, str],
    arena_dims: tuple,
    device="cuda:0",
    visualize=False,
):
    """
    Assess the provided model with uncertainty, storing model output as well as
    info like error, confidence sets, and a calibration curve in the h5 format
    at path `outfile`.

    Optionally, visualize confidence sets a few times throughout training.
    """
    outfile = Path(outfile)

    N = dataloader.dataset.n_vocalizations

    with h5py.File(outfile, "w") as f:
        scaled_locations = f.create_dataset("scaled_locations", shape=(N, 2))

        raw_model_output = f.create_dataset("raw_model_output", shape=(N, model.n_outputs))
        point_predictions = f.create_dataset("point_predictions", shape=(N, 2))

        ca = CalibrationAccumulator(arena_dims)

        model.eval()
        with torch.no_grad():
            for idx, (audio, location) in enumerate(dataloader):
                audio = audio.to(device)
                output: ModelOutput = model(audio)

                raw_model_output[idx] = output.raw_output.squeeze()
                point_predictions[idx] = output.point_estimate(units=Unit.MM)

                # unscale location from [-1, 1] square to units in arena (in mm)
                scaled_location = output._convert(location, Unit.ARBITRARY, Unit.MM).cpu().numpy()
                scaled_locations[idx] = scaled_location

                # other useful info
                if isinstance(output, ProbabilisticOutput):
                    ca.calculate_step(output, scaled_location)

                if visualize and idx == FIRST_N_VOX_TO_PLOT:
                    # plot the densities
                    visualize_dir = outfile.parent / "pmfs_visualized"
                    visualize_dir.mkdir(exist_ok=True, parents=True)
                    visualize_outfile = visualize_dir / f"{outfile.stem}_visualized.png"

                    sets_to_plot = ca.confidence_sets[:idx]
                    associated_locations = scaled_locations[:idx]

                    _, axs = subplots(len(sets_to_plot))

                    xgrid, ygrid = make_xy_grids(
                        arena_dims, shape=sets_to_plot[0].shape, return_center_pts=True
                    )
                    for i, ax in enumerate(axs):
                        ax.set_title(f"vocalization {i}")
                        ax.contourf(xgrid, ygrid, sets_to_plot[i])
                        # add a red dot indicating the true location
                        ax.plot(*associated_locations[i], "ro")
                        ax.set_aspect("equal", "box")

                    plt.savefig(visualize_outfile)
                    print(f"Model output visualized at file {visualize_outfile}")

        results = ca.results()
        f.attrs["calibration_curve"] = results["calibration_curve"]

        # f.create_dataset("scaled_output", data=np.array(scaled_output))

        # array-like quantities outputted by the calibration accumulator
        OUTPUTS = (
            "confidence_sets",
            "confidence_set_areas",
            "location_in_confidence_set",
            "distances_to_furthest_point",
        )
        for output_name in OUTPUTS:
            f.create_dataset(output_name, data=results[output_name])

        _, axs = plot_results(f)
        plt.tight_layout()
        plt.savefig(Path(outfile).parent / f"{Path(outfile).stem}_results.png")

    print(f"Model output saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Used to specify model configuration via a JSON file.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to an h5 file on which to assess the model",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help="Path at which to store results. Must be an h5 file.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Include flag to plot confidence sets occasionally during assessment.",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        raise ValueError(
            f"Requested config JSON file could not be found: {args.config}"
        )

    config_data = build_config(args.config)

    # load the model
    weights_path = config_data.get("WEIGHTS_PATH", None)

    # if not weights_path:
    #     raise ValueError(
    #         f"Cannot evaluate model as the config stored at {args.config} doesn't include path to weights."
    #     )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        config_data["GENERAL"]["DEVICE"] = "cpu"

    model, _ = build_model(config_data)
    if weights_path:
        weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights)

    arena_dims = config_data["DATA"]["ARENA_DIMS"]

    dataset = GerbilVocalizationDataset(
        str(args.data),
        arena_dims=arena_dims,
        make_xcorrs=config_data["DATA"]["COMPUTE_XCORRS"],
        crop_length=config_data["DATA"].get("CROP_LENGTH", None),
        sequential=True,
    )

    # function that torch dataloader uses to assemble batches.
    # in our case, applying this is necessary because of the structure of the
    # dataset class (which was created to accomodate variable length batches, so it's a little wonky)
    collate_fn = lambda x: x[0]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # make the parent directories for the desired outfile if they don't exist
    parent = Path(args.outfile).parent
    parent.mkdir(exist_ok=True, parents=True)

    assess_model(
        model,
        dataloader,
        args.outfile,
        arena_dims,
        device=device,
        visualize=args.visualize,
    )
