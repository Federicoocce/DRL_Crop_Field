from __future__ import annotations

import importlib.resources
from typing import Any

import cv2
import jinja2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point, Polygon

from virtual_maize_field.world_generator.models import (
    CROP_MODELS,
    LITTER_MODELS,
    MARKER_MODELS,
    WEED_MODELS,
    GazeboModel,
    to_gazebo_models,
)
from virtual_maize_field.world_generator.segments import (
    CurvedSegment,
    IslandSegment,
    SinCurvedSegment,
    StraightSegment,
)
from virtual_maize_field.world_generator.utils import BoundedGaussian
from virtual_maize_field.world_generator.world_description import WorldDescription


class Field2DGenerator:
    def __init__(
        self,
        world_description: WorldDescription = WorldDescription(),
    ) -> None:
        self.wd = world_description

        self.crop_models = None
        self.weed_models = None
        self.litter_models = None
        self.marker_models = None

    def gather_available_models(self) -> None:
        self.crop_models = to_gazebo_models(
            CROP_MODELS,
            self.wd.structure["params"]["crop_types"],
        )

        if self.wd.structure["params"]["weeds"] > 0:
            self.weed_models = to_gazebo_models(
                WEED_MODELS,
                self.wd.structure["params"]["weed_types"],
            )

        if self.wd.structure["params"]["litters"] > 0:
            self.litter_models = to_gazebo_models(
                LITTER_MODELS, self.wd.structure["params"]["litter_types"]
            )

        self.marker_models = MARKER_MODELS

    def render_matplotlib(self) -> None:
        # Segments
        for segment in self.segments:
            segment.render()

        # Plants
        plt.scatter(
            self.crop_placements[:, 0],
            self.crop_placements[:, 1],
            color="c",
            marker=".",
        )

    def plot_field(self):
        plt.plot()
        plt.figure(figsize=(10, 10))
        plt.gca().axis("equal")
        labels = []

        # crops
        plt.scatter(
            self.crop_placements[:, 0],
            self.crop_placements[:, 1],
            color="g",
            marker=".",
        )
        labels.append("crop plants")

        # weeds
        if self.weed_placements.shape[0] > 0:
            plt.scatter(
                self.weed_placements[:, 0],
                self.weed_placements[:, 1],
                color="r",
                marker=".",
                s=100,
                alpha=0.5,
            )
            labels.append("weeds")

        # litter
        if self.litter_placements.shape[0] > 0:
            plt.scatter(
                self.litter_placements[:, 0],
                self.litter_placements[:, 1],
                color="b",
                marker=".",
                s=100,
                alpha=0.5,
            )
            labels.append("litter")

        # start
        plt.scatter(
            self.start_loc[:, 0], self.start_loc[:, 1], color="g", marker=".", alpha=0
        )  # just to extend the axis of the plot
        plt.text(
            self.start_loc[:, 0],
            self.start_loc[:, 1],
            "START",
            bbox={"facecolor": "green", "alpha": 0.5, "pad": 10},
            ha="center",
            va="center",
        )

        # location markers
        if self.wd.structure["params"]["location_markers"]:
            plt.scatter(
                self.marker_a_loc[:, 0],
                self.marker_a_loc[:, 1],
                color="r",
                marker=".",
                alpha=0,
            )  # just to extend the axis of the plot
            plt.text(
                self.marker_a_loc[:, 0],
                self.marker_a_loc[:, 1],
                "A",
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
                ha="center",
                va="center",
            )

            plt.scatter(
                self.marker_b_loc[:, 0],
                self.marker_b_loc[:, 1],
                color="r",
                marker=".",
                alpha=0,
            )  # just to extend the axis of the plot
            plt.text(
                self.marker_b_loc[:, 0],
                self.marker_b_loc[:, 1],
                "B",
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
                ha="center",
                va="center",
            )

        # Axis
        plt.xlabel("x")
        plt.ylabel("y")

        plt.legend(labels)
        self.minimap = plt

    def generate(self, cache_dir: str) -> tuple[str, np.ndarray]:
        self.gather_available_models()
        self.chain_segments()
        self.center_plants()
        self.place_objects()
        self.generate_ground()
        self.fix_gazebo()
        self.render_to_template(cache_dir)
        self.plot_field()
        return self.sdf, self.heightmap

    def chain_segments(self) -> None:
        # Generate start points
        x_start = 0
        x_end = (self.wd.rows_count - 1) * self.wd.row_width

        current_p = np.array(
            [
                np.linspace(x_start, x_end, self.wd.rows_count),
                np.repeat(0, self.wd.rows_count),
            ]
        ).T
        current_dir = [0, 1]

        # Placement parameters
        offset = None
        self.crop_placements = [[] for _ in range(self.wd.rows_count)]

        # Chain all segments from the world description
        self.segments = []
        for segment in self.wd.structure["segments"]:
            if segment["type"] == "straight":
                seg = StraightSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["length"],
                    rng=self.wd.rng,
                )
            elif segment["type"] == "sincurved":
                seg = SinCurvedSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["offset"],
                    segment["length"],
                    segment["curve_dir"],
                    rng=self.wd.rng,
                )
            elif segment["type"] == "curved":
                seg = CurvedSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["radius"],
                    segment["curve_dir"],
                    segment["arc_measure"],
                    rng=self.wd.rng,
                )
            elif segment["type"] == "island":
                seg = IslandSegment(
                    current_p,
                    current_dir,
                    self.wd.structure["params"],
                    segment["radius"],
                    segment["island_model"],
                    segment["island_model_radius"],
                    segment["island_row"],
                    rng=self.wd.rng,
                )
            else:
                raise ValueError("Unknown segment type. [" + segment["type"] + "]")

            # Collect all plant placements
            seg_placements, offset = seg.placements(offset)
            for row, seg_row in zip(self.crop_placements, seg_placements):
                row.extend(seg_row)

            # Update current end points, direction and row length
            current_p, current_dir = seg.end()
            self.segments.append(seg)

        # generate holes in the maize field
        self.rows = []
        for index, row in zip(range(self.wd.rows_count), self.crop_placements):
            row = np.vstack(row)

            # generate indexes of the end of the hole
            probs = self.wd.rng.random(row.shape[0])
            probs = probs < self.wd.structure["params"]["hole_prob"][index]

            # iterate in reverse order, and remove plants in the holes
            i = probs.shape[0] - 1
            while i > 0:
                if probs[i]:
                    hole_size = self.wd.rng.integers(
                        1, self.wd.structure["params"]["hole_size_max"][index]
                    )
                    row = np.delete(row, slice(max(1, i - hole_size), i), axis=0)
                    i = i - hole_size

                i = i - 1
            self.rows.append(row)

        # Add bounden noise to placements
        bg = BoundedGaussian(
            -self.wd.structure["params"]["plant_placement_error_max"],
            self.wd.structure["params"]["plant_placement_error_max"],
            rng=self.wd.rng,
        )

        # TODO There is a better way to do this
        for i in range(len(self.rows)):
            new_row = []
            for x, y in self.rows[i]:
                x += bg.get()
                y += bg.get()
                new_row.append([x, y])

            self.rows[i] = np.array(new_row)

    # Because the heightmap must be square and has to have a side length of 2^n + 1
    # this means that we could have smaller maps, by centering the placements around 0,0
    def center_plants(self) -> None:
        self.crop_placements = np.vstack(self.rows)

        x_min = self.crop_placements[:, 0].min()
        y_min = self.crop_placements[:, 1].min()

        x_max = self.crop_placements[:, 0].max()
        y_max = self.crop_placements[:, 1].max()

        for i in range(len(self.rows)):
            self.rows[i] -= np.array([x_min, y_min])
            self.rows[i] -= np.array([x_max - x_min, y_max - y_min]) / 2

    # The function calculates the placements of the weed plants and
    def place_objects(self) -> None:
        def random_points_within(poly: Polygon, num_points: int) -> np.ndarray:
            min_x, min_y, max_x, max_y = poly.bounds

            points = []

            while len(points) < num_points:
                np_point = [
                    self.wd.rng.uniform(min_x, max_x),
                    self.wd.rng.uniform(min_y, max_y),
                ]
                random_point = Point(np_point)
                if random_point.within(poly):
                    points.append(np_point)

            return np.array(points)

        # Get outher boundary of the of the crops
        outer_plants = np.concatenate((self.rows[0], np.flipud(self.rows[-1])))
        self.field_poly = Polygon(outer_plants)

        # place x_nr of weeds within the field area
        if self.wd.structure["params"]["weeds"] > 0:
            self.weed_placements = random_points_within(
                self.field_poly, self.wd.structure["params"]["weeds"]
            )
            random_weed_models = self.wd.rng.choice(
                list(self.weed_models.values()),
                self.wd.structure["params"]["weeds"],
            )
        else:
            self.weed_placements = np.array([]).reshape(0, 2)
            random_weed_models = []

        # place y_nr of litter within the field area
        if self.wd.structure["params"]["litters"] > 0:
            self.litter_placements = random_points_within(
                self.field_poly, self.wd.structure["params"]["litters"]
            )
            random_litter_models = self.wd.rng.choice(
                list(self.litter_models.values()),
                self.wd.structure["params"]["litters"],
            )

        else:
            self.litter_placements = np.array([]).reshape(0, 2)
            random_litter_models = []

        # place start marker at the beginning of the field
        line = LineString([self.rows[0][0], self.rows[-1][0]])
        offset_start = line.parallel_offset(1, "right", join_style=2, mitre_limit=0.1)
        self.start_loc = np.array(
            [
                [
                    offset_start.interpolate(distance=0.375).xy[0][0],
                    offset_start.interpolate(distance=1).xy[1][0],
                ]
            ]
        )

        # place location markers at the desginated locations
        if self.wd.structure["params"]["location_markers"]:
            line = LineString([self.rows[0][0], self.rows[-1][0]])
            offset_a = line.parallel_offset(2.5, "right", join_style=2, mitre_limit=0.1)
            self.marker_a_loc = np.array(
                [[offset_a.centroid.xy[0][0], offset_a.centroid.xy[1][0]]]
            )

            line = LineString([self.rows[0][-1], self.rows[-1][-1]])
            offset_b = line.parallel_offset(2.5, "left", join_style=2, mitre_limit=0.1)
            self.marker_b_loc = np.array(
                [[offset_b.centroid.xy[0][0], offset_b.centroid.xy[1][0]]]
            )

            added_marker_models = [
                self.marker_models["location_marker_a"],
                self.marker_models["location_marker_b"],
            ]
        else:
            self.marker_a_loc = np.array([]).reshape(0, 2)
            self.marker_b_loc = np.array([]).reshape(0, 2)
            added_marker_models = []

        self.object_placements = np.concatenate(
            (
                self.weed_placements,
                self.litter_placements,
                self.marker_a_loc,
                self.marker_b_loc,
            )
        )

        self.object_types = [
            *random_weed_models,
            *random_litter_models,
            *added_marker_models,
        ]

    def generate_ground(self) -> None:
        ditch_depth = self.wd.structure["params"]["ground_ditch_depth"]
        ditch_distance = self.wd.structure["params"]["ground_headland"]

        self.crop_placements = np.vstack(self.rows)

        if self.object_placements.ndim == 2:
            self.placements = np.concatenate(
                (self.crop_placements, self.object_placements), axis=0
            )
        else:
            self.placements = self.crop_placements
        # Calculate image resolution
        metric_x_min = self.placements[:, 0].min()
        metric_x_max = self.placements[:, 0].max()
        metric_y_min = self.placements[:, 1].min()
        metric_y_max = self.placements[:, 1].max()

        metric_width = metric_x_max - metric_x_min + 2 * ditch_distance + 1
        metric_height = metric_y_max - metric_y_min + 2 * ditch_distance + 1

        min_resolution = self.wd.structure["params"][
            "ground_resolution"
        ]  # min resolution
        min_image_size = int(
            np.ceil(max(metric_width / min_resolution, metric_height / min_resolution))
        )
        # gazebo expects heightmap in format 2**n -1
        image_size = int(2 ** np.ceil(np.log2(min_image_size))) + 1

        self.resolution = min_resolution * (min_image_size / image_size)

        # Generate noise
        heightmap = np.zeros((image_size, image_size))

        n = 0
        while 2**n < image_size:
            heightmap += (
                cv2.resize(
                    self.wd.rng.random((image_size // 2**n, image_size // 2**n)),
                    (image_size, image_size),
                )
                * (n + 1) ** 2
            )
            n += 1

        # Normalize heightmap
        heightmap -= heightmap.min()
        heightmap /= heightmap.max()

        max_elevation = self.wd.structure["params"]["ground_elevation_max"]

        self.heightmap_elevation = ditch_depth + (max_elevation / 2)

        heightmap *= (max_elevation) / self.heightmap_elevation
        field_height = (ditch_depth - (max_elevation / 2)) / self.heightmap_elevation

        field_mask = np.zeros((image_size, image_size))

        offset = image_size // 2

        def metric_to_pixel(pos: int) -> int:
            return int(pos // self.resolution) + offset

        # Make plant placements flat and save the heights for the sdf renderer
        PLANT_FOOTPRINT = (2 * 0.02**2) ** 0.5
        flatspot_radius = int((PLANT_FOOTPRINT / 2) // self.resolution) + 2

        self.placements_ground_height = []
        for mx, my in self.placements:
            px = metric_to_pixel(mx)
            py = metric_to_pixel(my)

            height = heightmap[py, px]
            heightmap = cv2.circle(heightmap, (px, py), flatspot_radius, height, -1)
            self.placements_ground_height.append(
                (field_height + height) * self.heightmap_elevation
            )

        # create ditch around the crop field
        for mx, my in self.crop_placements:
            px = metric_to_pixel(mx)
            py = metric_to_pixel(my)

            field_mask = cv2.circle(
                field_mask, (px, py), int((ditch_distance) / self.resolution), 1, -1
            )

        blur_size = (int(0.2 / self.resolution) // 2) * 2 + 1
        field_mask = cv2.GaussianBlur(field_mask, (blur_size, blur_size), 0)

        heightmap += field_height * field_mask

        assert heightmap.max() <= 1
        assert heightmap.min() >= 0

        # Convert to grayscale
        self.heightmap = (255 * heightmap[::-1, :]).astype(np.uint8)

        self.metric_size = image_size * self.resolution
        # Calc heightmap position. Currently unused, overwritten in @ref fix_gazebo
        self.heightmap_position = [
            metric_x_min - 2 + 0.5 * self.metric_size,
            metric_y_min - 2 + 0.5 * self.metric_size,
        ]

    def fix_gazebo(self) -> None:
        # move the plants to the center of the flat circles
        self.crop_placements -= self.resolution / 2
        self.object_placements -= self.resolution / 2

        # set heightmap position to origin
        self.heightmap_position = [0, 0]

    def render_to_template(self, cache_dir: str) -> None:
        def into_dict(
            xy: np.ndarray,
            ground_height: float,
            radius: float,
            height: float,
            mass: float,
            model: GazeboModel,
            index: int,
            ghost: bool = False,
        ) -> dict[str, Any]:
            coordinate = dict()
            coordinate["type"] = model.model_name
            coordinate["name"] = f"{model.model_name}_{index:04d}"
            coordinate["static"] = str(model.static).lower()

            if ghost and model.ghostable:
                coordinate["ghost"] = ghost
                coordinate["custom_visual"] = model.get_model_visual()

            # Model mass
            inertia = dict()
            inertia["ixx"] = (mass * (3 * radius**2 + height**2)) / 12.0
            inertia["iyy"] = (mass * (3 * radius**2 + height**2)) / 12.0
            inertia["izz"] = (mass * radius**2) / 2.0
            coordinate["inertia"] = inertia
            coordinate["mass"] = mass

            # Model pose
            coordinate["x"] = model.default_x + xy[0]
            coordinate["y"] = model.default_y + xy[1]
            coordinate["z"] = model.default_z + ground_height
            coordinate["roll"] = model.default_roll
            coordinate["pitch"] = model.default_pitch
            coordinate["yaw"] = model.default_yaw
            if model.random_yaw:
                coordinate["yaw"] += self.wd.rng.random() * 2.0 * np.pi

            coordinate["radius"] = (
                radius
                + (2 * self.wd.rng.random() - 1)
                * self.wd.structure["params"]["plant_radius_noise"]
            )
            if coordinate["type"] == "cylinder":
                coordinate["height"] = height

            return coordinate

        # plant crops
        coordinates = [
            into_dict(
                plant,
                self.placements_ground_height[i],
                self.wd.structure["params"]["plant_radius"],
                self.wd.structure["params"]["plant_height_min"],
                self.wd.structure["params"]["plant_mass"],
                self.wd.rng.choice(list(self.crop_models.values())),
                i,
            )
            for i, plant in enumerate(self.crop_placements)
        ]

        # place objects
        object_coordinates = [
            into_dict(
                plant,
                self.placements_ground_height[i + len(self.crop_placements)],
                self.wd.structure["params"]["plant_radius"],
                self.wd.structure["params"]["plant_height_min"],
                self.wd.structure["params"]["plant_mass"],
                self.object_types[i],
                i,
                ghost=self.wd.structure["params"]["ghost_objects"],
            )
            for i, plant in enumerate(self.object_placements)
        ]

        coordinates.extend(object_coordinates)

        template = importlib.resources.read_text(__package__, "field.world.template")
        template = jinja2.Template(template)
        self.sdf = template.render(
            coordinates=coordinates,
            seed=self.wd.structure["params"]["seed"],
            heightmap={
                "size": self.metric_size,
                "pos": {
                    "x": self.heightmap_position[0],
                    "y": self.heightmap_position[1],
                },
                "max_elevation": self.wd.structure["params"]["ground_elevation_max"],
                "ditch_depth": self.wd.structure["params"]["ground_ditch_depth"],
                "total_height": self.heightmap_elevation,
                "cache_dir": cache_dir,
            },
        )
