"""FLAMEParamsEdit — combined merge-policy + user-adjustable FLAME params editor.

Replaces the previous `KaoLRMParamsToFLAMEParams` + `FLAMEParamsMerge` pair. This
node takes one required `FLAME_PARAMS` input (normally the KaoLRM output) and an
optional second input (normally SMIRK). When the second input is present, the
fixed merge policy from `.claude/plan/final-plan.md` §6 is applied first:

    shape       ← params          identity anchor (KaoLRM side)
    expression  ← params_override  if provided, else params
    pose[:, :3] ← params           global rotation
    pose[:, 3:] ← params_override  if provided, else params
    scale       ← params
    translation ← params
    fix_z_trans ← params (unless overridden via the combo)

After the merge, the user-facing sliders (strengths + offsets + fix_z override)
apply on top. This is the seed node for the FLAME Param Optimizer roadmap entry
— landmark-fit / photometric losses land later, but the slider UI lives here.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from .flame_params_wire import FLAME_PARAMS, validate_flame_params

FIX_Z_OPTIONS = ["inherit", "force_true", "force_false"]


def _apply_merge_policy(base: dict, override: dict) -> dict:
    merged_pose = torch.cat(
        [base["pose"][:, :3], override["pose"][:, 3:]],
        dim=1,
    )
    return {
        "shape": base["shape"],
        "expression": override["expression"],
        "pose": merged_pose,
        "scale": base["scale"],
        "translation": base["translation"],
        "fix_z_trans": bool(base["fix_z_trans"]),
    }


def _apply_edits(
    params: dict,
    *,
    shape_strength: float,
    expression_strength: float,
    jaw_strength: float,
    scale_multiplier: float,
    global_pose_offset_x: float,
    global_pose_offset_y: float,
    global_pose_offset_z: float,
    translation_offset_x: float,
    translation_offset_y: float,
    translation_offset_z: float,
    fix_z_trans_override: str,
) -> dict:
    shape = params["shape"] * float(shape_strength)
    expression = params["expression"] * float(expression_strength)

    pose = params["pose"].clone()
    pose[:, 3:] = pose[:, 3:] * float(jaw_strength)
    global_offset = torch.tensor(
        [float(global_pose_offset_x), float(global_pose_offset_y), float(global_pose_offset_z)],
        dtype=pose.dtype,
        device=pose.device,
    )
    pose[:, :3] = pose[:, :3] + global_offset

    scale = params["scale"] * float(scale_multiplier)

    translation = params["translation"] + torch.tensor(
        [float(translation_offset_x), float(translation_offset_y), float(translation_offset_z)],
        dtype=params["translation"].dtype,
        device=params["translation"].device,
    )

    if fix_z_trans_override == "force_true":
        fix_z = True
    elif fix_z_trans_override == "force_false":
        fix_z = False
    else:
        fix_z = bool(params["fix_z_trans"])

    return {
        "shape": shape.contiguous(),
        "expression": expression.contiguous(),
        "pose": pose.contiguous(),
        "scale": scale.contiguous(),
        "translation": translation.contiguous(),
        "fix_z_trans": fix_z,
    }


class FLAMEParamsEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FLAMEParamsEdit",
            display_name="FLAME Params Edit",
            category="KaoLRM",
            description=(
                "Merge a second FLAME_PARAMS (expression/jaw from SMIRK) into the first "
                "(identity from KaoLRM), then apply user-adjustable offsets and strengths. "
                "If params_override is omitted, the node is a pure editor on params."
            ),
            inputs=[
                FLAME_PARAMS.Input(
                    "params",
                    tooltip="Base FLAME_PARAMS — provides shape, global pose, scale, translation.",
                ),
                FLAME_PARAMS.Input(
                    "params_override",
                    optional=True,
                    tooltip=(
                        "Optional secondary FLAME_PARAMS (e.g. from SMIRKPredict). "
                        "When present, expression and jaw pose are swapped in per the "
                        "merge policy before edits apply."
                    ),
                ),
                io.Float.Input(
                    "shape_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip="Uniform scale on shape coefficients. 1.0 = unchanged, 0.0 = mean face.",
                ),
                io.Float.Input(
                    "expression_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip="Uniform scale on expression coefficients. 1.0 = unchanged, 0.0 = neutral.",
                ),
                io.Float.Input(
                    "jaw_strength",
                    default=1.0, min=0.0, max=1.5, step=0.05,
                    optional=True,
                    tooltip="Uniform scale on jaw rotation (pose[:, 3:]). 1.0 = unchanged, 0.0 = closed mouth.",
                ),
                io.Float.Input(
                    "scale_multiplier",
                    default=1.0, min=0.1, max=3.0, step=0.05,
                    optional=True,
                    tooltip="Multiplicative on scale.",
                ),
                io.Float.Input(
                    "global_pose_offset_x",
                    default=0.0, min=-1.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Additive offset on global pose[:, 0] (radians).",
                ),
                io.Float.Input(
                    "global_pose_offset_y",
                    default=0.0, min=-1.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Additive offset on global pose[:, 1] (radians).",
                ),
                io.Float.Input(
                    "global_pose_offset_z",
                    default=0.0, min=-1.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Additive offset on global pose[:, 2] (radians).",
                ),
                io.Float.Input(
                    "translation_offset_x",
                    default=0.0, min=-1.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Additive offset on translation x.",
                ),
                io.Float.Input(
                    "translation_offset_y",
                    default=0.0, min=-1.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Additive offset on translation y.",
                ),
                io.Float.Input(
                    "translation_offset_z",
                    default=0.0, min=-1.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Additive offset on translation z (ignored if fix_z_trans resolves True).",
                ),
                io.Combo.Input(
                    "fix_z_trans_override",
                    options=FIX_Z_OPTIONS,
                    default="inherit",
                    optional=True,
                    tooltip=(
                        "'inherit' keeps the base flag. 'force_true' zeros translation z downstream "
                        "(KaoLRM mono default). 'force_false' applies translation z as authored."
                    ),
                ),
            ],
            outputs=[FLAME_PARAMS.Output(display_name="flame_params")],
        )

    @classmethod
    def execute(
        cls,
        params,
        params_override=None,
        shape_strength: float = 1.0,
        expression_strength: float = 1.0,
        jaw_strength: float = 1.0,
        scale_multiplier: float = 1.0,
        global_pose_offset_x: float = 0.0,
        global_pose_offset_y: float = 0.0,
        global_pose_offset_z: float = 0.0,
        translation_offset_x: float = 0.0,
        translation_offset_y: float = 0.0,
        translation_offset_z: float = 0.0,
        fix_z_trans_override: str = "inherit",
    ):
        validate_flame_params(params, source="params")

        if params_override is not None:
            validate_flame_params(params_override, source="params_override")
            b_base = params["shape"].shape[0]
            b_over = params_override["shape"].shape[0]
            if b_base != b_over:
                raise RuntimeError(
                    f"FLAMEParamsEdit: batch mismatch — params B={b_base}, params_override B={b_over}."
                )
            working = _apply_merge_policy(params, params_override)
        else:
            working = {
                "shape": params["shape"],
                "expression": params["expression"],
                "pose": params["pose"],
                "scale": params["scale"],
                "translation": params["translation"],
                "fix_z_trans": bool(params["fix_z_trans"]),
            }

        if fix_z_trans_override not in FIX_Z_OPTIONS:
            raise RuntimeError(
                f"FLAMEParamsEdit: fix_z_trans_override must be one of {FIX_Z_OPTIONS}, "
                f"got {fix_z_trans_override!r}."
            )

        edited = _apply_edits(
            working,
            shape_strength=shape_strength,
            expression_strength=expression_strength,
            jaw_strength=jaw_strength,
            scale_multiplier=scale_multiplier,
            global_pose_offset_x=global_pose_offset_x,
            global_pose_offset_y=global_pose_offset_y,
            global_pose_offset_z=global_pose_offset_z,
            translation_offset_x=translation_offset_x,
            translation_offset_y=translation_offset_y,
            translation_offset_z=translation_offset_z,
            fix_z_trans_override=fix_z_trans_override,
        )
        return io.NodeOutput(edited)
