"""Check that the zone disambiguation switches reach the FoundationPose init node.

The launch description is evaluated with a LaunchContext, and the foundationpose_init
shell command is resolved from its substitutions. The init node is checked by AST and
by running its validation statements alone, so the YOLO and FoundationPose weights
are never loaded.
"""

import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = REPO_ROOT / "ros2/rcia_energy_trace_ros/launch/full_energy_tracking.launch.py"
NODE_PATH = REPO_ROOT / "ros2/rcia_energy_trace_ros/scripts/foundationpose_init_node.py"
EVAL_PATH = REPO_ROOT / "scripts/foundationpose_eval.py"

sys.path.insert(0, str(REPO_ROOT))

import scripts.foundationpose_eval as foundationpose_eval  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_foundationpose_cmd(overrides=None):
    """Return the shell string that the launch file passes to foundationpose_init."""
    module = load_module(LAUNCH_PATH, "full_energy_tracking_launch_zone_under_test")
    description = module.generate_launch_description()
    context = LaunchContext()
    for name, value in (overrides or {}).items():
        context.launch_configurations[name] = str(value)
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    processes = [
        entity
        for entity in description.entities
        if isinstance(entity, ExecuteProcess)
        and not isinstance(entity, Node)
        and perform_substitutions(context, entity.name) == "foundationpose_init"
    ]
    if len(processes) != 1:
        raise ValueError(f"expected one foundationpose_init process, found {len(processes)}")
    # Executable 把 cmd 保存为子列表，第三项是交给 bash -lc 的完整命令字符串。
    cmd = processes[0]._ExecuteLocal__process_description._Executable__cmd
    return perform_substitutions(context, cmd[2])


NON_LITERAL = object()


def _literal_or_marker(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return NON_LITERAL


def find_flag_values(command, flag):
    return re.findall(rf"(?:^|\s)-p {re.escape(flag)}:=(\S+)(?=\s|$)", command)



class FoundationPoseInitLaunchTest(unittest.TestCase):
    def test_zone_disambiguation_defaults_bind_to_foundationpose_init(self):
        command = resolve_foundationpose_cmd()

        self.assertEqual(find_flag_values(command, "zone_disambiguation_enabled"), ["false"])
        self.assertEqual(find_flag_values(command, "zone_disambiguation_radius_m"), ["0.14"])

    def test_zone_disambiguation_overrides_propagate_to_foundationpose_init(self):
        command = resolve_foundationpose_cmd({
            "zone_disambiguation_enabled": "true",
            "zone_disambiguation_radius_m": "0.15",
        })

        self.assertEqual(find_flag_values(command, "zone_disambiguation_enabled"), ["true"])
        self.assertEqual(find_flag_values(command, "zone_disambiguation_radius_m"), ["0.15"])


class FoundationPoseEvalParserTest(unittest.TestCase):
    def parse_run(self, *extra):
        parser = foundationpose_eval.build_parser()
        return parser.parse_args(["run", "--frames_root", "frames", "--out", "out.json", *extra])

    def test_zone_disambiguation_defaults(self):
        args = self.parse_run()

        self.assertIs(args.zone_disambiguation_enabled, False)
        self.assertEqual(args.zone_disambiguation_radius_m, 0.14)
        self.assertIsInstance(args.zone_disambiguation_radius_m, float)

    def test_zone_disambiguation_flags_are_parsed(self):
        args = self.parse_run("--zone_disambiguation_enabled", "--zone_disambiguation_radius_m", "0.15")

        self.assertIs(args.zone_disambiguation_enabled, True)
        self.assertEqual(args.zone_disambiguation_radius_m, 0.15)


class FoundationPoseInitNodeParameterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = NODE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.init_method = cls._find_init()

    @classmethod
    def _find_init(cls):
        for node in cls.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "FoundationPoseInitNode":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        return item
        raise AssertionError("FoundationPoseInitNode.__init__ not found")

    def declared_defaults(self):
        declared = {}
        for node in ast.walk(self.init_method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "declare_parameter"
                and node.args
            ):
                name = ast.literal_eval(node.args[0])
                declared[name] = _literal_or_marker(node.args[1]) if len(node.args) > 1 else None
        return declared

    def test_zone_disambiguation_parameters_are_declared_with_defaults(self):
        declared = self.declared_defaults()

        self.assertIn("zone_disambiguation_enabled", declared)
        self.assertIn("zone_disambiguation_radius_m", declared)
        self.assertIs(declared["zone_disambiguation_enabled"], False)
        self.assertEqual(declared["zone_disambiguation_radius_m"], 0.14)
        self.assertIsInstance(declared["zone_disambiguation_radius_m"], float)

    def test_zone_radius_validation_rejects_non_positive_and_non_finite(self):
        body = self.init_method.body
        start = next(
            i for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "zone_radius_m"
        )
        check = body[start + 1]
        self.assertIsInstance(check, ast.If)
        raises = [n for n in ast.walk(check) if isinstance(n, ast.Raise)]
        self.assertTrue(raises, "radius check must raise")
        self.assertEqual(ast.unparse(raises[0].exc.func), "ValueError")

        # 只执行 __init__ 中读取并校验半径的两条语句，避免创建节点和加载权重。
        module = ast.Module(body=[body[start], check], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, str(NODE_PATH), "exec")

        class FakeParam:
            def __init__(self, value):
                self.value = value

        class FakeNode:
            def __init__(self, radius):
                self.radius = radius
                self.requested = []

            def get_parameter(self, name):
                self.requested.append(name)
                return FakeParam(self.radius)

        def run_check(radius):
            node = FakeNode(radius)
            scope = {"np": np, "self": node}
            exec(code, scope)
            self.assertEqual(node.requested, ["zone_disambiguation_radius_m"])
            return scope["zone_radius_m"]

        self.assertEqual(run_check(0.14), 0.14)
        self.assertEqual(run_check(0.15), 0.15)
        for bad in (0.0, -0.1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(radius=bad):
                with self.assertRaises(ValueError):
                    run_check(bad)


if __name__ == "__main__":
    unittest.main()
