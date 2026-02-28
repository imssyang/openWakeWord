# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/pytorch/pytorch/blob/v2.3.1/caffe2/python/net_drawer.py
#
# https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#   $ python onnx_net_drawer.py --input <my.onnx> --output my_output.dot --embed_docstring
#   $ dot -Tsvg my_output.dot -o my_output.svg
from __future__ import annotations
import argparse
import json
import onnx
import os
import pydot
from collections import defaultdict
from collections.abc import Callable
from typing import Any


_NodeProducer = Callable[[onnx.NodeProto, int], pydot.Node]


class OnnxDrawer:
    def __init__(
        self,
        model_path: str,
        rankdir: str,
        embed_docstring: bool,
    ):
        self.op_style = {
            "shape": "box",
            "color": "#0F9D58",
            "style": "filled",
            "fontcolor": "#FFFFFF",
        }
        self.blob_style = {"shape": "octagon"}
        self.model = self._load_model(model_path)
        self.pydot = self._get_pydot_graph(
            self.model.graph,
            name=self.model.graph.name,
            rankdir=rankdir,
            embed_docstring=embed_docstring,
        )

    def save(self, output_path: str):
        _, ext = os.path.splitext(output_path)
        if ext == ".svg":
            self.pydot.write_svg(output_path)
        elif ext == ".dot":
            self.pydot.write_dot(output_path)
        else:
            raise RuntimeError(f"Invalid {output_path}")

    def _load_model(self, model_path: str):
        model = onnx.ModelProto()
        with open(model_path, "rb") as fid:
            content = fid.read()
            model.ParseFromString(content)
        return model

    def _get_pydot_graph(
        self,
        graph: onnx.GraphProto,
        name: str,
        rankdir: str,
        embed_docstring: bool,
    ) -> pydot.Dot:
        node_producer = self._get_node_producer(embed_docstring=embed_docstring, **self.op_style)
        pydot_graph = pydot.Dot(name, rankdir=rankdir)
        pydot_nodes: dict[str, pydot.Node] = {}
        pydot_node_counts: dict[str, int] = defaultdict(int)
        for op_id, op in enumerate(graph.node):
            op_node = node_producer(op, op_id)
            pydot_graph.add_node(op_node)
            for input_name in op.input:
                if input_name not in pydot_nodes:
                    input_node = pydot.Node(
                        self._escape_label(input_name + str(pydot_node_counts[input_name])),
                        label=self._escape_label(input_name),
                        **self.blob_style,
                    )
                    pydot_nodes[input_name] = input_node
                else:
                    input_node = pydot_nodes[input_name]
                pydot_graph.add_node(input_node)
                pydot_graph.add_edge(pydot.Edge(input_node, op_node))
            for output_name in op.output:
                if output_name in pydot_nodes:
                    pydot_node_counts[output_name] += 1
                output_node = pydot.Node(
                    self._escape_label(output_name + str(pydot_node_counts[output_name])),
                    label=self._escape_label(output_name),
                    **self.blob_style,
                )
                pydot_nodes[output_name] = output_node
                pydot_graph.add_node(output_node)
                pydot_graph.add_edge(pydot.Edge(op_node, output_node))
        return pydot_graph

    def _get_node_producer(
        self,
        embed_docstring: bool = False,
        **kwargs: Any
    ) -> _NodeProducer:
        def really_get_op_node(op: onnx.NodeProto, op_id: int) -> pydot.Node:
            if op.name:
                node_name = f"{op.name}/{op.op_type} (op#{op_id})"
            else:
                node_name = f"{op.op_type} (op#{op_id})"
            for i, input_ in enumerate(op.input):
                node_name += "\n input" + str(i) + " " + input_
            for i, output in enumerate(op.output):
                node_name += "\n output" + str(i) + " " + output
            node = pydot.Node(node_name, **kwargs)
            if embed_docstring:
                url = self._form_and_sanitize_docstring(op.doc_string)
                node.set_URL(url)
            return node
        return really_get_op_node

    def _form_and_sanitize_docstring(self, s: str) -> str:
        url = "javascript:alert("
        url += self._escape_label(s).replace('"', "'").replace("<", "").replace(">", "")
        url += ")"
        return url

    def _escape_label(self, name: str) -> str:
        # json.dumps is poor man's escaping
        return json.dumps(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ONNX graph to DOT / SVG using pydot")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input ONNX model file (.onnx).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file (.dot or .svg).",
    )
    parser.add_argument(
        "--rankdir",
        type=str,
        choices=["LR", "TB", "BT", "RL"],
        default="TB",
        help="Graph layout direction (default: TB).",
    )
    parser.add_argument(
        "--embed_docstring",
        action="store_true",
        help="Embed docstring as JS alert (SVG only).",
    )
    args = parser.parse_args()
    drawer = OnnxDrawer(
        model_path=args.input,
        rankdir=args.rankdir,
        embed_docstring=args.embed_docstring,
    )
    drawer.save(args.output)


if __name__ == "__main__":
    main()
