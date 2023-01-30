import threading
from typing import Callable, Optional, cast

from rich.style import Style
from rich.text import Text
from textual.message import Message, MessageTarget
from textual.reactive import reactive
from textual.widgets._tree import TOGGLE_STYLE, Tree, TreeNode

from summ.shared.chain import DPrinter

Node = DPrinter.Entry


class OutputTree(Tree[Node]):
    last_node: TreeNode[Node]
    unregister: Optional[Callable]

    displayed_node: reactive[Optional[TreeNode[Node]]] = reactive(
        None, init=False, repaint=False
    )

    class RecordOutput(Message):
        def __init__(self, sender: MessageTarget, text: str) -> None:
            self.text = text
            super().__init__(sender)

    def __init__(self, question: str, **kwargs) -> None:
        self.outputs: dict[Node, str] = {}

        root = Node(color="green", indent=-1, text=question, title="Question", thread=0)
        super().__init__(root.title, root, **kwargs)
        self.last_node = self.root
        self.root.expand()

    def _update_output(self, evt: RecordOutput):
        self.parent.parent.on_output_tree_record_output(evt)

    def _on_node(self, entry: Node):
        self.log(entry)

        if not entry.color:
            self.outputs[self.last_node.data] = entry.text
            # LOL big hack
            self._update_output(self.RecordOutput(self, entry.text))
            return

        parent = self.last_node
        while parent.data.indent >= entry.indent:
            parent = parent.parent

        entry = entry.copy(update={"text": entry.text.replace("\n", " ").strip()})
        self.last_node = parent.add(entry.title, entry, expand=True)
        self.scroll_to_node(self.last_node)

    def auditor(self):
        app = self.app

        def cb(*args):
            if app._thread_id == threading.get_ident():
                self._on_node(*args)
            else:
                app.call_from_thread(self._on_node, *args)

        return cb

    def on_mount(self) -> None:
        self.unregister = DPrinter.register_auditor(self.auditor())

    def remove(self):
        if self.unregister:
            self.unregister()
            self.unregister = None
        super().remove()

    def _toggle_node(self, node: TreeNode[Node]) -> None:
        if not (output := self.outputs.get(node.data)) or node == self.displayed_node:
            return
        self.displayed_node = node
        self._update_output(self.RecordOutput(self, output))

    def render_label(self, node: TreeNode[Node], base_style: Style, style: Style):
        prefix = node._label.copy().append(": ")
        prefix.stylize(style)

        data = cast(Node, node.data)

        if self.outputs.get(data):
            arrow = (
                "▼ " if node == self.displayed_node else "▶ ",
                base_style + TOGGLE_STYLE,
            )
        else:
            arrow = ("", base_style)

        if data.color:
            label = (data.text, Style(color=data.color))
        else:
            label = data.text
        return Text.assemble(arrow, prefix, label, style=base_style)
