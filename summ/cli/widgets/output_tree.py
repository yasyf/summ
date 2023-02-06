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
    unregister: Optional[Callable]

    displayed_node: reactive[Optional[TreeNode[Node]]] = reactive(
        None, init=False, repaint=False
    )
    last_node: reactive[TreeNode[Node]] = reactive(None, init=False)
    question: reactive[str] = reactive("", init=False)
    title: reactive[str] = reactive("Question", init=False)

    class RecordOutput(Message):
        def __init__(self, sender: MessageTarget, text: str) -> None:
            self.text = text
            super().__init__(sender)

    def __init__(self, question: str, **kwargs) -> None:
        self.outputs: dict[Node, str] = {}
        self.nodes: dict[Node, TreeNode[Node]] = {}

        root = Node(
            color="green",
            indent=-1,
            text=question,
            title=self.title,
            thread=DPrinter.main_thread,
            parent=None,
        )
        super().__init__(root.title, root, **kwargs)

        self.question = question
        self.last_node = self.root
        self.root.expand()

    def watch_title(self, title: str):
        self.root.data = self.root.data.copy(update={"title": title})
        self.root.set_label(title)
        self.root._reset()
        self.root._tree._invalidate()

    def watch_question(self, question: str):
        self.root.data = self.root.data.copy(update={"text": question})
        self.root._reset()
        self.root._tree._invalidate()

    def watch_last_node(self, last_node: TreeNode[Node]):
        self.scroll_to_node(last_node)

    def _update_output(self, evt: RecordOutput):
        # LOL big hack
        try:
            self.parent.parent.on_output_tree_record_output(evt)
        except AttributeError:
            # We're in shutdown
            pass

    def _attach_output(self, entry: Node, parent: TreeNode[Node]):
        while parent.data in self.outputs:
            if not parent.parent:
                return
            parent = parent.parent
        self.outputs[parent.data] = entry.text

    def _on_node(self, entry: Node):
        self.log(entry)

        if entry.parent:
            parent = self.nodes.get(entry.parent)
        else:
            parent = self.last_node

        if not parent:
            self.log("No parent", entry)
            return

        cleaned = entry.copy(update={"text": entry.text.replace("\n", " ").strip()})
        normalized = entry.copy(update={"parent": None})

        if not entry.color:
            self._attach_output(entry, parent)
            self._update_output(self.RecordOutput(self, entry.text))
            self.nodes[normalized] = parent
            return

        while parent.parent and parent.data.indent >= entry.indent:
            parent = parent.parent

        self.last_node = parent.add(cleaned.title, cleaned, expand=True)
        self.nodes[normalized] = self.last_node

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
        data = cast(Node, node.data)

        prefix = node._label.copy().append(": ") if node._label else Text()
        style = style + Style(color=data.color or "white")
        prefix.stylize(style)

        text = data.text if node._label else (data.text, style)

        if self.outputs.get(data):
            arrow = (
                "▼ " if node == self.displayed_node else "▶ ",
                base_style + TOGGLE_STYLE,
            )
        else:
            arrow = ("", base_style)

        return Text.assemble(arrow, prefix, text, style=base_style)
