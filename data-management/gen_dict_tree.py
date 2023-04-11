import os
import pathlib
import sys
from collections import deque

PIPE = "│"
ELBOW = "└──"
TEE = "├──"
PIPE_PREFIX = "│   "
SPACE_PREFIX = "    "


class DirectoryTree:
    def __init__(self, root_dir, dir_only=False, output_file=sys.stdout):
        self._output_file = output_file
        self._generator = _TreeGenerator(root_dir, dir_only)

    def generate(self):
        tree = self._generator.build_tree()
        if self._output_file != sys.stdout:
            # Wrap the tree in a markdown code block
            tree.appendleft("```")
            tree.append("```")
            self._output_file = open(
                self._output_file, mode="w", encoding="UTF-8"
            )
        with self._output_file as stream:
            for entry in tree:
                print(entry, file=stream)


class _TreeGenerator:
    def __init__(self, root_dir, dir_only=False):

        self._root_dir = pathlib.Path(root_dir)
        self._dir_only = dir_only
        self._tree = deque()

    def build_tree(self):
        self._tree_head()
        self._tree_body(self._root_dir)
        return self._tree

    def _tree_head(self):
        self._tree.append(f"{self._root_dir}{os.sep}")

    def _tree_body(self, directory, prefix=""):
        entries = self._prepare_entries(directory)
        last_index = len(entries) - 1
        for index, entry in enumerate(entries):
            connector = ELBOW if index == last_index else TEE
            if entry.is_dir():
                if index == 0:
                    self._tree.append(prefix + PIPE)
                self._add_directory(
                    entry, index, last_index, prefix, connector
                )
            else:
                self._add_file(entry, prefix, connector)

    def _prepare_entries(self, directory):
        entries = sorted(
            directory.iterdir(), key=lambda entry: str(entry)
        )
        if self._dir_only:
            return [entry for entry in entries if entry.is_dir()]
        return sorted(entries, key=lambda entry: entry.is_file())

    def _add_directory(
            self, directory, index, last_index, prefix, connector
    ):
        if directory.name in exp_name:
            return 0
        self._tree.append(f"{prefix}{connector} {directory.name}{os.sep}")
        if index != last_index:
            prefix += PIPE_PREFIX
        else:
            prefix += SPACE_PREFIX
        self._tree_body(
            directory=directory,
            prefix=prefix,
        )
        if prefix := prefix.rstrip():
            self._tree.append(prefix)

    def _add_file(self, file, prefix, connector):
        self._tree.append(f"{prefix}{connector}{file.name}")


if __name__ == '__main__':
    src = os.getcwd()  # r'C:\Users\it_admin\Desktop\mta-2021-1210\python
    src = r'D:\Downloads\gta_human'
    exp_name = ['env', 'jpeg']
    tree = DirectoryTree(src)
    tree.generate()
