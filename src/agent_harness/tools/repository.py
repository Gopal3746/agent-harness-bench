from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
)


class ToolExecutionError(RuntimeError):
    """An expected failure while executing a coding-agent tool."""


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """One repository search result."""

    path: str
    line_number: int
    text: str


class RepositoryTools:
    """Read-only tools scoped to one repository workspace."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = 1_000_000,
        max_read_lines: int = 400,
    ) -> None:
        resolved_root = root.expanduser().resolve()

        if not resolved_root.is_dir():
            raise FileNotFoundError(
                f"repository root does not exist: {resolved_root}"
            )

        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")

        if max_read_lines < 1:
            raise ValueError("max_read_lines must be positive")

        self.root = resolved_root
        self.max_file_bytes = max_file_bytes
        self.max_read_lines = max_read_lines

    def list_files(
        self,
        path: str | Path = ".",
        *,
        max_results: int = 200,
    ) -> tuple[str, ...]:
        """List repository files beneath a relative path."""

        if max_results < 1:
            raise ToolExecutionError("max_results must be positive")

        target = self._resolve(path)

        if not target.exists():
            raise ToolExecutionError(f"path does not exist: {path}")

        if not target.is_file() and not target.is_dir():
            raise ToolExecutionError(
                f"path is not a file or directory: {path}"
            )

        files = sorted(
            self._relative(file_path)
            for file_path in self._iter_files(target)
        )

        return tuple(files[:max_results])

    def read_file(
        self,
        path: str | Path,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        """Read a bounded range of lines from a UTF-8 text file."""

        if start_line < 1:
            raise ToolExecutionError("start_line must be at least 1")

        requested_end = (
            end_line
            if end_line is not None
            else start_line + self.max_read_lines - 1
        )

        if requested_end < start_line:
            raise ToolExecutionError(
                "end_line must be greater than or equal to start_line"
            )

        requested_lines = requested_end - start_line + 1
        if requested_lines > self.max_read_lines:
            raise ToolExecutionError(
                f"cannot read more than {self.max_read_lines} lines"
            )

        target = self._resolve(path)

        if not target.exists():
            raise ToolExecutionError(f"file does not exist: {path}")

        if not target.is_file():
            raise ToolExecutionError(f"path is not a file: {path}")

        if target.stat().st_size > self.max_file_bytes:
            raise ToolExecutionError(
                f"file exceeds {self.max_file_bytes} byte limit: {path}"
            )

        try:
            lines = target.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
        except UnicodeDecodeError as error:
            raise ToolExecutionError(
                f"file is not valid UTF-8 text: {path}"
            ) from error

        if not lines:
            return ""

        if start_line > len(lines):
            raise ToolExecutionError(
                f"start_line {start_line} exceeds file length "
                f"{len(lines)}"
            )

        return "".join(lines[start_line - 1 : requested_end])

    def search_repo(
        self,
        query: str,
        path: str | Path = ".",
        *,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[SearchMatch, ...]:
        """Search repository text files for a literal string."""

        if not query:
            raise ToolExecutionError("query must not be empty")

        if max_results < 1:
            raise ToolExecutionError("max_results must be positive")

        target = self._resolve(path)

        if not target.exists():
            raise ToolExecutionError(f"path does not exist: {path}")

        expected = query if case_sensitive else query.casefold()
        matches: list[SearchMatch] = []

        for file_path in self._iter_files(target):
            if file_path.stat().st_size > self.max_file_bytes:
                continue

            try:
                lines = file_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except (UnicodeDecodeError, OSError):
                continue

            for line_number, line in enumerate(lines, start=1):
                candidate = line if case_sensitive else line.casefold()

                if expected not in candidate:
                    continue

                matches.append(
                    SearchMatch(
                        path=self._relative(file_path),
                        line_number=line_number,
                        text=line[:500],
                    )
                )

                if len(matches) >= max_results:
                    return tuple(matches)

        return tuple(matches)

    def _resolve(self, path: str | Path) -> Path:
        supplied_path = Path(path)

        if supplied_path.is_absolute():
            raise ToolExecutionError(
                "absolute paths are not allowed"
            )

        resolved = (self.root / supplied_path).resolve()

        if not resolved.is_relative_to(self.root):
            raise ToolExecutionError(
                "path escapes the repository workspace"
            )

        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _iter_files(self, target: Path) -> Iterator[Path]:
        if target.is_file():
            yield target
            return

        for candidate in target.rglob("*"):
            relative_parts = candidate.relative_to(self.root).parts

            if any(
                part in _IGNORED_DIRECTORY_NAMES
                for part in relative_parts
            ):
                continue

            if candidate.is_file() and not candidate.is_symlink():
                yield candidate
