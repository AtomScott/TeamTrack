import git
from pathlib import Path


def get_git_root():
    """Get the root of the git repository."""
    git_repo = git.Repo(__file__, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)
