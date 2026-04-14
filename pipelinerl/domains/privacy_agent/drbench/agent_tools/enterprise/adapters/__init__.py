"""Enterprise service adapters"""
from .nextcloud import NextcloudAdapter
from .mattermost import MattermostAdapter
from .filebrowser import FileBrowserAdapter
from .email import EmailAdapter

__all__ = ["NextcloudAdapter", "MattermostAdapter", "FileBrowserAdapter", "EmailAdapter"]