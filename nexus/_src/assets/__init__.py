"""newton-assets: content-addressed asset hosting/resolution, with an S3 + CloudFront backend.

The runtime resolver is stdlib-only, with no Amazon Web Services (AWS) deps. The only thing in this
package is fetching a ``{name, sha256}`` asset to a verified local cache. Publishing, meaning the sha,
the upload target and the glb, is a prep step done out of band by
``scripts/assets/prepare_asset_upload.py``; uploads are manual.
"""

from .resolver import default_cache, fetch, resolve, sha256_file

__all__ = ["default_cache", "fetch", "resolve", "sha256_file"]
